"""Tools to deal with sequential data pipelines.

Author: Basil Kraft
"""

from __future__ import annotations

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Union, Tuple, Dict, Any, Optional
import inspect
import re


def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
    r"""Scans the class signature and returns argument names, types and default values.
    github.com/PyTorchLightning/pytorch-lightning/tree/8c0ea92af237542f5b36ae684543f871da829379/pytorch_lightning

    Returns:
        List with tuples of 3 values:
        (argument name, set with argument types, argument default value).
    """
    cls_default_params = inspect.signature(cls).parameters
    name_type_default = []
    for arg in cls_default_params:
        arg_type = cls_default_params[arg].annotation
        arg_default = cls_default_params[arg].default
        try:
            arg_types = tuple(arg_type.__args__)
        except AttributeError:
            arg_types = (arg_type, )

        name_type_default.append((arg, arg_types, arg_default))

    return name_type_default


def get_init_arguments(cls) -> List[str]:
    """Scans the class signature and returns argument names.
    github.com/PyTorchLightning/pytorch-lightning/tree/8c0ea92af237542f5b36ae684543f871da829379/pytorch_lightning

    Returns:
        List with argument names.
    """
    return [arg[0] for arg in get_init_arguments_and_types(cls)]


class SeqScheme(object):
    r"""Training strategies for sequential xarray data, deals with missing values.

    This class provides training strategies for multivariate sequential datasets (with features, `f` and targets, `t`)
    with options to account for missing data, different training strategies, and prediction modes.

    TODO: provide mask?

    Prediction scheme
    -----------------
    The prediction scheme if defined by `f_window_size`, `t_window_size`, and `predict_shift`. Some examples, with
    `-` = missing, `o` = observed, `f` = features, `t` = targets, each example shows two steps.

::

    ----------------------------------------------------
    ONE-TO-ONE <== DEFAULT
    ----------------------------------------------------
       f_window_size=1, t_window_size=1, predict_shift=0
        t : | - o o o o - |         Y : | - o o o o - |
                ^                             ^
                |                             |
                =                             =
        f : | - o o o o - |         X : | - o o o o - |
    ----------------------------------------------------
    MANY-TO-ONE
    ----------------------------------------------------
       f_window_size=2, t_window_size=1, predict_shift=0
        t : | - o o o o - |         Y : | - o o o o - |
                  ^                             ^
                  |                             |
                = =                           = =
        f : | - o o o o - |         X : | - o o o o - |
    ----------------------------------------------------
    MANY-TO-ONE | predict 1 ahead
    ----------------------------------------------------
       f_window_size=2, t_window_size=1, predict_shift=1
        t : | - o o o o - |         Y : | - o o o o - |
                    ^                             ^
                   /                             /
                = =                           = =
        f : | - o o o o - |         X : | - o o o o - |
    ----------------------------------------------------
    MANY-TO-ONE | use past and future
    ----------------------------------------------------
       f_window_size=3, t_window_size=1, predict_shift=-1
        t : | - o o o o - |         Y : | - o o o o - |
                  ^                             ^
                  |                             |
                = = =                         = = =
        f : | - o o o o - |         X : | - o o o o - |
    ----------------------------------------------------
    MANY-TO-MANY
    ----------------------------------------------------
       f_window_size=2, t_window_size=2, predict_shift=0
        t : | - o o o o - |         Y : | - o o o o - |
                ^ ^                           ^ ^
                | |                           | |
                = =                           = =
        f : | - o o o o - |         X : | - o o o o - |

    Irregular frequencies
    ---------------------
    With irregular frequency units as argument `f_window_size` or `t_window_size` (years: 'Y' and months: 'M') the
    targets always covers the largest possible time range (e.g., 31 days '1M' and 366 days for 'Y') and the features
    cover the shortest possible range. For example, with `f_window_size=2M` and `t_window_size=1M`, the target covers
    the exact range (t_days) and the features cover 59 days (28 + 30) - t_days, i.e., the shortest possible
    combination of two months.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset, variables from args `features` and `targets` must be present.
    features : List of str or str
        The features (herein: `f`).
    targets : List of str or str
        The targets (herein: `t`).
    f_window_size : Union[int >= 1, str]
        The feature window size along the `seq_dim` (e.g., `time`), i.e., how many steps in a given dimension the
        features must be present. The default (1) is a special case, where no antecedent sequence elements are
        used ('instantaneous model'). If the argument is a string it is interpreted as a pandas-like offset (see
        `pandas offset-aliases`), such as '1Y' for one year, or '2M' for two month. Only works if `seq_dim` is a time
        axis. The argument is used to find instances of the frequencies that satisfy the conditions given by
        `f_allow_miss` and `f_require_all`, i.e., if `f_allow_miss==True`, only some values must be present in each
        frequency unit, etc. Note that only 'Y' (year), 'M' (month), 'W' (week), 'D' (day), and `H` (hour) are
        supported currently. See `Prediction scheme` for more information. If -1 is passed, the window size will be
        inferred from the sequence length, i.e., the window has the length of the sequence.
    t_window_size : Union[int >= 1, str]
        The target window size along the `seq_dim` (e.g., `time`), i.e., how many steps in a given dimension the
        targets must be present. The default (1) is the most common case, where 1 value is predicted.
        See `t_window_size` for supported frequencies and `Prediction scheme` for more information. If -1 is passed,
        the window size will be
        inferred from the sequence length, i.e., the window has the length of the sequence.
    predict_shift : int >= 0
        An integer indicating the shift of the target compared to the features. The default, 0, means that no shift
        is done, positive values indicate that the prediction is done n steps into the future.
        See `Prediction scheme` for more information.
    f_frac: float [0.0, 1.0] or int [0, t_window_size]
        The fraction (if float) or amount (if int) of values that must be present in `f_window_size`. E.g., with 0.5,
        50 % of the values must be present in the window. Default is 1.0, meaning that all values must be present, 0.0
        means that no values are required. For integers, the value is interpreted as absolute number of values that
        are required.
    t_frac: float [0.0, 1.0] or int [0, t_window_size]
        Same as `f_frac`, but for the target. Default is 1.0.
    f_require_all : bool
        If `True` (default), the features are masked if ANY feature is missing and else if ALL features are missing.
        In other words, if you want *at least one* feature to be present, set `False`, if you want *all* features to
        be present, set `True`.
    t_allow_miss : bool
        If `True` (default), the targets are masked if ANY targets is missing and else if ALL features are missing.
        In other words, if you want *at least* one target to be present, set `False`, if you want *all* targets to
        be present, set `True`.
    f_is_qc: bool
        If `True` (default), the features are interpreted as quality control data, meaning that values of `1` are
        `valid`, everything else including NaN is interpreted as `missing`. Otherwise (`False`), finite values are
        interpreted as `valid`, which excludes NaN and inf.
    t_is_qc: bool
        Same as `f_is_qc` with `True` as default, but for targets.
    seq_dim : str
        The sequence dimension, default is `time`. Must be present in `ds`.
    """
    def __init__(
            self,
            ds: xr.Dataset,
            features: Union[List[str], str],
            targets: Union[List[str], str],
            f_window_size: Union[int, str] = 1,
            t_window_size: Union[int, str] = 1,
            predict_shift: int = 0,
            f_frac: Union[float, int] = 1.0,
            t_frac: Union[float, int] = 1.0,
            f_require_all: bool = True,
            t_require_all: bool = True,
            f_is_qc: bool = True,
            t_is_qc: bool = True,
            seq_dim: str = 'time') -> None:

        self.features = [features] if isinstance(features, str) else features
        self.targets = [targets] if isinstance(targets, str) else targets
        self.predict_shift = predict_shift
        self.f_frac = f_frac
        self.t_frac = t_frac
        self.f_require_all = f_require_all
        self.t_require_all = t_require_all
        self.f_is_qc = f_is_qc
        self.t_is_qc = t_is_qc
        self.seq_dim = seq_dim
        self.seq_data = ds[self.seq_dim]

        if seq_dim not in ds.dims:
            raise ValueError(
                f'argument seq_dim=`{seq_dim}` is not a valid dimension of `ds`. Use one of {list(ds.dims)}.'
            )

        seq_len = len(ds[seq_dim])
        for k, v in [['f_window_size', f_window_size], ['t_window_size', t_window_size]]:
            if not isinstance(v, str):
                if v == -1:
                    v = seq_len
                elif v < 1:
                    raise ValueError(
                        f'argument `{k}` cannot be < 1, is {v}.'
                    )
                elif v > seq_len:
                    raise ValueError(
                        f'argument `{k}` cannot be > seq_len={seq_len}, is {v}.'
                    )

        if isinstance(f_window_size, str) or isinstance(t_window_size, str):
            if not xr.core.dtypes.is_datetime_like(ds[seq_dim]):
                raise ValueError(
                    'cannot use frequency strings for `f_window_size` and `t_window_size` as the frequency '
                    f'dimension `{seq_dim}` is not datetime like. Use integers to indicate window sizes.'
                )

        self.time_freq = pd.infer_freq(ds[seq_dim].values)
        if self.time_freq not in ['D', 'H']:
            raise TypeError(
                f'The time frequency must either be daily (`D`) of hourly (`H`), is `{self.time_freq}`.'
            )

        if predict_shift < 0:
            raise ValueError(
                f'`predict_shift` cannot be negative, is `{predict_shift}`.'
            )

        if isinstance(f_window_size, str):
            f_window_size, _ = self._handle_freq(freq=f_window_size)
        if isinstance(t_window_size, str):
            _, t_window_size = self._handle_freq(freq=t_window_size)

        self.f_window_size = f_window_size
        self.t_window_size = t_window_size

        if self.f_is_qc:
            f = ds[features].isin(1)
        else:
            f = ds[features].notnull()

        if self.t_is_qc:
            t = ds[targets].isin(1)
        else:
            t = ds[targets].notnull()

        f_mask = self._get_roll_nonmissing(
            x=f,
            mode='all' if f_require_all else 'any',
            roll_dim=seq_dim,
            roll_size=f_window_size,
            min_required=f_frac
        ).compute()

        t_mask = self._get_roll_nonmissing(
            x=t,
            mode='all' if t_require_all else 'any',
            roll_dim=seq_dim,
            roll_size=t_window_size,
            min_required=t_frac
        ).shift(time=-predict_shift, fill_value=False).compute()

        mask = f_mask & t_mask
        self.dims = mask.dims

        self.f_mask = f_mask
        self.t_mask = t_mask
        self.mask = mask.compute()
        self.coords = np.argwhere(self.mask.values)
        if len(self.coords) == 0:
            raise RuntimeError(
                'the length of the coordinates is 0, no samples can be generated.'
            )
        self.perc_masked = 100 - int(self.mask.sum() / np.product(self.mask.shape) * 100)

    def _get_roll_nonmissing(
            self,
            x: xr.Dataset,
            mode: str,
            roll_dim: str,
            roll_size: int,
            min_required: Union[float, int]) -> xr.DataArray:
        """Generate a mask of missing values in a moving window.

        Parameters
        ----------
        x: xr.Dataset
            The dataset.
        mode: str
            Whether all variables must be present in the rolling window (`all`) or any (`any`).
        roll_dim : str
            The dimension to apply moving window on.
        roll_size : int
            The moving window size.
        min_required : float [0.0, 1.0] or int [0, t_window_size]
            The minimum fraction (if float) or amount (if int) of values that must be present in the window, a float
            in the range [0.0, 1,0] or an integer in the range [0, roll_size]. A value of 1.0 means that all values
            must be present. With 0.5, for example, 50 % of the values must be present in the window.

        Returns
        -------
            xr.DataArray: The mask with 1 for valid, 0 for invalid.
        """
        if mode == 'any':
            fn = lambda x: x.any('variable')
        elif mode == 'all':
            fn = lambda x: x.all('variable')
        else:
            raise ValueError(
                f'arg `mode` must be one of `all` | `any`, is `{mode}`.'
            )

        if isinstance(min_required, int):
            if 0 > min_required > roll_size:
                raise ValueError(
                    f'the argument `min_required={min_required}` must be in range [0, roll_size={roll_size}].'
                )
        else:
            if 0.0 > min_required > 1.0:
                raise ValueError(
                    'argument `min_required` must be in the range [0.0, 1.0].'
                )
            min_required = roll_size * min_required

        r = fn(x.to_array('variable')).astype(int).rolling({roll_dim: roll_size}).sum() >= min_required

        return r

    def _handle_freq(
            self,
            freq: str) -> Tuple[int, int]:
        """Get sequence lengths in days from a pandas-style frequancy tag.

        Two lengths are returned, the minimum possible days that could be covered by the frequency (e.g., 365 days
        for '1Y'/one year, or 28 days for '1M'/one month) and the maximum frequency (e.g., 366 days for '1Y'/one year,
        or 31 days for '1M'/one month)

        Parameters
        ----------
        freq : str
            A pandas-like time frequency. The units sopported are: (`Y` | `A` | `M` | `W` | `D` | `H`).

        Returns
        -------
        seq_len_min : int
            The minimum of days that could be covered by the given frequency.
        seq_len_max : int
            The maximum of days that could be covered by the given frequency.
        """

        dig_search = re.search(r'\d+', freq)

        if dig_search is None:
            raise ValueError(
                f'frequency signature `{freq}` is invalid. It must start with an integer and end with a frequnecy.')

        cut_index = dig_search.span()[1]
        num_unit = int(freq[:cut_index])
        freq_unit = freq[cut_index:]

        if num_unit < 1:
            raise ValueError(
                f'invalid frequency signature `{freq}`. Frequency must be <= 1.')

        if freq_unit == 'A':
            freq_unit = 'Y'

        if freq_unit == 'H':
            if self.time_freq == 'D':
                raise ValueError('`freq` cannot be `H` as the dataset has daily frequency.')
            seq_len_min = seq_len_max = num_unit
        elif freq_unit == 'D':
            seq_len_min = seq_len_max = num_unit
        elif freq_unit == 'W':
            seq_len_min = seq_len_max = 7 * num_unit
        elif freq_unit == 'M':
            seq_sums = self.seq_data.resample(
                {self.seq_dim: freq}).last().dt.daysinmonth.rolling({self.seq_dim: num_unit}).sum()
            seq_len_min = int(seq_sums.min().item())
            seq_len_max = int(seq_sums.max().item())
        elif freq_unit == 'Y':
            seq_sums = self.seq_data.resample(
                {self.seq_dim: '1M'}).last().dt.daysinmonth.rolling({self.seq_dim: num_unit * 12}).sum()
            seq_len_min = int(seq_sums.min().item())
            seq_len_max = int(seq_sums.max().item())
        else:
            raise ValueError(
                f'invalid frequency signature `{freq}`. Frequency unit must be one of (`Y` | `A` | `M` | `W`).')

        if (self.time_freq == 'H') and (freq_unit != 'H'):
            seq_len_min *= 24
            seq_len_max *= 24

        return seq_len_min, seq_len_max

    def _get_freq_len(
            self,
            f_freq: str,
            t_freq: str) -> Tuple[int, int]:
        """Get sequence lengths in days from a pandas-style frequancy tag for features and targets.

        Two lengths are returned, the minimum possible days that could be covered by the frequency (e.g., 355 days
        for '1Y'/one year, or 28 days for '1M'/one month) for the features and the maximum (e.g., 356 days
        for '1Y'/one year, or 31 days for '1M'/one month) for hte targets.

        Parameters
        ----------
        t_freq : str
            A pandas-like time frequency for the features, one of (`Y`=year | `A`=year | `M`=month | `W`=week).
        t_freq : str
            A pandas-like time frequency for the targets, one of (`Y`=year | `A`=year | `M`=month | `W`=week).

        Returns
        -------
        seq_len_min : int
            The minimum of days that could be covered by the given frequency.
        seq_len_max : int
            The maximum of days that could be covered by the given frequency.
        """
        f_len, _ = self._handle_freq(freq=f_freq)
        _, t_len = self._handle_freq(freq=t_freq)

        return f_len, t_len

    def __len__(self) -> int:
        """Returns the length, i.e., number of coords."""
        return len(self.coords)

    def __getitem__(self, ind: int) -> Tuple[Dict[str, Union[int, slice]], Dict[str, Union[int, slice]]]:
        """Returns query dictionaries for features and targets for the given index, see `self.ind2slice(...)`."""
        return self.ind2slice(ind)

    def ind2slice(self, ind: int) -> Tuple[Dict[str, Union[int, slice]], Dict[str, Union[int, slice]]]:
        """Returns query dictionaries for features and targets for the given index.

        The returned query can be used to subset the features and targets using `ds[features].sel(f_sel)` and
        `ds[targets].sel(t_sel)`, respectively

        Parameters
        ----------
        ind: The index to select, must be in range 0 ... len(self) - 1.

        Returns
        -------
        A tuple:
            f_sel (dict): features selection.
            t_sel (dict): targets selection.
        """
        if ind > (len(self) - 1):
            raise IndexError(
                f'index `{ind}` out of coors with length {len(self)}.'
            )

        coords = self.coords[ind]

        f_sel = {}
        t_sel = {}
        for i, d in enumerate(self.dims):
            d_coord = coords[i]
            if d != self.seq_dim:
                f_sel.update(
                    {
                        d: d_coord
                    }
                )
                t_sel.update(
                    {
                        d: d_coord
                    }
                )
            else:
                f_sel.update(
                    {
                        d: slice(
                            d_coord - self.f_window_size + 1, d_coord + 1
                        )
                    }
                )
                t_sel.update(
                    {
                        d: slice(
                            d_coord - self.t_window_size + 1 + self.predict_shift, d_coord + self.predict_shift + 1
                        )
                    }
                )

        return f_sel, t_sel

    @classmethod
    def test(cls, seed=0, timefreq='', **kwargs) -> Tuple[SeqScheme, xr.Dataset]:
        """"Generate a test instance using dummy data.

        Parameters
        ----------
        seed: int
            A random seed, default is 0.
        timefreq: str
            Frequency of the time axis, e.g., `H`, default behaviour is to not make sequential axis temporal.
        **kwargs:
            Keyword arguments are passed to SeqScheme(**kwargs).

        Returns
        -------
        A tuple:
            SeqScheme: the test sampler.
            xr.Dataset: the dummy data.
        """
        random_state = np.random.RandomState(seed=seed)

        def new_var(num_var=3, num_sites=4, num_nan=2):
            ds = xr.Dataset()
            for i in range(num_var):
                v_list = []
                for _ in range(num_sites):
                    v = np.ones(48)
                    s = random_state.choice(len(v), num_nan)
                    v[s] = np.nan
                    v_list.append(v)
                ds[f'var_{i:02d}'] = xr.DataArray(v_list, dims=['site', 'time'])
            return ds

        ds = new_var(3, num_nan=5)
        if timefreq != '':
            ds = ds.assign_coords(time=pd.date_range(start='2000-01-01', periods=48, freq=timefreq))

        test_sampler = cls(ds, ['var_00', 'var_01'], ['var_02'], **kwargs)

        ds['feature_mask'] = xr.DataArray(test_sampler.f_mask, dims=['site', 'time'])
        ds['target_mask'] = xr.DataArray(test_sampler.t_mask, dims=['site', 'time'])
        ds['mask'] = xr.DataArray(test_sampler.mask, dims=['site', 'time'])

        return test_sampler, ds

    @classmethod
    def test_plot(cls, seed=0, timefreq='', **kwargs) -> None:
        """"Generate a test plot with dummy data.

        Parameters
        ----------
        seed: int
            A random seed, default is 0.
        timefreq: str
            Frequency of the time axis, e.g., `H`, default behaviour is to not make sequential axis temporal.
        **kwargs:
            Keyword arguments are passed to SeqScheme(**kwargs).
        """
        ts, ds = cls.test(seed=seed, timefreq=timefreq, **kwargs)

        print(ts)

        features = ['var_00', 'var_01']
        targets = ['var_02']
        masks = ['feature_mask', 'target_mask', 'mask']

        _, axes = plt.subplots(2, 2, figsize=[10, 5], gridspec_kw={'hspace': 0.4, 'wspace': 0.}, sharex=True)

        for s in range(4):
            ax = axes.flat[s]

            for i in np.arange(len(ds.time)):
                ax.axvline(i, color='0.2', lw=0.2)

            for f_i, f in enumerate(features + targets + masks):
                d = ds[f].isel(site=s)
                if np.dtype(d) == bool:
                    d = d.astype(float)
                    d = d.where(d == 1, np.nan)
                (d * 0 + f_i).plot(
                    ax=ax, marker='.', ls='', label=f)

            ax.set_xticks(np.arange(len(ds.time)), minor=True)
            ax.set_yticks([])
            ax.set_ylabel('')

        axes[0, 0].legend(
            bbox_to_anchor=(0., 1.2, 2., .102), loc=3,
            ncol=6, mode="expand", borderaxespad=0.
        )

    def __repr__(self) -> str:
        args = []
        max_len = 0
        for k in get_init_arguments(type(self)):
            if (k[0] != '_') and (k != 'ds'):
                v = getattr(self, k)
                if isinstance(v, str) and k not in ['features', 'targets']:
                    v = f'`{v}`'
                a = f'{k}={v}'
                args.append(a)
                max_len = max(max_len, len(a))
        args = ",\n    ".join(args)
        s = self.__class__.__name__
        s += f'(\n    {args}\n)'
        s += f'\n{"-" * (max_len + 6)}'
        s += f'\n  ~{self.perc_masked}% are masked out'
        return s

    def __str__(self) -> str:
        return self.__repr__()
