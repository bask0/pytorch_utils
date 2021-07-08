"""Tools to deal with time-series data pipelines."""

from __future__ import annotations

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union, Tuple, Dict, Any
import inspect


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
    r"""Handles sampling from time series with missing values.

    This class handles sampling strategies for multivariate datasets (with features, `f` and targets, `t`) with
    options to account for missing data, different sampling scheemes, and prediction modes.

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

    Parameters
    ----------
    ds : xr.Dataset
        A dataset, variables from args `features` and `targets` must be present.
    features : List of str or str
        The features (herein: `f`).
    targets : List of str or str
        The targets (herein: `t`).
    f_window_size : int >= 1
        The feature window size along the `seq_dim` (e.g., `time`), i.e., how many steps in a given dimension the
        features must be present. The default (1) is a special case, where no antecedent sequence elements are
        used ('instantaneous model').
        See `Prediction scheme` for more information.
    t_window_size : int >= 1
        The target window size along the `seq_dim` (e.g., `time`), i.e., how many steps in a given dimension the
        targets must be present. The default (1) is the most common case, where 1 value is predicted.
        See `Prediction scheme` for more information.
    predict_shift : int
        An integer indicating the shift of the target compared to the features. The default, 0, means that no shift
        is done, positive values indicate that the prediction is done n steps into the future. Negative values
        indicate that past values are predicted.
        See `Prediction scheme` for more information.
    f_allow_miss : bool
        Whether missing features are allowed within the moving window. If `True`, features are not checked.
        Default is `False`.
    t_allow_miss : bool
        Whether missing targets are allowed within the moving window. If `True`, targets are not checked.
        Default is `False`.
    f_require_all : bool
        If `True` (default), the features are masked if ANY feature is missing and else if ALL features are missing.
        In other words, if you want *at least one* feature to be present, set `False`, if you want *all* features to
        be present, set `True`.
    t_allow_miss : bool
        If `True` (default), the targets are masked if ANY targets is missing and else if ALL features are missing.
        In other words, if you want *at least* one target to be present, set `False`, if you want *all* targets to
        be present, set `True`.
    seq_dim : str
        The sequence dimension, default is `time`. Must be present in `ds`.
    """
    def __init__(
            self,
            ds: xr.Dataset,
            features: Union[List[str], str],
            targets: Union[List[str], str],
            f_window_size: int = 1,
            t_window_size: int = 1,
            predict_shift: int = 0,
            f_allow_miss: bool = False,
            t_allow_miss: bool = False,
            f_require_all: bool = True,
            t_require_all: bool = True,
            seq_dim: str = 'time') -> None:

        self.features = [features] if isinstance(features, str) else features
        self.targets = [targets] if isinstance(targets, str) else targets

        if seq_dim not in ds.dims:
            raise ValueError(
                f'argument seq_dim=`{seq_dim}` is not a valid dimension of `ds`. Use one of {list(ds.dims)}.'
            )

        seq_len = len(ds[seq_dim])
        for k, v in [['f_window_size', f_window_size], ['t_window_size', t_window_size]]:
            if v < 1:
                raise ValueError(
                    f'argument `{k}` cannot be < 1, is {v}.'
                )
            if v > seq_len:
                raise ValueError(
                    f'argument `{k}` cannot be > seq_len={seq_len}, is {v}.'
                )

        self.f_window_size = f_window_size
        self.t_window_size = t_window_size
        self.predict_shift = predict_shift
        self.f_allow_miss = f_allow_miss
        self.t_allow_miss = t_allow_miss
        self.f_require_all = f_require_all
        self.t_require_all = t_require_all
        self.seq_dim = seq_dim

        window_valid = xr.Dataset(
            {
                'features':
                    self._get_roll_nonmissing(
                        x=ds[features],
                        mode='all' if f_require_all else 'any',
                        roll_dim=seq_dim,
                        roll_size=f_window_size
                    ),
                'targets':
                    self._get_roll_nonmissing(
                        x=ds[targets],
                        mode='all' if t_require_all else 'any',
                        roll_dim=seq_dim,
                        roll_size=t_window_size
                    ).shift(time=-predict_shift, fill_value=False)
            }
        )

        self.dims = window_valid.features.dims

        mask: xr.DataArray
        if f_allow_miss:
            if t_allow_miss:
                mask = xr.ones_like(window_valid.features)
            else:
                mask = window_valid.targets
        else:
            if t_allow_miss:
                mask = window_valid.features
            else:
                mask = window_valid.features & window_valid.targets

        self.f_mask = window_valid.features
        self.t_mask = window_valid.targets
        self.mask = mask
        self.coords = np.argwhere(self.mask.values)
        if len(self.coords) == 0:
            raise RuntimeError(
                'the length of the coordinates is 0, no samples can be generated.'
            )

    def _get_roll_nonmissing(self, x: xr.Dataset, mode: str, roll_dim: str, roll_size: int) -> xr.DataArray:
        """Generate a mask f missing values in a moving window.

        Parameters
        ----------
        x: xr.Dataset
            The dataset.
        mode: str
            Whether all variables must be present or any.
        roll_dim : str)
            The dimension to apply moving window on.
        roll_size : int
            The moving window size.

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
        return fn(x.to_array('variable').notnull()).astype(int).rolling({roll_dim: roll_size}).sum() == roll_size

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
    def test(cls, seed=0, **kwargs) -> Tuple[SeqScheme, xr.Dataset]:
        """"Generate a test instance using dummy data.

        Parameters
        ----------
        seed: int
            A random seed, default is 0.
        **kwargs:
            Keyword arguments are passed to SeqScheme(**kwargs).

        Returns
        -------
        A tuple:
            SeqScheme: the test sampler.
            xr.Dataset: the dummy data.
        """
        random_state = np.random.RandomState(seed=seed)

        def new_var(num_var=3, num_sites=4, num_nan=4):
            ds = xr.Dataset()
            for i in range(num_var):
                v_list = []
                for _ in range(num_sites):
                    v = np.arange(20.)
                    s = random_state.choice(len(v), num_nan)
                    v[s] = np.nan
                    v_list.append(v)
                ds[f'var_{i:02d}'] = xr.DataArray(v_list, dims=['site', 'time'])
            return ds

        ds = new_var(3, num_nan=5)
        test_sampler = cls(ds, ['var_00', 'var_01'], ['var_02'], **kwargs)

        ds['feature_mask'] = xr.DataArray(test_sampler.f_mask, dims=['site', 'time'])
        ds['target_mask'] = xr.DataArray(test_sampler.t_mask, dims=['site', 'time'])
        ds['mask'] = xr.DataArray(test_sampler.mask, dims=['site', 'time'])

        return test_sampler, ds

    @classmethod
    def test_plot(cls, seed=0, **kwargs) -> None:
        """"Generate a test plot with dummy data.

        Parameters
        ----------
        seed: int
            A random seed, default is 0.
        **kwargs:
            Keyword arguments are passed to SeqScheme(**kwargs).
        """
        ts, ds = cls.test(seed=seed, **kwargs)

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
        perc_masked = 100 - int(self.mask.sum() / np.product(self.mask.shape) * 100)
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
        s += f'\n  ~{perc_masked}% are masked out'
        return s

    def __str__(self) -> str:
        return self.__repr__()
