"""Data normalization for numpy, xarray, and Pytorch tensors.

Author: Basil Kraft
"""

import numpy as np
import xarray as xr
import torch

import pickle

from typing import List, Dict, Iterable, Tuple, Union, Any, Optional


class Normalize(object):
    def __init__(
            self,
            dtype: type = np.float32) -> None:
        """Data normalization functionality for torch.Tensor, xarray.Datasets, and np.ndarrays.

        Usage:

            1. Create a new instance `Noramlize()`.
            2. Register variables. This means, a variable name and data is passed and the
               mean and standard deviation is recorded.
            3. a) Pass variable (single or multiple ones) to (de-)normalize them using the
               stats that were regitered before.
               b) You can also create torch.nn.Modules using `get_normalization_layer`

        Example:

            Register variables:

            >>> n = Normalize()
            >>> n.register('var_a', torch.arange(10))
            >>> n.register('var_b', np.random.normal(loc=2, scale=10., size=10))
            >>> print(n)
            Normalize(dtype=float32)
            ----------------------------------------
             * var_a: 49.500 (28.866 std)
             * var_b: -0.048 (9.808 std)

            Register variables (pass dict):

            >>> n = Normalize()
            >>> n.register_dict({'var_a': torch.arange(10), 'var_b': np.random.normal(loc=2, scale=10., size=10)})
            >>> print(n)
            Normalize(dtype=float32)
            ----------------------------------------
             * var_a: 4.500 (2.872 std)
             * var_b: -1.733 (12.913 std)

            Normalize data:

            >>> n.normalize('var_a', np.arange(10)).std()
            1.000000003595229

            Denormalize data:

            >>> # A standard normal distributed torch.Tensor is denormalized.
            >>> n.denormalize('var_b', torch.randn(100)).std()
            10.12

            Normalize dict:

            >>> n.normalize_dict({'var_a': torch.arange(2), 'var_b': np.random.normal(loc=2, scale=10., size=2)})
            {'var_a': tensor([-1.5667, -1.2185]),
             'var_b': array([-1.35922953, -0.81025503])}

            Normalize dict and stack (note that we cannot mix np.ndarrays and torch.Tensors here):

            >>> n.normalize_dict(
            ...     {'var_a': np.arange(2),
            ...      'var_b': np.random.normal(loc=2, scale=10., size=2)}, return_stack=True)
            array([[-1.56669891,  0.0801657 ],
                   [-1.2185436 , -0.56352366]])

            Register stats manually:

            n.register_manually('my_var', mean=1.0, std=2.5)

        Args:
            dtype (dtype):
                The data type of the transformed data. Deault is np.float32.

        """
        self._stats = {}
        self.dtype = dtype

    def normalize(
            self,
            key: str,
            x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize `x`, stats for `key` must have been registered previously.

        Normlization: (x - mean) / std

        Args:
            key (str):
                The name of the variable to normlize.
            x (np.ndarray or torch.Tensor):
                The data to normalize.
        Returns:
            Union[np.ndarray, torch.Tensor]: normalized data, same type as input.
        """
        return self._transform(key, x, invert=False)

    def denormalize(
            self,
            key: str,
            x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Denormalize `x`, stats for `key` must have been registered previously.

        Denormlization: x * std + mean

        Args:
            key (str):
                The name of the variable to denormlize.
            x (np.ndarray or torch.Tensor):
                The data to denormalize.

        Returns:
            np.ndarray or torch.Tensor: denormalized data, same type as input.
        """
        return self._transform(key, x, invert=True)

    def normalize_dict(
            self,
            d: Dict[str, Union[np.ndarray, torch.Tensor, float]],
            variables: Optional[List[str]] = None,
            return_stack: bool = False) -> Union[Dict[str, Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor]:
        """Normalize data in `d`, stats for keys must have been registered previously.

        Normlization: (x - mean) / std

        Args:
            d (dict):
                The name of the variable to normlize.
            variables (Optional[List[str]]):
                Optional subset of variables to return. All variables must be present in `stats`.
            return_stack (bool):
                Whether to return a stack of all values in `d`. If `False`, a dict with the normalized
                data is returned. If `True`, the values are stacked along the last dimension. The values
                can be troch.Tensors or np.ndarrays. Defaults to `False`.

        Returns:
            dict, np.ndarray, torch.Tensor: normalized data, same type as input. If `return_stack`
                is `True`, a np.ndarray or torch.Tensor is returned.
        """
        self._assert_dtype('d', d, dict)
        self._assert_dtype('return_stack', return_stack, bool)

        if variables is not None:
            self._assert_dtype('variables', variables, list)
            d = {v: d[v] for v in variables}

        d = self._transform_dict(d, invert=False)

        if return_stack:
            return self._stack_dict(d)
        else:
            return d

    def denormalize_dict(
            self,
            d: Dict[str, Union[np.ndarray, torch.Tensor]],
            variables: Optional[List[str]] = None,
            return_stack: bool = False) -> Union[Dict[str, Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor]:
        """Denormalize data in `d`, stats for keys must have been registered previously.

        Denormlization: x * std + mean

        Args:
            d (dict):
                The name of the variable to denormlize.
            variables (Optional[List[str]]):
                Optional subset of variables to return. All variables must be present in `stats`.
            return_stack (bool):
                Whether to return a stack of all values in `d`. If `False`, a dict with the denormalized
                data is returned. If `True`, the values are stacked along the last dimension. The values
                can be troch.Tensors or np.ndarrays. Defaults to `False`.

        Returns:
            dict, np.ndarray, torch.Tensor: denormalized data, same type as input. If `return_stack`
                is `True`, a np.ndarray or torch.Tensor is returned.
        """
        self._assert_dtype('d', d, dict)
        self._assert_dtype('return_stack', return_stack, bool)

        if variables is not None:
            self._assert_dtype('variables', variables, list)
            d = {v: d[v] for v in variables}

        d = self._transform_xr(d, invert=True)
        if return_stack:
            return self._stack_dict(d)
        else:
            return d

    def normalize_xr(
            self,
            ds: xr.Dataset,
            variables: Optional[List[str]] = None,
            return_stack: bool = False,
            keep_all_variables: bool = False) -> Union[xr.Dataset, Dict[str, Union[xr.Dataset, np.ndarray]]]:
        """Normalize an xr.Dataset, stats for keys must have been registered previously.

        Normlization: (x - mean) / std

        Args:
            ds (xr.Dataset):
                The name of the variable to normlize.
            variables (Optional[List[str]]):
                Optional subset of variables to return. All variables must be present in `stats`.
            return_stack (bool):
                Whether to return a np.ndarray stack of all variables. If `False`, an xr.Dataset with the normalized
                data is returned. If `True`, the values are stacked along the last dimension. Defaults to `False`.
            keep_all_variables (bool):
                Whether to keep all variables in `ds`, even if not in `variables`. Default is `False`. Note that if
                `keep_all_variables=True`, `return_stack=True` may cause errors as variables may be kept in `ds` that
                have different dimensionality.

        Returns:
            xr.Dataset, np.ndarray: normalized xr.Dataset. If `return_stack` is `True`, a np.ndarray is returned.
        """
        self._assert_dtype('ds', ds, xr.Dataset)
        self._assert_dtype('return_stack', return_stack, bool)

        if variables is not None:
            if isinstance(variables, str):
                variables = [variables]
            self._assert_dtype('variables', variables, list)

            if keep_all_variables:
                keep_vars = list(set(ds.data_vars) - set(variables))
                keep_ds = ds[keep_vars]

            ds = ds[variables]

        ds = self._transform_xr(ds, invert=False)

        if keep_all_variables:
            ds = xr.merge((ds, keep_ds))

        if return_stack:
            return ds.to_array().transpose(..., 'variable').values
        else:
            return ds

    def denormalize_xr(
            self,
            ds: xr.Dataset,
            variables: Optional[List[str]] = None,
            return_stack: bool = False,
            keep_all_variables: bool = False) -> Union[xr.Dataset, Dict[str, Union[xr.Dataset, np.ndarray]]]:
        """Denormalize an xr.Dataset, stats for keys must have been registered previously.

        Denormlization: (x - mean) / std
        Args:
            ds (xr.Dataset):
                The name of the variable to normlize.
            variables (Optional[List[str]]):
                Optional subset of variables to return. All variables must be present in `stats`.
            return_stack (bool):
                Whether to return a np.ndarray stack of all variables. If `False`, an xr.Dataset with the normalized
                data is returned. If `True`, the values are stacked along the last dimension. Defaults to `False`.
            keep_all_variables (bool):
                Whether to keep all variables in `ds`, even if not in `variables`. Default is `False`. Note that if
                `keep_all_variables=True`, `return_stack=True` may cause errors as variables may be kept in `ds` that
                have different dimensionality.

        Returns:
            xr.Dataset, np.ndarray: normalized xr.Dataset. If `return_stack` is `True`, a np.ndarray is returned.
        """
        self._assert_dtype('ds', ds, xr.Dataset)
        self._assert_dtype('return_stack', return_stack, bool)

        if variables is not None:
            if isinstance(variables, str):
                variables = [variables]
            self._assert_dtype('variables', variables, list)

            if keep_all_variables:
                keep_vars = list(set(ds.data_vars) - set(variables))
                keep_ds = ds[keep_vars]

            ds = ds[variables]

        ds = self._transform_xr(ds, invert=True)

        if keep_all_variables:
            ds = xr.merge((ds, keep_ds))

        if return_stack:
            return ds.to_array().transpose(..., 'variable').values
        else:
            return ds

    def register(self, key: str, x: Union[np.ndarray, torch.Tensor]) -> None:
        """Register data stats (mean and standard deviation).

        Args:
            key (str):
                The name of the variable.
            x (np.ndarray or torch.Tensor):
                The data to calculate mean and standard deviation from.

        """
        self._assert_dtype('key', key, str)
        self._assert_dtype('x', x, (np.ndarray, torch.Tensor, xr.DataArray))

        mean, std = self._get_mean_and_std(x)
        self._update_stats({key: {'mean': mean, 'std': std}})

    def register_manually(self, key: str, mean: Any, std: Any) -> None:
        """Register data stats (mean and standard deviation) manually.

        Args:
            key (str):
                The name of the variable.
            mean (numeric):
                The mean. Must be castable to `self.dtype`.
            std (numeric):
                The standard deviation. Must be castable to `self.dtype`.

        """
        self._assert_dtype('key', key, str)
        mean = self._cast_to_dtype('mean', mean)
        std = self._cast_to_dtype('std', std)

        self._stats.update({key: {'mean': mean, 'std': std}})

    def register_xr(self, ds: xr.Dataset, variables: Optional[Union[str, List[str]]] = None) -> None:
        """Register xarray data stats (mean and standard deviation per variable).

        Args:
            ds (xr.Dataset):
                The dataset to record data stats for.
            variables (Optional[str]):
                Variable names to register stats for. Defaults to `None` fir all varaibles in the dataset.

        """
        self._assert_dtype('ds', ds, xr.Dataset)
        if isinstance(variables, str):
            variables = [variables]
        self._assert_dtype('variables', variables, (list, type(None)))

        if variables is None:
            variables = list(ds.data_vars)

        for variable in variables:
            self.register(variable, ds[variable])

    def register_dict(self, d: Dict[str, Union[np.ndarray, torch.Tensor]]) -> None:
        """Register data stats (mean and standard deviation) for dict elements.

        Args:
            d (dict):
                Stats are registered for every key, value pair.

        """
        self._assert_dtype('d', d, dict)

        for key, val in d.items():
            self.register(key, val)

    def deleteitem(self, key: str) -> None:
        """Delet an item from stats.

        key (str):
            The key to delete from the stats.
        """
        self._assert_dtype('key', key, str)

        del self._stats[key]

    def deletelist(self, keys: Iterable[str]) -> None:
        """Delet several items from stats.

        keys (list):
            The keys to delete from the stats.
        """
        if not hasattr(keys, '__iter__'):
            raise TypeError(f'`keys` must be an iterable but is {type(keys)}.')

        for key in keys:
            self.deleteitem(key)

    def save(self, path: str) -> None:
        """Save object to file. Can be restored later using `.load(...)`.

        Args:
            pash (str):
                File path to save object to.
        """
        d_save = self.stats
        d_save.update({'dtype': self.dtype})
        with open(path, 'wb') as f:
            pickle.dump(d_save, f)

    @classmethod
    def load(cls, path: str) -> 'Normalize':
        """Load from file.

        Example:
            >>> n = Normalize()
            >>> n.register_dict({'var_a': torch.arange(10)})
            >>> n.save('test.pkl')
            >>> n_restored = Normalize.load('test.pkl')

        Args:
            path (str):
                File path to save object to.

        Returns:
            Normalize: restored object.
        """
        with open(path, 'rb') as f:
            d = pickle.load(f)
        dtype = d.pop('dtype')
        n = cls(dtype)
        n._set_stats(d)
        return n

    def get_normalization_layer(
            self_,
            variables: Union[List[str], str],
            invert: bool,
            stack: bool = False,
            stack_along_new_dim: bool = True,
            stack_dim: int = -1) -> torch.nn.Module:
        """Returns a torch data (de-)normalization layer.

        This is useful to make the code independent from this Normalization
        module, e.g., saving / loading can be done with PyTorch.

        Args:
            variabes (List[str] or str):
                Variables to transform, must have been registered previously.
            invert (bool):
                Whether to normalize (`True`) or to denormalize (`False`).
            stack (bool):
                Whether to stack dict to tensor along last dimension (`True`) or to return
                a dic (`False`). Default is `False`.
            stack_along_new_dim (bool):
                If `True`, the dict elements are stacked along a new dimension. Example: we have two dict
                elements, it is len(variables)=2, each with shape (10, 1). With `stack_along_new_dim=False`,
                the resulting tensor has shape (10, 1, 1), (10, 2) else.
                Only applies if `stack=True`, default is `True`.
            stack_dim (int):
                The dimension along which the resulting tensor is stacked. Depending on `stack_along_new_dim`,
                either a new dimension is added (`stack_along_new_dim=True`), or else the existing dimensions are
                stacked. Only applies if `stack=True`, default is -1 (the last one).

        Return:
            torch.nn.Module: a normalization layer.

        """
        # Check types to avoid runtime errors.
        self_._assert_dtype('variables', variables, (str, list))
        self_._assert_dtype('invert', invert, bool)
        self_._assert_dtype('stack', stack, bool)
        self_._assert_dtype('stack_along_new_dim', stack_along_new_dim, bool)
        self_._assert_dtype('stack_dim', stack_dim, int)

        class DataNorm(torch.nn.Module):
            """Normalization layer for (optionally inverse) standard normal distribution tranfrormation.

            Attributes:
                stats (Dict[str, Dict[str, float]]):
                    Dictionary contining variables (keys), each with `mean`: (float)
                    and `std` (float) stats. Signature: dict(var_a=dict(mean=0.1, std=0.8)).
                variables (List[str]):
                    List of variables to transform, must be present in `stats`.
                invert (bool):
                    If `True`, the transformation is inverted, e.g., denormalization is done.
                stack (bool):
                    If `True`, the transformed dict is stacked along last dimension. Else, a dict
                    containing transformed data is returned.
                stack_along_new_dim (bool):
                    If `True`, the dict elements are stacked along a new dimension. Example: we have two dict
                    elements, it is len(variables)=2, each with shape (10, 1). With `stack_along_new_dim=False`,
                    the resulting tensor has shape (10, 1, 1), (10, 2) else.
                    Only applies if `stack=True`, default is `True`.
                stack_dim (int):
                    The dimension along which the resulting tensor is stacked. Depending on `stack_along_new_dim`,
                    either a new dimension is added (`stack_along_new_dim=True`), or else the existing dimensions are
                    stacked. Only applies if `stack=True`, default is -1 (the last one).

            Args (__call__):
                x: dict[str, Tensor] or Tensor

            Usage:
                norm_layer = NormLayer('varname', invert=False)
                denorm_layer = NormLayer('varname', invert=True)
                x = dict(varname=torch.randn(10))
                assert torch.isclose(
                        x['varname'],
                        denorm_layer(norm_layer(x))['varname'],
                    )
            """
            def __init__(self) -> None:
                super().__init__()

                self.variables = [variables] if isinstance(variables, str) else variables
                self.invert = invert
                self.stack = stack
                self.stack_along_new_dim = stack_along_new_dim
                self.stack_dim = stack_dim

                missing = []
                for var in self.variables:
                    if var not in self_.stats:
                        missing.append(var)

                if len(missing) > 0:
                    raise KeyError(f'no stats found for key(s): `{*missing, }`.')

                self._stats = {var: self_.stats[var] for var in self.variables}

            def forward(
                    self,
                    x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
                """Transform input.

                Args:
                    x (Dict[str: torch.Tensor] or torch.Tensor):
                        - a dict of variable, value pairs. Keys must be present in `stats`, or
                        - a tensor with last dimension matching the number of `variables`. The order
                          is assumed to correspond to `variables`.

                Returns:
                    Either a dict of key, value (torch.Tensor) pairs (if `stack=True`), or a torch.Tensor
                    with variable stacked in last dimension.
                """

                if not isinstance(x, (dict, torch.Tensor)):
                    raise TypeError(
                        f'`x` must be of type `dict` or `torch.Tensor` but is `{type(x).__name__}`.'
                    )

                if isinstance(x, torch.Tensor):
                    x = self._tensor_to_dict(x)

                out = {k: self._normalize(k, x[k]) for k, v in self.stats.items()}

                if self.stack:
                    if self.stack_along_new_dim:
                        return torch.stack(list(out.values()), dim=self.stack_dim)
                    else:
                        return torch.cat(list(out.values()), dim=self.stack_dim)
                else:
                    return out

            def _normalize(self, key, val) -> torch.Tensor:
                """Normalize single variable."""

                if key not in self.stats:
                    raise KeyError(f'no stats found for key `{key}`.')

                stats = self.stats[key]
                mn = stats['mean']
                st = stats['std']

                if self.invert:
                    return val * st + mn
                else:
                    return (val - mn) / st

            def _tensor_to_dict(self, x):
                if x.shape[-1] != len(self.variables):
                    raise ValueError(
                        f'`x` last dimension ({x.shape[-1]}) must match number of '
                        f'target variables ({len(self.variables)}).'
                    )

                return {t: x[..., i] for i, t in enumerate(self.variables)}

            @property
            def stats(self) -> Dict:
                """A dict containing means and standard deviations."""
                return self._stats

            def __str__(self) -> str:
                layer_name = 'DataDenorm' if self.invert else 'DataNorm'
                stack = f'stack={self.stack}'
                s = f'{layer_name}(variables=[{", ".join(self.variables)}], {stack})'
                return s

            def __repr__(self) -> str:
                return self.__str__()

        return DataNorm()

    @property
    def stats(self) -> Dict:
        """A dict containing means and standard deviations."""
        return self._stats

    def _transform(
            self, key: str,
            x: Union[np.ndarray, torch.Tensor, float],
            invert: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Transform data, either normalize or denormalize (if `invert`)"""
        if key not in self.stats:
            raise KeyError(f'no stats found for key `{key}`.')
        self._assert_dtype('invert', invert, bool)

        stats = self.stats[key]
        m = stats['mean']
        s = stats['std']

        if invert:
            return x * s + m
        else:
            return (x - m) / s

    def _transform_dict(
            self,
            d: Dict[str, Union[np.ndarray, torch.Tensor, float]],
            invert: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor, float]]:
        """Transform data, either normalize or denormalize (if `invert`)"""
        self._assert_dtype('d', d, dict)

        r = {}
        for key, val in d.items():
            r.update({key: self._transform(key, val, invert=invert)})

        return r

    def _transform_xr(
            self,
            ds: xr.Dataset,
            invert: bool = False) -> xr.Dataset:
        """Transform xr.Dataset, either normalize or denormalize (if `invert`)"""
        self._assert_dtype('ds', ds, xr.Dataset)

        ds_norm = xr.Dataset()

        for variable in ds.data_vars:
            ds_norm[variable] = self._transform(variable, ds[variable], invert=invert)

        return ds_norm

    def _stack_dict(
            self,
            d: Dict[str, Union[np.ndarray, torch.Tensor, float]]) -> Union[np.ndarray, torch.Tensor]:
        """Stack values in a dict along last dimension."""
        if self._contains_torch(d):
            return torch.stack(list(d.values()), dim=-1)
        else:
            return np.stack(list(d.values()), axis=-1)

    def _get_mean_and_std(
        self,
        x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[float, float]:
        """Calculate mean and standard deviation for np.ndarray or torch.Tensor."""
        data_mean = np.nanmean(x)
        data_std = np.nanstd(x)

        # Handle std=0, which would lead to division by zero error.
        if data_std == 0.0:
            data_std = data_mean * 0 + 1  # Make sure that data_std is same type as data_mean, but 1.0.

        return data_mean.astype(self.dtype), data_std.astype(self.dtype)

    def _assert_dtype(self, key: str, val: Any, dtype: Union[type, Tuple[type, ...]]) -> None:
        """Check the type and raise TypeError if wrong.

        Args:
            key (str):
                The name of the variable to check.
            val (Any):
                The value to check.
            dtype (type or tuple of types):
                The required type.

        Raises:
            TypeError: if wrong type.
        """

        dtype_as_str = dtype.__name__ if isinstance(dtype, type) else ' or '.join([t.__name__ for t in dtype])
        if not isinstance(val, dtype):
            raise TypeError(f'`{key}` must be of type `{dtype_as_str}` but is `{type(val).__name__}`.')

    def _assert_iterable(self, key: str, val: Any) -> None:
        """Check if val is an iterable (excluding str).

        Args:
            key (str):
                The name of the variable to check.
            val (Any):
                The value to check.

        Raises:
            TypeError if not iterable.
        """
        if not hasattr(key, '__iter__') or isinstance(val, str):
            raise TypeError(f'`{key}` must be an iterable but is `{type(val)}`.')

    def _contains_torch(self, d: Dict[str, Union[torch.Tensor, np.ndarray, float]]) -> bool:
        """Checks if a dict contains torch.Tensor or np.ndarray.

        Args:
        d (dict):
            A dict containing either torch.Tensor, np.ndarray, or float.

        Returns:
            bool: `True` if dict contains torch.Tensor, `False` if np.ndarray.

        Raises:
            ValueError: if not torch.Tensor or np.ndarray.

        """
        self._assert_dtype('d', d, dict)
        first_key = list(d.keys())[0]
        first_item = d[first_key]
        if isinstance(first_item, torch.Tensor):
            return True
        elif isinstance(first_item, np.ndarray) or np.issubdtype(first_item, np.floating):
            return False
        else:
            raise ValueError(
                'dict contains values that are neither of type torch.Tensor nor '
                f'np.ndarray, but type `{type(first_item).__name__}`.'
            )

    def _cast_to_dtype(self, key: str, x: Any) -> Any:
        """Cast a number to the `self.dtype`.

        key (str):
            The value name.
        x (Any):
            A number that can be cast to `self.dtype`.

        Returns:
            self.dtype: A number of type `self.dtype`.
        """

        try:
            t = self.dtype(x)
        except Exception:
            raise ValueError(
                f'failed to cast {key}=`{x}` to type {self.dtype.__name__}.'
            )

        if t.ndim != 0:
            raise ValueError(
                f'casted `{key}={x}` to type {self.dtype.__name__}. Result ({t}) must be a '
                f'0-D array, but is {t.ndim}-D.'
            )

        return t

    def _update_stats(self, d: Dict[str, Dict[str, float]]):
        """Update stats dict. Internal use only, do not use.

        Args:
            d (dict):
                A dict of variable means and standard deviations: {'var_a': {'mean': __, 'std': __}}.
        """
        self._assert_dtype('d', d, dict)
        for key, val in d.items():
            self._assert_dtype(f'd[`{key}`]', val, dict)
            if not ('mean' in val and 'std' in val):
                raise ValueError(
                    f'tried to assign stats dict, but d[`{key}`] does not have keys `mean` and `std`.'
                )
        self._stats.update(d)

    def _set_stats(self, d: Dict[str, Dict[str, float]]) -> None:
        """Assign stats dict. Internal use only, do not use.

        Args:
            d (dict):
                A dict of variable means and standard deviations: {'var_a': {'mean': __, 'std': __}}.
        """
        self._stats = {}
        self._update_stats(d)

    def __str__(self) -> str:
        s = f'Normalize(dtype={self.dtype.__name__})\n{"-"* 40}\n'
        if len(self.stats) == 0:
            s += '  no stats registered.'
        else:
            for key, stats in self.stats.items():
                s += ' * '
                s += f'{key}: {stats["mean"]:0.3f} ({stats["std"]:0.3f} std)'
                s += '\n'

        return s

    def __repr__(self) -> str:
        return self.__str__()
