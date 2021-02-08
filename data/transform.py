import numpy as np
import torch

from typing import List, Dict, Iterable, Tuple, Union, Any, Optional


class Normalize(object):
    def __init__(self, dtype: type = np.float32) -> None:
        """Data normalization functionality for torch.Tensor and np.ndarrays.

        Usage:
            1. Create a new instance `Noramlize()`.
            2. Register variables. This means, a variable name and data is passed and the
               mean and standard deviation is recorded.
            3. Pass variable (single or multiple ones) to (un-)normalize them using the
               stats that were regitered before.

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

            Unnormalize data:

            >>> # A standard normal distributed torch.Tensor is unnormalized.
            >>> n.unnormalize('var_b', torch.randn(100)).std()
            10.12

            Normalize dict:

            >>> n.normalize_dict({'var_a': torch.arange(2), 'var_b': np.random.normal(loc=2, scale=10., size=2)})
            {'var_a': tensor([-1.5667, -1.2185]),
             'var_b': array([-1.35922953, -0.81025503])}

            Normalize dict and stack (note that we cannot mix np.ndarrays and torch.Tensors here):

            >>> n.normalize_dict({'var_a': np.arange(2), 'var_b': np.random.normal(loc=2, scale=10., size=2)}, return_stack=True)
            array([[-1.56669891,  0.0801657 ],
                   [-1.2185436 , -0.56352366]])

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

    def unnormalize(
            self,
            key: str,
            x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Un-normalize `x`, stats for `key` must have been registered previously.

        Un-normlization: x * std + mean

        Args:
            key (str):
                The name of the variable to un-normlize.
            x (np.ndarray or torch.Tensor):
                The data to un-normalize.

        Returns:
            np.ndarray or torch.Tensor: un-normalized data, same type as input.
        """
        return self._transform(key, x, invert=True)

    def normalize_dict(
            self,
            d: Dict[str, Union[np.ndarray, torch.Tensor]],
            variables: Optional[List[str]] = None,
            return_stack: bool = False) -> Union[Dict[str, Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor]:
        """Normalize data in `d`, stats for keys must have been registered previously.

        Normlization: (x - mean) / std

        Args:
            d (dict):
                The name of the variable to normlize.
            variables (List[str]):
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

    def unnormalize_dict(
            self,
            d: Dict[str, Union[np.ndarray, torch.Tensor]],
            variables: Optional[List[str]] = None,
            return_stack: bool = False) -> Union[Dict[str, Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor]:
        """Un-normalize data in `d`, stats for keys must have been registered previously.

        Un-normlization: x * std + mean

        Args:
            d (dict):
                The name of the variable to un-normlize.
            variables (List[str]):
                Optional subset of variables to return. All variables must be present in `stats`.
            return_stack (bool):
                Whether to return a stack of all values in `d`. If `False`, a dict with the un-normalized
                data is returned. If `True`, the values are stacked along the last dimension. The values
                can be troch.Tensors or np.ndarrays. Defaults to `False`.

        Returns:
            dict, np.ndarray, torch.Tensor: un-normalized data, same type as input. If `return_stack`
                is `True`, a np.ndarray or torch.Tensor is returned.
        """
        self._assert_dtype('d', d, dict)
        self._assert_dtype('return_stack', return_stack, bool)

        if variables is not None:
            self._assert_dtype('variables', variables, list)
            d = {v: d[v] for v in variables}

        d = self._transform_dict(d, invert=True)
        if return_stack:
            return self._stack_dict(d)
        else:
            return d

    def register(self, key: str, x: Union[np.ndarray, torch.Tensor]) -> None:
        """Register data stats (mean and standard deviation).

        Args:
            key (str):
                The name of the variable.
            x (np.ndarray or torch.Tensor):
                The data to calculate mean and standard deviation from.

        """
        self._assert_dtype('key', key, str)
        self._assert_dtype('x', x, (np.ndarray, torch.Tensor))

        mean, std = self._get_mean_and_std(x)
        self._stats.update({key: {'mean': mean, 'std': std}})

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

    def _transform(
            self, key: str,
            x: Union[np.ndarray, torch.Tensor],
            invert: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Transform data, either normalize or unnormalize (if `invert`)"""
        if key not in self._stats:
            raise KeyError(f'no stats found for key `{key}`.')
        self._assert_dtype('invert', invert, bool)

        stats = self._stats[key]
        m = stats['mean']
        s = stats['std']

        if invert:
            return x * s + m
        else:
            return (x - m) / s

    def _transform_dict(
            self,
            d: Dict[str, Union[np.ndarray, torch.Tensor]],
            invert: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Transform data, either normalize or unnormalize (if `invert`)"""
        self._assert_dtype('d', d, dict)

        r = {}
        for key, val in d.items():
            r.update({key: self._transform(key, val, invert=invert)})

        return r

    def _stack_dict(
            self,
            d: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Union[np.ndarray, torch.Tensor]:
        """Stack values in a dict along last dimension."""
        if self._contains_torch(d):
            return torch.stack(list(d.values()), dim=-1)
        else:
            return np.stack(list(d.values()), axis=-1)

    def _get_mean_and_std(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Calculate mean and standard deviation for np.ndarray or torch.Tensor."""
        return np.nanmean(x).astype(self.dtype), np.nanstd(x).astype(self.dtype)

    def _assert_dtype(self, key: str, val: Any, dtype: Union[type, Tuple[type]]) -> None:
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
            raise TypeError(f'`{key}` must be of type `{dtype_as_str}` but is `{type(val)}`.')

    def _assert_iterable(self, key, val):
        """Check if val is an iterable (excluding str).

        Args:
            key (str):
                The name of the variable to check.
            val (Any):
                The value to check.

        Raises:
            TypeError if not iterable.
        """
        if not hasattr(keys, '__iter__') or isinstance(val, str):
            raise TypeError(f'`{key}` must be an iterable but is `{type(val)}`.')

    def _contains_torch(self, d: Dict[str, Union[torch.Tensor, np.ndarray, np.floating]]) -> bool:
        """Checks if a dict contains a torch.Tensor or a np.ndarray.

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
        elif isinstance(first_item, (np.ndarray, np.floating)):
            return False
        else:
            raise ValueError(
                'dict contains values that are neither of type torch.Tensor nor '
                f'np.ndarray, but type `{first_item.dtype}`.'
            )

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

    @property
    def stats(self):
        """A dict containing means and standard deviations."""
        return self._stats

    def _set_stats(self, d: Dict[str, Dict[str, float]]) -> None:
        """Assign stats dict. Internal use only, do not use.

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
        self._stats = d

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
