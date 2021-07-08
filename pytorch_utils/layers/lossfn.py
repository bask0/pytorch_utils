"""Custom loss funtions for PyTorch."""

import torch
from torch import nn


class RegressionLoss(nn.Module):
    """L1 and L2 loss functions between elements of input and target, ignores NaN in target.

    The loss funtion allows for having missing / non-finite values in the target.

    Example
    -------
    >>> mse_loss = RegressionLoss(sample_wise=True)
    >>> input = torch.ones(2, 2, requires_grad=True)
    >>> target = torch.ones(2, 2, requires_grad=False) + 1.
    >>> target[0, 0] = float('NaN')
    >>> loss = mse_loss(input, target)
    >>> loss.backward()
    >>> print('input:\n', input)
    >>> print('target:\n', target)
    >>> print('mse:\n', loss)
    >>> print('gradients:\n', input.grad)
    input:
     tensor([[1., 1.],
             [1., 1.]], requires_grad=True)
    target:
     tensor([[nan, 2.],
             [2., 2.]])
    mse:
     tensor(1., grad_fn=<MeanBackward0>)
    gradients:
     tensor([[ 0.0000, -1.0000],
             [-0.5000, -0.5000]])

    Parameters
    ----------
    mode : str ("mse" | "rmse" | "mae")
        Either "mse" (default) for Mean Squared Error (MSE), "rmse" for Root Mean Squared Error (RMSE), or "mae"
        for Mean Absolute Error (MAE).
    sample_wise : bool
        Whether to calculate sample-wise loss first and average then (`True`, default) or to calculate the loss across
        all elements. The former weights each sample equally, the latter weights each observation equally. This is
        relevant especially with many NaN in the target tensor, while there is no diffeence without NaN.

    Shape
    -----
    * input: (N, *) where * means, any number of additional dimensions
    * target: (N, *), same shape as the input

    """
    def __init__(self, mode: str = 'mse', sample_wise: bool = True) -> None:
        super(RegressionLoss, self).__init__()

        if mode not in ('mse', 'rmse', 'mae'):
            raise ValueError(
                f'argument `mode` must be one of ("mse" | "rmse" | "mae"), is {mode}.'
            )

        self.mode = mode
        self.sample_wise = sample_wise
        self.mask: torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward call, calculate loss from input and target, must have same shape."""
        mask = target.isfinite()
        input.register_hook(lambda grad: grad.where(mask, torch.zeros(1)))
        data_dims = tuple(range(1, input.ndim))

        target = target.where(mask, input)

        if self.mode == 'mae':
            err = (input - target).abs()
        else:
            err = (input - target) ** 2

        if self.sample_wise:
            num_valid = mask.float().sum(data_dims)
            merr = torch.mean(err.sum(data_dims) / num_valid)
        else:
            num_valid = mask.float().sum()
            merr = err.sum() / num_valid

        if self.mode == 'rmse':
            return torch.sqrt(merr)
        else:
            return merr
