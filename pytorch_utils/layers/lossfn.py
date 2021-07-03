import torch
from torch import nn


class L2loss(nn.Module):
    """L2 norm loss between elements of input and target, ignores NaN in target.

    The loss funtion allows for having missing values in the target. The loss is first calculated per sample and then
    averaged over the batch dimension.

    Parameters
    ----------
    mode : str ("mse" | "rmse")
        Either "mse" (default) for Mean Squared Error (MSE) or "rmse"s for Root Mean Squared Error (RMSE).

    Shape
    -----
    * input: (N, *) where * means, any number of additional dimensions
    * target: (N, *), same shape as the input

    """
    def __init__(self, mode: str='mse') -> None:
        super(L2loss, self).__init__()

        if mode not in ('mse', 'rmse'):
            raise ValueError(
                f'argument `mode` must be one of ("mse" | "rmse"), is {mode}.'
            )

        self.mask: torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward call, calculate loss from input and target, must have same shape."""
        mask = target.isfinite()
        input.register_hook(lambda grad: grad.where(mask, torch.zeros(1)))
        data_dims = tuple(range(1, input.ndim))

        se = (input - target) ** 2
        se = se.where(mask, input)
        num_valid = mask.float().sum(data_dims)
        mse = torch.mean(se.sum(data_dims) / num_valid)

        if mode == 'mse':
            return mse
        else:
            return torch.sqrt(mse)



mse_loss = MSE()

input = torch.ones(3, 5, 6, requires_grad=True)
target = torch.ones(3, 5, 6, requires_grad=False) + 0.1
target[0, 0] = float('NaN')

output = mse_loss(input, target)
print('mse: ', output)
output.backward()