import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead


class SingleSharedBias(nn.Module):
    """
    Single shared bias used in the Double DQN paper.
    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.
    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

    def __call__(self, x):
        return x + self.bias.expand_as(x)


class LinearQFunction(nn.Module):
    """
    Q function parametrized by soley linear layers
    """
    def __init__(self, in_features, n_actions, hidden_size=64,):
        super().__init__()
        self.q_func = nn.Sequential(
            init_chainer_default(nn.Linear(in_features, hidden_size)),
            init_chainer_default(nn.Linear(hidden_size, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    
    def forward(self, x):
        return self.q_func(x)


def compute_value_loss(
    y: torch.Tensor,
    t: torch.Tensor,
    clip_delta: bool = True,
    batch_accumulator: str = "mean",
) -> torch.Tensor:
    """
    from pfrl
    Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        return F.smooth_l1_loss(y, t, reduction=batch_accumulator)
    else:
        return F.mse_loss(y, t, reduction=batch_accumulator) / 2
