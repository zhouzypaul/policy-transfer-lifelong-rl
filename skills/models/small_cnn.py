import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl.initializers import init_chainer_default


def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)
    return init_bias


class SmallCNN(nn.Module):
    """
    the standard nature DQN is too big to process the agent space in monte
    (because of kernel size and strides, the small 24 * 24 agent space image
    just disappears)
    so we build a custom one
    """
    def __init__(
        self, n_input_channels=4, n_output_channels=64, activation=F.relu, bias=0.1
    ):
        """
        input shape is assumed to be (N, 4, 24, 24)
        """
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 16, 5, stride=2),
                nn.Conv2d(16, 20, 5, stride=2),
                nn.Conv2d(20, 20, 3, stride=1),
            ]
        )
        self.output = nn.Linear(20, n_output_channels)

        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.view(h.size(0), -1)
        return self.activation(self.output(h_flat))
