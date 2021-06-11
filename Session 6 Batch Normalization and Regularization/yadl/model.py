
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, use_batchnorm=False, use_groupnorm=False, use_layernorm=False):
        super(Net, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_groupnorm = use_groupnorm
        self.use_layernorm = use_layernorm

        # dims : [28,28,1];
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),  # 28
            nn.ReLU(),
            self._get_norm(8, 28, 28),
            nn.Dropout2d(drop_value),
            # RF - 3x3

            nn.Conv2d(8, 16, 3, padding=1, bias=False),  # 28
            nn.ReLU(),
            self._get_norm(16, 28, 28),
            nn.Dropout2d(drop_value),
            # RF - 5x5
        )

        # dims : [28x28x32]
        # Bottle neck - Transition layer

        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 8, 1, bias=False),  # 28
            nn.ReLU(),
            # RF - 7x7

            nn.MaxPool2d(2, 2),  # 14
            # RF - 14x14
        )

        # dims : [14x14x8]

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, bias=False),  # 12
            nn.ReLU(),
            self._get_norm(12, 12, 12),
            nn.Dropout2d(drop_value),
            # RF - 16x16

            nn.Conv2d(12, 12, 3, bias=False),  # 10
            nn.ReLU(),
            self._get_norm(12, 10, 10),
            nn.Dropout2d(drop_value),
            # RF - 18x18

            nn.Conv2d(12, 16, 3, bias=False),  # 8
            nn.ReLU(),
            self._get_norm(16, 8, 8),
            nn.Dropout2d(drop_value),
            # RF - 20x20

            nn.Conv2d(16, 16, 3, bias=False),  # 4
            nn.ReLU(),
            self._get_norm(16, 4, 4),
            nn.Dropout2d(drop_value),
            # RF - 22x22
        )

        # dims : 4x4x32

        # GAP Layer
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),       #
            nn.Conv2d(16, 10, 1, bias=False),  # 6
            # RF - 20x20
        )

    def _get_norm(self, inp_shape):
        if self.use_batchnorm:
            return nn.BatchNorm2d(inp_shape[0])
        elif self.use_groupnorm:
            return nn.GroupNorm(num_groups=2, num_channels=inp_shape[0])
        elif self.use_layernorm:
            return nn.LayerNorm(inp_shape)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.out(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)
