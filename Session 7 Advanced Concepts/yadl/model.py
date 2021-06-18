import torch
import torch.nn as nn
import torch.nn.functional as F


drop_value = 0.069
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # dims : [28,28,1];
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding = 1, bias=False), # 28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(drop_value),
            # RF - 3x3

            nn.Conv2d(8, 16, 3, padding = 1, bias=False), # 28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop_value),
            # RF - 5x5
        )

        # dims : [28x28x32]
        # Bottle neck - Transition layer

        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 8, 1, bias=False), # 28
            nn.ReLU(),
            # RF - 7x7

            nn.MaxPool2d(2, 2), # 14
            # RF - 14x14
        )

        # dims : [14x14x8]
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, bias=False), # 12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(drop_value),
            # RF - 16x16

            nn.Conv2d(12, 12, 3, bias=False), # 10
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(drop_value),
            # RF - 18x18

            nn.Conv2d(12, 16, 3, bias=False), # 8
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop_value),
            # RF - 20x20

            nn.Conv2d(16, 16, 3, bias=False), # 4
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop_value),
            # RF - 22x22
        )

        # dims : 4x4x32

        # GAP Layer
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),       # 
            nn.Conv2d(16, 10, 1, bias=False), # 6
            # RF - 20x20
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.out(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)