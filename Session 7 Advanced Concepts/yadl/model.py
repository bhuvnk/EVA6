import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        
            # Depthwise Saperable
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # nn.Conv2d(64, 64, 3, dilation=8, bias=False),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
        ) # output_size = 32, receptive field: 7

        # Transition Block 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0, bias=False),
        ) # output_size = 32, receptive field: 7


        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Depthwise saperable
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, dilation=4, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ) # output_size = 16, receptive field: 16

        # Transition Block 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0, bias=False),
        ) # output_size = 16, receptive field: 16

        #  Convolution Block 3
        # Depthwise Separable Convolution
        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        ) # output_size = 4, receptive field: 54


         # Output Block 
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),       #
            nn.Conv2d(128, 10, 1, padding=0, bias=False) 
        ) # output_size = 1, receptive field: 

    def forward(self, x):
        
        # Input Block: Convolution Block 1
        x = self.convblock1(x)
        
        # Transition Block 1
        x = self.transblock1(x)
        
        # Convolution Block 2
        x = self.convblock2(x)
        
        # Transition Block 2
        x = self.transblock2(x)
        
        # Transition Block 3
        x = self.convblock3(x)
        
        # Output Block
        x = self.out(x) 

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
