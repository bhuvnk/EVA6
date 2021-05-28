# Part1: Simple Neural Network on Excel Sheet

## Neural Network Diagram
<img src="./images/1. Neural Network Diagram.JPG" />

## FeedForward : Calculate Errors
Feedforward equations based on the diagram above.

<img src="./images/2. Forward pass.JPG"  />

## Backpropagation Equations : Get Gradients
<img src="./images/3. Backward Pass solving gradients.JPG" />

1. We start with w5 and calculate the error delta for w5 and similarly solve error gradients for w6, w7 and w8.

2. Using the chain rule and above calculated gradients we calculate gradients for W1, w2, w3 and w4 too. Here we break the equation into two parts:

   a. output  ----> hidden layer

   b. hidden layer  ---> input layer.

---

## Training the Network
<img src="./images/4. Training.JPG" width="1000"/>

1. Initialize weights as the values given in diagram above.
2. Calculate the hidden node values and output values.
3. Calculate the Total Error by comparing outputs with Targets.
4. Get total error.
5. Use backward pass equation to get gradients for each weight against the error.
6. Reduce the weight values by the corresponding gradient values.
7. Move to next iteration with the weights from step6.
8. Repeat until you are tired.

## Results

<img src="./images/5. Error vs Iterations.JPG" width="500"/>

<img src="./images/6. Impact of Learinging rate.JPG" width="500"/>



# Part 2 : 99.5% Test accuracy on MNIST with less than 13000 parameters.

### Few of the Architectural Considerations :

1. Number of Parameters: **12968**
2. 3x3 Conv Layers = 2  conv blocks,  (each with 3 convolution layers : total 6)
3. Activation function: ReLU()
4. Batch Normalization: in each conv block
5. Dropout : 0.069 after each conv block
6. 1x1 for transition layer ( to assist Excite and Squeeze ) and for last layer
7. Maxpool in Transition Block
8. Global Average Pooling followed by 1x1 instead of FC layer
9. Receptive field of 22x22 for 24x24 image; should have gone above if images had complexity of scenes and objects
10. Learning Rate: Used One Cycle Policy for maximum achievement.
    - base lr: 0.001 | Max lr : 0.15
    - Lr cycle, epoch wise;  [1 |4 | 19] : [0.01| 0.15 | 0.001]

### Results:

Total Epochs : 19

1. Parameters: 13.8k
2. Best Train Accuracy: 99.24
3. Reached 99.41 at 12th Epoch
4.  Best Test Accuracy: 99.53 (19th Epoch)



### Network Code

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # dims : [28,28,1];
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, bias=False), # 26
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.069),
            # RF - 3x3

            nn.Conv2d(8, 16, 3, bias=False), # 24
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.069),
            # RF - 5x5

            nn.Conv2d(16, 32, 3, bias=False), # 22
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.069),
            # RF - 7x7
        )

        # dims : [22x22x32]
        # Bottle neck - Transition layer

        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 8, 1, bias=False), # 22
            nn.ReLU(),
            # RF - 7x7

            nn.MaxPool2d(2, 2), # 11
            # RF - 14x14
        )

        # dims : [11x11x8]
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, bias=False), # 9
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.069),
            # RF - 16x16

            nn.Conv2d(8, 16, 3, bias=False), # 7
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.069),
            # RF - 18x18

            nn.Conv2d(16, 32, 3, bias=False), # 5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.069),
            # RF - 20x20
        )

        # dims : 5x5x32

        # GAP Layer
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),       # 
            nn.Conv2d(32, 10, 1, bias=False), # 5
            # RF - 22x22
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.out(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)
```

 ### Network Parameter Summary

<img src="./images/7. layers and parameters.JPG" width="500"/>

### Final Results

<img src="./images/7. Mnist results.png" width="700"/>

