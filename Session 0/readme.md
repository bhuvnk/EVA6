### 1. Channels And Kernels

Collection of similar features extracted by using Kernels are called channels. 

Each kernel's job is to extract specific kind of feature, and it creates a map of those features as an output of convolution operations where similar features are bagged together are called channels.

Kernels = Feature extractors

Channels = bag of similar features

in simple words outputs of kernel operations are our feature maps/channels. 

### 2.Why use 3X3 Kernels?

1. **Shape:** 

   1. 3x3 kernel has axis/point of symmetry; the shapes which needs an axis can only be achieved by a kernel with odd number as a dimension.

2. **Computation considerations:**

   1. **Less parameters** same for same receptive field as larger kernels:

      Sequential operation of **twice, 3x3kernels** can create same effect as **one 5x5** Kernel; and **using 3x3 thrice** in a sequence is equivalent of using **one 7x7matrix**.

      parameters saved by replacing one (5x5) with two(3x3) = 25-(9+9) = 7

      parameters saved by replacing one (7x7) with three(3x3) = 49-(9+9+9) = 22

      that scales to a large number of parameters saved when we consider whole network

   2. **Graphic cards** are now optimized and accelerated to process {3x3 kernel|batchNorm|reLU} as a single computation pipeline.



### 3. How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 ?

1. Every 3x3 kernel operation reduces the image size by 2 in each dimension.

   199 | 197 | 195 ...
   199/2 = 99.5 ~ **99 times**



in other terms **for 3x3 matrix:**

n = (input-output)/2

n=(199-1)/2=99

- note: Just using 3x3 we cannot reach 1x1 if the input shape is of even dimensions.

  - 4x4 | 2x2 <now what??>

    so if we had 200x200 we would go only till 99 operations and then we will get stuck at a 2x2 image.

    (200-1)/2 = 199/2 = 99.5~ 99 and we will have 2x2 left



### 4.  How are kernels initialised?

Kernels are basically the weights in convolutions neural networks that are optimised using gradient descent, say training. 

Wrong initial values can hinder the training process. Small init values can cause vanishing gradients, and too large values may lead to exploding gradients.

This problem can be solved, if

1. we keep mean of the init activations to zero, and
2. variance of these init activations should be same among all layers.

Xavier initialization deals with these problems nicely, considering number of layers and size of the layers.  Each weight is initialized with a small Gaussian value with mean = 0.0 and variance based on the fan-in and fan-out of the weight; fan-in is number of input nodes and fan_out is number of output/hidden nodes.

He initialization can be used with ReLU, while Xavier initialization works with tanh activations.



### 5.  What happens during the training of a DNN?

**Step 1: Initialize the weights**: Randomly select weights for the layers. ( ref Q4)

**Step 2 : Get the network output:** Input the training data and perform the calculations forward, to get the output.

**Step 3 : Calculate the error:** Calculate the error at the outputs by comparing with the labels given with training samples. Use the output error to calculate error fractions at each hidden layer

**Step 4: Update the weights :** Update the weights to reduce the error, recalculate and repeat the process of training & updating the weights for all the examples. (back-propagation)

**Step 5: Repeat till done:** Stop the training and weights updating process when we either run out of time or run out of money, or in best case the minimum error criteria is met. 



In case of training the CNNs, initial layers start detecting, edges and gradients, next layers detect patterns, then parts of objects, then objects and finally scenes.






## References:

1. Submission for EVA1
