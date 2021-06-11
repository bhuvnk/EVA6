# Batch Normalization and Regularization

Three normalization methods seen in Current SOTAs: Batch Normalization, Group Normalization, and Layer Normalization

![All Norms](imgs/0.%20AllNormalizations.JPG "All Norms")

While looking at the image above lets go throug these:

### Batch Normalization

While using BN we calculate Mean and Std. across all images for each channel.

So if we have 32 channels, we get 32 means and 32 stds.

If Batch : (N,C,H,W)
Mean and Std. calculated along: N direction

As the name suggests we calculate this every batch - for all images in the batch. 

Shortcoming: BN depends on the size of the batch, smaller size less the impact.(Kind of)


### Layer Normalization

Layer Normalization came to get rid of the batch dependency. LN normalizes the activations along the feature direction.

If Batch : (N,C,H,W)
Mean and Std. along : C direction.

Hence, images are independent of each other while normalization.


### Group Normalization

It should have been named Group-Layer or Layer-Group Normalization, but okay.
Works similar to Layer Normalization but additionally we devide the channels into groups. Depending on the number of groups the number of parameters increase in a multiple.

(N,C,H,W)
Mean and Std.Along (H,W) axis — with gorup of channels.


## Experiments on MNIST
 
[The notebook:](https://github.com/bhuvnk/EVA6/blob/main/Session%206%20Batch%20Normalization%20and%20Regularization/Session_06_Normalization_Assignment_Submission.ipynb
 "Experiments")

Pytorch has implemenation given for all three normalizations.

LR - `0.15`  with one cycle policy
Model is trained for `20` epochs, `SGD` optimizer.


#### Model Summary

```
- Convolution Block 1
    - `Conv2D:  1, 14, 3; pad = 1; Out: 28x28x16`
    - `Conv2D: 14, 28, 3; pad = 1; Out: 28x28x32`
- Transition Block
    - `Conv2D: 28,  8, 1; Out: 24x24x8`
    - `MaxPool2D: 2x2;    Out: 12x12x8`
- Convolution Block 2
    - `Conv2D:  8,  12, 3; Out: 10x10x8`
    - `Conv2D: 12, 12, 3; Out: 8x8x16`
    - `Conv2D: 12, 16, 3; Out: 6x6x16`
    - `Conv2D: 16, 16, 3; Out: 4x4x16`
- Output GAP
    - `AdaptiveAvgPool2d: -;  Out: 1x1x14`
    - `Conv2D:  -, 10, 1; Out: 1x1x10`
```

Params : `7_864`

### Setup 1 : Groupnormalization
GroupNorm : group size = 2 

#### Group Normalization Results
![GroupNormResult](imgs/1.%20GroupNormalizationResult.png "GroupNormResult")

#### Group Normalization Misclassifications
![GroupNormMis](imgs/2.%20GroupNormalizationMis.png "GroupNormMis")


### Setup 2 : LayerNormalization
LayerNorm : Default Settings

#### LayerNormalization Results
![LayerNormResult](imgs/3.%20LayerNormalizationResult.png "LayerNormResult")

#### Group Normalization Misclassifications
![LayerNormMis](imgs/4.%20LayerNormalizationMis.png "LayerNormMis")


### Setup 3 : BatchNormalization + l1
BatchNorm + L1 : lambda of `0.0001`

#### BatchNormalization Results
![BatchNormaResult](imgs/5.%20BatchNormalizationResult.png "BatchNormaResult")

#### Group Normalization Misclassifications
![BatchNormMis](imgs/6.%20BatchNormalizationMis.png "BatchNormMis")
LayerNorm : default values



### Results

![GroupNorm, LayerNorm, BatchNorm+L1](imgs/7.%20CombinedResults.png "BatchNorm+L1, GroupNorm, LayerNorm")


### Inference

1. LayerNorm was most stable while training, than rest two.
2. Adding L1 as regularization with Batch Norm makes the training very jumpy and unstable, specially with high Lambda. But low lambda gave better results.
3. Result on all Normalizations are almost similar when it comes to test data
4. BatchNorm with L1(low value of L1) seems to be the best approach to take based on current results.
