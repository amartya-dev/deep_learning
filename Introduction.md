# Introduction

## The deep in deep learning

- Deep just refers to successive layers of representations.
- Number of layers contributing to the model is called the depth of the model
- Other ML approaches focus on one-two layers of data and hence are called shallow-learning
- Although the models are called neural nets, they are in no way related to what brain does or how it learns stuff. Deep learning models are not models of the brain.

## Understanding how it works

- Deep learning maps inputs to outputs using a deep sequence of data transformation layers and these transformations are learnt by exposure to examples.
- The transformation implemented by any layer is parameterized by its weights.
- The `loss function` or the `objective function` of the  network essentially determines how far away the predictions are from the true target. The loss functions take the predictions and computes a distance score, capturing how well the network has done on this specific example.
- This score is used as a feedback signal to adjust the values of weights a little, in the direction that lowers the loss score for the current example. This is called `Backpropogation` algorithm, the central algorithm in deep learning.
- The loss function initially returns very high scores which are then adjusted according to the examples the network sees and thus, ultimately adjusts the weights to get the minimum loss values.87