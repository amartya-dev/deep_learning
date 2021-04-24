# A first look at a neural network

- It will be a practical example of using a neural network
- The dataset used is MNIST, the hello world to Deep learning :)
- MNIST contains around 60,000 training images with 10,000 test images of grayscale handwritten digits (28 X 18 pixels).



## The network architecture

- The core building block of a neural network is the layer, a data-processing module that basically acts like filter for the data. Specifically, the layer extracts representations out of the data fed into them.	
- The network in the attached example ipynb consists of two dense layers, which are densely connected (also called fully connected) neural layers.
- The second layer is a 10-way softmax layer, meaning it will return an array of 10 probability scores (summing to 1). Each score will represent the probability of the digit representing one of the 10 cases.
- The following parts are required to make the networkhe mechanism  ready for training
  - A *loss function* - How the network will be able to measure its performance on the training data and how will it learn to predict stuff correctly
  - An *optimizer* - The mechanism through which the network will update itself based on data it sees and its loss function

