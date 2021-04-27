The multi dimensional numpy arrays used in the first look are called `tensors`.

Tensors are a generalisation of matrices to an arbitrary number of dimensions, a `dimension` in context of tensors is often called an `axis`



# Scalars (0 D tensors)

- Tensor containing only one number.
- In numpy, a `float32` or a `float64` number is a scalar tensor.
- Number of axis of a tensor is also called its rank, in numpy, the number of axis can be displayed by `ndim`. 

```python
import numpy as np
x = np.array(12) --> array(12)
x.ndim --> 0
```



# Vectors (1 D tensors)

-  1 D tensor has exactly 1 axis. 

```python
x = np.array([12, 3, 6, 14]) --> array([12, 3, 6, 14])
x.ndim --> 1
```

- This vector has 5 entries and hence it is called a 5-dimensional vector. 5D vector means it has only one axis and 5 dimensions along its axis whereas a 5 D tensor means it has five axes. 
- `Dimensionality` can denote either the number of entries along a specific axis or the number of axes in a tensor.



# Matrices (2 D tensors)

- Array of vectors is a matrix, or a 2 D tensor

```python
x = np.array([[5, 78, 12, 34, 0],
             [6, 79, 88, 67, 1],
             [7, 80, 89, 68, 2]])
x.ndim --> 2
```



# Key attributes of a tensor

A tensor is defined by the following key attributes:

- **Number of axes (rank)**
- **Shape** - Tuple of integers that describes how many dimensions a tensor has along each axis. 3 D tensor will have shape like (3, 3, 5), vector like (5, ) and a scalar will have empty shape ().
- **Data Type** - Example `float32`, `uint8`, `float64` and `char` on rare occasions. String tensors do not exist in numpy because tensors live in pre-allocated, contiguous memory segments and strings being variable length cannot use such a representation.



# Manipulating tensors in Numpy

- Selecting specific elements in a tensor is called `tensor slicing`, syntax is more or less like list slicing : `tensor[i]` 

  ```python
  from tensorflow.keras.datasets import mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  
  my_slice = train_images[10:100]
  my_slice.shape --> (90, 28, 28)
  ```

- You can also specify the start index and stop index for each slice along each tensor axis, like so:

  ```python
  my_slice = train_images[10:100, :, :] # Essentially same as above but this is more detailed version
  
  # To get only 14 X 14 pixels in the bottom-right corner of all images
  my_slice = train_images[:, 14:, 14:]
  
  # Also possible to specify negative indices, they indicate position to the end of current axis
  my_slice = train_images[:, 7:-7, 7:-7] # Selects 14 X 14 pixels centered to the middle
  ```



# The notion of data batches

- In general, the first axis in all data tensors will actually be the sample axis, (also called sample dimensions).
- We do not usually process the entire data at once in deep learning, we instead break the data into small batches, when considering the batches, the first axis is called the batch axis. 



# General real world examples

- **Vector data** - 2D tensors of shape `(samples, features)`
- **Timeseries data or sequence data** - 3 D tensors of shape `(samples, timestamp, features)`
- **Images** - 4D tensors of shape `(sample, height, width, channels)` or `(samples, channels, height, width)`
- **Videos** - 5D tensors of shape `(samples, frames, height, width, channels)` or `(samples, frames, channels, height, width)`