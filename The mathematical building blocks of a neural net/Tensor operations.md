All transformations learnt by deep neural networks can be reduced to a handful of tensor operations applied to tensors of numerical data.

Example: The `Dense` layers can be interpreted as functions which take input a 2D tensor and returns another 2D tensor, specifically, when we are using a dense layer of the form: `keras.layers.Dense(512, activation='relu')`, it essentially translates to `output = relu(dot(W, input) + b)` we have three tensor operations here:

- a dot product between input tensors and a tensor `W`
- an addition between resulting tensor and a vector `b`
- A `relu` operation, `relu` is basically `max(x, 0)`.



# Element-wise operations

- The `relu` and the `addition` are element-wise operations, i.e. operations that are applied independently to each entry in the tensors being considered. 

- Element-wise operations are highly amenable to massively parallel implementations. 

- In practice, the numpy uses Basic Linear Algebra Subprograms (BLAS) which are low-level, highly parallel, efficient tensor-manipulation routines that are typically implemented in Fortran or C. 

- Example:

  ```python
  import numpy as np
  z = x + y # Element-wise addition
  z = np.maximum(x, 0.) # ELement wise relu
  ```



# Broadcasting

- `native_add` works and supports only 2D tensors with identical shapes, in case we are dealing with let's say the `Dense` layer, a 2D tensor is added to a vector, thus when the shapes of the two tensors differ, when there is no ambiguity, the smaller tensor will be `broadcasted` to match the shape of the larger tensor. Broadcasting consists of the following steps:
  - Axes (called `broadcast axes`) are added to the smaller tensor to match the ndim of the larger tensor
  - The smaller tensor is repeated alongside the new axes to match the full shape of the larger tensor.
- Example: Consider a tensor `X` with shape `(32, 10)` and `y` with shape `(10, )`, first an empty axis is added to `y` making the shape `(1, 10)`, then `y` is repeated 32 times alongside this new axes, making the shape `(32, 10)`, where `Y[i, :] == Y` for `i` in `range(0, 32)` , now we can add `X` and `Y`.
- In terms of implementation, no new 2D tensor is created, the repetition is entirely virtual, it happens at the algorithmic level rather than at the memory level.
- Broadcasting is automatically applied for element wise operations whether it be addition, maximum or anything else.



# Tensor dot

- The element wise product operation in numpy is done using the `*` operator, the tensor dot is done using the `dot` operator:

  ```python
  import numpy as np
  z = np.dot(x, y)
  ```

- The dot product of two vectors is computed as follows:

  ```python
  def native_vector_dot(x, y):
      assert len(x.shape) == 1
      assert len(y.shape) == 1
      assert x.shape[0] == y.shape[0]
      z = 0
      for i in range(x.shape[0]):
          z += x[i] * y[i]
      return z
  ```

- The dot product generalizes to tensors with arbitrary dimension, the most common application of dot product is two matrices, a dot product of two matrices can only be taken iff `x.shape[1] == y.shape[0]`. The result is a matrix with shape `(x.shape[0], y.shape[1])`, where the coefficients are the vector products between the rows of `x` and the columns of `y`.



# Tensor reshaping

- Reshaping a tensor essentially means rearranging its rows and columns to match a target shape. Naturally the reshaped tensor has the same total number of coefficients as the initial tensor. Example:

  ```python
  x = np.array([[0., 1.],
                [2., 3.],
                [4., 5.]])
  print(x.shape)
  x = x.reshape((6, 1))
  x
  """
  array([[0.],
  	  [1.],
  	  [2.],
  	  [3.],
  	  [4.],
  	  [5.]])
  """
  ```

- Transposition is a special case of reshaping, means cols = rows and rows = cols.