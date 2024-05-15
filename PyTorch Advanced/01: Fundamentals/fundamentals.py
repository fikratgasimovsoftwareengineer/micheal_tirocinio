"""
00. Pytorch Fundamentals

Resource notebook: https://www.learnpytorch.io/00_pytorch_fundamentals/

If you have a question: https://github.com/mrdbourke/pytorch-deep-learning/discussions
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

"""
Introduction to Tensors

Creating tensors
Pytorch tensors are created using torch.tensor() = https://pytorch.org/docs/stable/tensors.html
"""
# Scalar
scalar = torch.tensor(7)
print(scalar) # tensor(7)
# It doesn't have dimensions
print(scalar.ndim) # 0
# Get tensor back as Python int
print(scalar.item()) # 7

# Vector
vector = torch.tensor([7, 7])
print(vector) # tensor([7, 7])
print(vector.ndim) # 1
print(vector.shape) # torch.Size([2])

# MATRIX
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX)
print(MATRIX.ndim) # 2
print(MATRIX.shape) # torch.Size([2, 2])
print(MATRIX[0]) # tensor([7, 8])

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim) # 3
print(TENSOR.shape) #  torch.Size([1, 3, 3])
print(TENSOR[0])

"""
Random tensors

Why random tensors?
Random tensors are important because the way many neural networks learn
is that they start with tensors full of random numbers and then adjust
those random numbers to better represent the data
Starts with random numbers -> look at data -> update random numbers
-> look at data -> update random numbers -> .....
Torch random tensors: https://pytorch.org/docs/stable/generated/torch.rand.html
"""
# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim) # 2
# Create a random tensor with similar shape to an image tensor
# size is the first parameter
random_image_size_tensor = torch.rand(size=(3, 224, 224)) # color channel, height, widht
print(random_image_size_tensor.shape) # torch.Size([3, 224, 224])
print(random_image_size_tensor.ndim) # 3

"""
Zeros and ones
"""
# Create a tensor of all zeros
zeros = torch.zeros(3, 4)
print(zeros)
# Create a tensor of all ones
ones = torch.ones(3, 4)
print(ones)
print(ones.dtype) # torch.float32
print(random_tensor.dtype) # torch.float32

"""
Create a range of tensors and tensors-like
"""
# Use torch.arange(), range() is deprecated
one_to_ten = torch.arange(start=1, end=11, step=1)
print(one_to_ten) # tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
# Creating tensors like
ten_zeros = torch.zeros_like(one_to_ten)
print(ten_zeros) # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

"""
Tensor datatypes

Tensor datatypes is one of the 3 big errors you'll run into with PyTorch
& deep learning:
1. Tensors not right datatype
2. Tensors not right shape
3. Tensors not on the right device

Precision in computing: https://en.wikipedia.org/wiki/Precision_(computer_science)
"""
# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # what data type is the tensor
                               device=None, # what device is your tensor on
                               requires_grad=False) # whether to track gradients with these tensors operations
print(float_32_tensor) # tensor([3., 6., 9.])
print(float_32_tensor.dtype) # torch.float32
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor) # tensor([3., 6., 9.], dtype=torch.float16)
print(float_16_tensor * float_32_tensor) # tensor([ 9., 36., 81.])

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(int_32_tensor) # tensor([3, 6, 9], dtype=torch.int32)
print(float_16_tensor * int_32_tensor) # tensor([ 9., 36., 81.], dtype=torch.float16)

"""
Getting information from tensors (tensor attributes)

1. Tensors not right datatype - to do get datatype from a tensor, can use 'tenros.dtype'
2. Tensors not right shape - to get shape from a tensor, can use 'tensor.shape'
3. Tensors not on the right device - to get device from a tensor, can use 'tensor.device'
"""
# Create a tensor
some_tensor = torch.rand(3, 4)
print(some_tensor)
# Find out details about some tensor
print(f'Datatype of tensor: {some_tensor.dtype}') # torch.float32
print(f'Shape of tensor: {some_tensor.shape}') # torch.Size([3, 4])
print(f'Device tensor is on: {some_tensor.device}') # cpu

"""
Manipulation tensors (tensor operations)

Tensor operations include:
* Addition
* Substraction
* Multiplication (element-wise)
* Division
"""
# Create a tensor and add 10 to it
tensor = torch.tensor([1, 2, 3])
print(tensor + 10) # tensor([11, 12, 13])
# Multiply tensor by 10
print(tensor * 10) # tensor([10, 20, 30])
# Substract 10
print(tensor - 10) # tensor([-9, -8, -7])
# Division 10
print(tensor / 10) # tensor([0.1000, 0.2000, 0.3000])
# Try out PyTorch in-built functions
print(torch.add(tensor, 10)) # tensor([11, 12, 13])
print(torch.mul(tensor, 10)) # tensor([10, 20, 30])

"""
Matrix multiplication

Two main ways to performing multiplication in neural networks and deep learning:
1. Element-wise multiplication
2. Matrix multiplication (dot product)

More information on multiplyng matrices - https://www.mathsisfun.com/algebra/matrix-multiplying.html

There are two main rules that performing matrix multiplication needs to satisfy:
1. The INNER DIMENSIONS must match:
* (3, |2|) @ (|3|, 2) won't work (2, 3) - @ is matrix multiplication
* (2, |3|) @ (|3|, 2) will work (3, 3)
* (3, |2|) @ (|2|, 3) will work (2, 2)
2. The resulting matrix has the shape of the OUTER DIMENSIONS:
* (|2|, 3) @ (3, |2|) -> (2, 2)
* (|3|, 2) @ (2, |3|) -> (3, 3)
"""
# Element wise multiplication
print(tensor, "*", tensor) # tensor([1, 2, 3]) * tensor([1, 2, 3])
print(f'Equals: {tensor * tensor}') # Equals: tensor([1, 4, 9])
# Matrix multiplication (faster) = 246 μs - micro
print(torch.matmul(tensor, tensor)) # tensor(14)
# Matrix multiplication by hand
print((1*1) + (2*2) + (3*3)) # 14
# Matrix multiplication for loop = 2.01 ms - milli
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
print(value) # tensor(14)

"""
One of the most common errors in deep learning: shape errors
"""
# Shapes for matrix multiplication
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])
tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])
# torch.mm is an alias of torch.matmul
# print(torch.mm(tensor_A, tensor_B)) # shape error

# To fix our tensor shape issue, we can manipulate the shape of one
# of our tensors using a "transpose"
# A transpose switches the axes or dimensions of a given tensor
print(tensor_B.T) # shape = 2, 3
print(tensor_B) # shape = 3, 2
print(torch.matmul(tensor_A, tensor_B.T).shape) # 3, 3
