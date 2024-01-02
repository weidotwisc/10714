"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

import PIL.ImageCms
import numpy as np

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api
import needle as ndl

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return self.scalar * array_api.power(node, self.scalar-1) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b = node.inputs[0], node.inputs[1]
        grad_a = (b**(-1)) * out_grad
        grad_b = (-a) * (b**(-2)) * out_grad
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        result = ndl.Tensor(1/self.scalar) * (out_grad) # weiz 2023-12-25, for bwd pass, result seems have to be ndl tensor type ?, whereas fwd has no such type constraint ?
        return result
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if(axes is None): # these two lines are added by weiz 2023-12-22
            axes=[-2,-1]
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.swapaxes(a, self.axes[0], self.axes[1]) # weiz: 2023-12-22, this function is really about swap  axes
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        orig_shape = node.inputs[0].shape
        return reshape(out_grad, orig_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape) # the broadcast really makes me feel np.array is a row-vector
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        orig_shape = node.inputs[0].shape
        new_shape = self.shape
        # step 1 construct orig shape (bcast from) and new shape (bcast to), expand orig shape to the same dimension as the new shape
        orig_shape_lst = list(orig_shape)
        new_shape_lst = list(new_shape)
        orig_shape_lst = [1]*(len(new_shape_lst)-len(orig_shape_lst))+orig_shape_lst
        # step 2 find the axes indices where bcast happened
        axis_indices = [index for index, (elem1, elem2) in enumerate(zip(new_shape_lst, orig_shape_lst)) if elem1 > elem2]
        # step 3 sum over the axes where bcast happend, reshape is needed because summation can endup with 1-dimension numpy array, and we need to make
        # it explicitly the same shape as the original shape (be it row vector or column vector)
        # e.g., gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(3, 1)), shape=(3, 3))
        #     gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 3)), shape=(3, 3))
        return summation(out_grad, tuple(axis_indices)).reshape(orig_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes


    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if(self.axes is None): # weiz 2023-12-30, when dealing with AD impl, realized self.axes could be None
            # self.axes = np.arange(len(node.inputs[0].shape)) # alternatively, one can do this, but i am not sure if this node's inputs share are going to change
            orig_shape = node.inputs[0].shape # when it is none, the out_grad is a 0-dimenional, just bcast it to the orignal shape
            return broadcast_to(out_grad, orig_shape)

        orig_shape = node.inputs[0].shape
        # step 1 get real shape of out_grad
        # as out_grad could be a 1-dim as a result of sum
        # but we want the explicit shape of out_grad
        l=[]
        for s in orig_shape:
            l.append(s)
        for s in self.axes:
            l[s] = 1
        real_shape = tuple(l)
        out_grad = out_grad.reshape(real_shape)

        # step 2 broadcast to the orig_shape
        return broadcast_to(out_grad, orig_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A = node.inputs[0]
        B = node.inputs[1]
        grad_A = matmul(out_grad, transpose(B, [-2,-1]))
        grad_B = matmul(transpose(A, [-2,-1]), out_grad)
        if(len(grad_A.shape) > len(A.shape)): # e.g., shape(A)=(5,4), shape(B)=(6,6,4,3), need to sum over the first two axes for grad_A (6,6,5,4) to (5,4)
            diff_len_A = len(grad_A.shape) - len(A.shape)
            grad_A = summation(grad_A, tuple(range(diff_len_A)))
        if (len(grad_B.shape) > len(B.shape)): # e.g., shape(A) = (6,6,5,4) , shape(B) = (4,3), need to sum over the first two axes for grad_B (6,6,4,3) to (4,3)
            diff_len_B = len(grad_B.shape) - len(B.shape)
            grad_B = summation(grad_B, tuple(range(diff_len_B)))
        assert(grad_A.shape == A.shape)
        assert(grad_B.shape == B.shape)
        return grad_A, grad_B
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        assert(out_grad.shape == node.inputs[0].shape)
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        assert(out_grad.shape == node.inputs[0].shape)
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
