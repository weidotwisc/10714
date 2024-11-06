"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy as np # weiz 2024-09-28 i need np.argsort for the permute() back prop

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api
import needle as ndl

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

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
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar # weiz 2024 __pow__ is available in NDArray.py
        #return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return self.scalar * (node.inputs[0] ** (self.scalar-1)) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b # weiz 2024-06-12  NDArray doesn't have divide
        # return array_api.divide(a, b)
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
        self.scalar = scalar #np.float32(scalar)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        #f(isinstance(self.scalar, int)): # weiz 2024-02-03, ugly hack to make the setter type check pass TODO!!
            #pass
            #self.scalar = a.dtype.type(self.scalar)
            #self.scalar = np.float32(self.scalar) # weiz 2024-10-20, seems self.scalar = a.dtype.type(self.scalar) no longer works
        # return array_api.divide(a, self.scalar) # weiz before 2024-06-09 was using this as numpy has divide() method,  but with CUDA and CPU backend there is no divide function
        if BACKEND == "np":
            self.scalar = a.dtype.type(self.scalar)
        else:
            pass # C language and hopefully later triton will also honor the same type casting aka cast it to fp32
        return  a / self.scalar # weiz 2024-06-09 use /, which probably uses __truediv__ method in NDArray
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #result = ndl.Tensor(1/self.scalar) * (out_grad) # weiz 2023-12-25, for bwd pass, result seems have to be ndl tensor type ?, whereas fwd has no such type constraint ?
        result = out_grad / self.scalar # weiz 2023-01-02 seems that even I don't have to cast it to ndl.Tensor, maybe because *operator make it result a class of ndl.Tensor
        return result
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if(axes is None): # these two lines are added by weiz 2023-12-22
            axes=(-2,-1) # weiz 2024-06-13, make axes to be tuple instead of list
            #axes=(0,1) # weiz 2024-06-13, make axes to be tuple instead of list
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # weiz 2024-06-13 I implemented swapaxes in NDArray to minimize code change here, otherwise, I would need to condition on backend as numpy has swapaxes() but no permute()
        # whereas ndarray has permute but no swapaxes
        return array_api.swapaxes(a, self.axes[0], self.axes[1]) # weiz: 2023-12-22, this function is really about swap  axes
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)

# weiz 2024-09-28
# For HW4 conv backward, i need to permute kernel and input
class Permute(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.permute(a, self.axes) 

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return permute(out_grad, np.argsort(self.axes))
        ### END YOUR SOLUTION
def permute(a, axes=None):
    return Permute(axes)(a) # weiz 2024-09-28, look at __call__ in TensorOp to see how a is passed as the input edge for the compute graph
# end of weiz 2024-09-28 implemenation of Permute ops

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if(BACKEND == "np"): # weiz 2024-11-01 add numpy backend support in hw4 so keep the backends as close as possible
            return array_api.reshape(a, self.shape)
        else: # backedn is nd or nd_cuda
            return a.compact().reshape(self.shape) # weiz 2024-10-29 when using NDArray backend need to compact() a first as a might not be the compact version, but reshape in NDArray requires it to be compact()!!
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
        if(axes is not None):
            if(type(axes) is not tuple):
                assert(type(axes) is int)
                self.axes=(axes,) # make it a tuple weiz 2024-01-02
            else:
                self.axes = axes
        else:
            self.axes = None
            #self.axes = tuple(range(len(self.)))


    def compute(self, a): # weiz 2024-06-19, note a is of NDArray
        ### BEGIN YOUR SOLUTION
        # weiz 2024-06-19 support reduce over multiple axes
        if self.axes is not None:
            reverse_axes = tuple(sorted(self.axes, reverse=True))
            for axis in reverse_axes:
                a = array_api.sum(a, axis=axis, keepdims=False) # still need to make keepdims to be false as this is the default reduce_view_out default and make tests happy
            return a
        else:
            #reverse_axes = (range(len(a.shape)))[::-1]
            return array_api.sum(a, axis=self.axes, keepdims=False) # weiz 2024-06-19, note keepdims=True didn't really work because the reduce_view_out call didn't support keepdims when axis is None, but weiz had fixed it
        # end of weiz 2024-06-19 support reduce over multiple axes
    
        #return array_api.sum(a, axis=self.axes) # used to be just one-liner if backend is numpy, now we need to support reduce over multiple axes
        
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
        return a@b # weiz 2024-06-12  mamtmul not available in NDArray.py, but __matmul__ is
        #return array_api.matmul(a,b)
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
        return -a # weiz 2024-06-12, negative() is not in NDArray
        #return array_api.negative(a)
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
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0].realize_cached_data() # weiz 2024-10-30, this works because NDArray implements __gt__ dunder and returns an NDArray 
        return out_grad * Tensor(input > 0, device=out_grad.device, dtype=out_grad.dtype) # # weiz 2024-10-30, this works because NDArray implements __gt__ dunder and returns an NDArray 
        ### END YOUR SOLUTION



def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0])**2)
        ### END YOUR SOLUTION

# note a is of type of Value (or Tensor)
def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # weiz 2024-06-30, note args is really tuple of NDArray and returned stack_tensor is NDArray, so it is consisten with parent class Op's signature. Python doesn't seem to enforce type-checking
        #                  not sure what is this args: TensorTyple --> Tensor trying to achieve here.

        # step1 assert all tensors have the same shape
        unit_tensor_shape = args[0].shape
        for t in args:
            assert(t.shape == unit_tensor_shape)

        # step 2 allocate stacked_tensor
        stacked_tensor_shape_as_list = list(unit_tensor_shape)#
        stacked_tensor_shape_as_list.insert(self.axis, len(args))
        #stacked_tensor = array_api.empty(tuple(stacked_tensor_shape_as_list)) # use backend empty() method to create memory for stacked tensor
        
        # print(array_api)
        #<module 'needle.backend_ndarray' from '/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/backend_ndarray/__init__.py'>
        # dir(array_api)
        # ['BackendDevice', 'NDArray', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'all_devices', 'array', 'broadcast_to', 'builtins', 'cpu', 'cpu_numpy', 'cuda', 'default_device', 'empty', 'exp', 'flip', 'full', 'log', 'math', 'max', 'maximum', 'ndarray', 'ndarray_backend_cpu', 'ndarray_backend_cuda', 'ndarray_backend_numpy', 'np', 'operator', 'prod', 'reduce', 'reshape', 'squeeze', 'sum', 'swapaxes', 'tanh']

        # empty() method is available in ndarray.py 
        stacked_tensor = array_api.empty(tuple(stacked_tensor_shape_as_list), device=args[0].device) # use backend empty() method to create memory for stacked tensor, also we need to provide device argument, otherwise it will generate a numpy backend
        
        # step 3 for each portion along self.axis assign tensor
        for i_th_tensor in range(len(args)):
            indexing_tuple = tuple(slice(None) if i != self.axis else i_th_tensor for i in range(stacked_tensor.ndim)) # slice(None) means includes all the elements in that dimension
            stacked_tensor[indexing_tuple] = args[i_th_tensor]
        return stacked_tensor
        #raise NotImplementedError()
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        result_list = []
        split_len = A.shape[self.axis]
        for i_th_tensor in range(split_len):
            indexing_tuple = tuple(slice(None) if i != self.axis else i_th_tensor for i in range(A.ndim))
            single_tensor = array_api.squeeze(A[indexing_tuple].compact(), self.axis)
            result_list.append(single_tensor)
        return tuple(result_list)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes) 
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.dilate(self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.undilate(self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


# weiz 2024-09-30 implement filter dilation
class FilterDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.filterdilate(self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #return undilate(out_grad, self.axes, self.dilation)
        return None
        ### END YOUR SOLUTION


def filterdilate(a, axes, dilation):
    print("weiz filter dilate")
    return FilterDilate(axes, dilation)(a)
# weiz 2024-09-30 implement filter dilation


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    # weiz 2024-07-30, A is Z, B is W
    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        #  pad A first 
        print("[Conv] A.shape: ", A.shape)
        A = A.pad( ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)) )
        print("[Conv] A_pad.shape: ", A.shape)
        N,H,W, C_in = A.shape
        Ns,Hs,Ws,C_ins = A.strides
        K, _, _, C_out = B.shape
        inner_dim = K * K* C_in

        # when no convolution striding
        # Z_shape = (N, H-K+1, W-K+1, K, K, C_in) 
        # Z_strides = (Ns,Hs,Ws,Hs,Ws, C_ins)
        # Z = A.as_strided(shape=Z_shape, strides=Z_strides).compact()
        # Z = Z.reshape((N*(H-K+1)*(W-K+1), inner_dim))
        # W_kernel = B.compact().reshape((inner_dim, C_out)) # weiz 2024-09-08, bug fix , I was using W in LHS, and W is unfortunately also used as in shape calculation two lines below
        # out = Z @ W_kernel
        # out = out.reshape((N, H-K+1, W-K+1, C_out)) # when there is no convolution 
        
        
        # when there is convolution striding
        assert( (H-K) % self.stride == 0) # weiz 2024-10-08 just make sure it is perfectly padded for striding. other cases are handled by SnuggyConv operator
        assert( (W-K) % self.stride == 0) # weiz 2024-10-08 just make sure it is perfectly padded for striding. other cases are handled by SnuggyConv operator
        H_output = ((H-K) // self.stride) + 1 # weiz bug fix 2024-09-16, previously it was (H-K+1) // self.stride, H=5,K=3,stride=2 would fail
        W_output = ((W-K) // self.stride) + 1 # weiz bug fix 2024-09-16, previously it was (W-K+1) // self.stride, W=5,K=3,stride=2 would fail
        Z_shape = (N, H_output, W_output, K, K, C_in) 
        Z_strides = (Ns,Hs * self.stride,Ws * self.stride,Hs,Ws, C_ins)
        Z = A.as_strided(shape=Z_shape, strides=Z_strides).compact()
        Z = Z.reshape(( N*H_output*W_output, inner_dim ))
        W_kernel = B.compact().reshape((inner_dim, C_out)) # weiz 2024-09-08, bug fix , I was using W in LHS, and W is unfortunately also used as in shape calculation two lines below
                    # weiz 2024-09-29 I didn't do B.compact() before so when B(aka filter) is reshaped, needle complains
        out = Z @ W_kernel
        out = out.reshape((N, H_output, W_output, C_out))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        # weiz 2024-09-29 handle stride > 1
        #out_grad = dilate(out_grad, (1,2), self.stride-1)
        out_grad = filterdilate(out_grad, (1,2), self.stride-1)

        # weiz 2024-09-28 
        # step 1 calculate gradients w.r.t filter F
        X = node.inputs[0]
        #print("X.shape:", X.shape)
        
        F = node.inputs[1]
        K,_,_,_ = F.shape
        # weiz 2024-10-02 handle stride case
        # if(self.stride > 1):
        #     N,H,W,C_in = X.shape
        #     if((H-K)%self.stride !=0 ):
        #         assert((W-K)%self.stride !=0) # weiz we assume image is always a square, and kernel is always a square
        #         H_effective = ((H-K)//self.stride)* self.stride + K
        #         W_effective = ((W-K)//self.stride)* self.stride + K
        #         X_data_effective = X.cached_data[:,0:H_effective, 0:W_effective,:]
        #         X = Tensor.make_const(X_data_effective, requires_grad=False) # weiz: TODO: i am not sure what the impact of requires_grad=False for Hessian is yet!!!
        # weiz 2024-10-02 handle stride case

        X_perm = permute(X, (3,1,2,0))
        out_grad_perm = permute(out_grad, (1,2,0,3))   
        f_grad_perm = conv(X_perm, out_grad_perm, padding=self.padding)    
        f_grad = permute(f_grad_perm, (1,2,0,3))
        
        # step 2 calculate gradients w.r.t input X
        F_flip = flip(F, (0,1)) # flip KK axes
        F_flip_perm = transpose(F_flip) # transpose is the shortcut to permute the last two axes
        
        x_grad = conv(out_grad, F_flip_perm, padding=K-self.padding-1) # when padding is no smaller than kernel, it will become a problem, weiz 2024-10-03
        # weiz 2024-10-02 handle stride case
        # if(self.stride > 1):
        #     if((H-K)%self.stride !=0 ):
        #         assert((W-K)%self.stride !=0) # weiz we assume image is always a square, and kernel is always a square
        #         _,H_x_grad,W_x_grad,_ = x_grad.shape
        #         x_grad_cached_data = x_grad.realize_cached_data()
        #         x_grad_padded_ndarray = x_grad_cached_data.pad( ((0,0), (0, H-H_x_grad), (0, W-W_x_grad), (0,0)) )
        #         x_grad = Tensor.make_const(x_grad_padded_ndarray, requires_grad=False) # weiz: TODO: i am not sure what the impact of requires_grad=False for Hessian is yet!!!
        # weiz 2024-10-02 handle stride case
        return x_grad, f_grad
        ### END YOUR SOLUTION


class SnuggyConv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding
        self.H_pad_bottom = 0
        self.W_pad_right = 0
        self.effective_A = None # the real A during convolution

    # weiz 2024-07-30, A is Z, B is W
    # A: NDArray, B: NDArray
    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        #  pad A first 
        print("[SnuggyConv] A.shape: ", A.shape)
        
        N,H,W, C_in = A.shape
        Ns,Hs,Ws,C_ins = A.strides
        K, _, _, C_out = B.shape
        inner_dim = K * K* C_in
        assert( (H+2*self.padding-K) % self.stride !=0 )
        assert( (W+2*self.padding-K) % self.stride !=0 )
        # weiz 2024-10-08 TODO: handle the H_pad_bottom < 0 and W_pad_right < 0 cases
        
        H_pad_bottom =  (H+2*self.padding-K) // self.stride * self.stride - H - self.padding + K 
        W_pad_right =  (W+2*self.padding-K) // self.stride * self.stride - W - self.padding + K 
        self.H_pad_bottom = H_pad_bottom
        self.W_pad_right = W_pad_right

        if(H_pad_bottom >=0 and W_pad_right >=0):
            A = A.pad( ((0,0), (self.padding, self.H_pad_bottom), (self.padding, self.W_pad_right), (0,0)) )
            self.H_selector = slice(self.padding, self.padding + H)
            self.W_selector = slice(self.padding, self.padding + W)
            self.real_H_grad = H 
            self.real_W_grad = W
        else:
            assert(self.padding == 0) # weiz 2024-10-09, just weird if they decide to pad and don't pad it properly
            assert(H_pad_bottom < 0 and W_pad_right <0) # weiz 2024-10-09 just make sure we get two negative paddings, otherwise it is a weird setup
            A=A[:,0:H_pad_bottom, 0:W_pad_right,:]
            self.H_selector = slice(self.padding,None)
            self.W_selector = slice(self.padding, None)
            self.real_H_grad = H + self.padding + H_pad_bottom
            self.real_W_grad = W + self.padding + W_pad_right
            

        print("[SnuggyConv] effective A.shape: ", A.shape)
        self.effective_A = A
        N,H,W, C_in = A.shape
        Ns,Hs,Ws,C_ins = A.strides
        K, _, _, C_out = B.shape
        inner_dim = K * K* C_in

        assert((H-K)%self.stride == 0)
        assert((W-K)%self.stride == 0)
        
        
        # when there is convolution striding
        H_output = ((H-K) // self.stride) + 1 # weiz bug fix 2024-09-16, previously it was (H-K+1) // self.stride, H=5,K=3,stride=2 would fail
        W_output = ((W-K) // self.stride) + 1 # weiz bug fix 2024-09-16, previously it was (W-K+1) // self.stride, W=5,K=3,stride=2 would fail
        Z_shape = (N, H_output, W_output, K, K, C_in) 
        Z_strides = (Ns,Hs * self.stride,Ws * self.stride,Hs,Ws, C_ins)
        Z = A.as_strided(shape=Z_shape, strides=Z_strides).compact()
        Z = Z.reshape(( N*H_output*W_output, inner_dim ))
        W_kernel = B.compact().reshape((inner_dim, C_out)) # weiz 2024-09-08, bug fix , I was using W in LHS, and W is unfortunately also used as in shape calculation two lines below
                    # weiz 2024-09-29 I didn't do B.compact() before so when B(aka filter) is reshaped, needle complains
        out = Z @ W_kernel
        out = out.reshape((N, H_output, W_output, C_out))
        return out # weiz 2024-10-09 out:NDArray conv->__call__->make_from_op()->tensor.realize_cached_data()->compute() returns NDArray
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        # weiz 2024-09-29 handle stride > 1
        #out_grad = dilate(out_grad, (1,2), self.stride-1)
        out_grad = filterdilate(out_grad, (1,2), self.stride-1)

        # weiz 2024-09-28 
        # step 1 calculate gradients w.r.t filter F
        X = node.inputs[0]
        X = Tensor.make_const(self.effective_A) # weiz 2024-10-09 to get effective A
        #print("X.shape:", X.shape)
        
        F = node.inputs[1]
        K,_,_,_ = F.shape
        # weiz 2024-10-02 handle stride case
        # if(self.stride > 1):
        #     N,H,W,C_in = X.shape
        #     if((H-K)%self.stride !=0 ):
        #         assert((W-K)%self.stride !=0) # weiz we assume image is always a square, and kernel is always a square
        #         H_effective = ((H-K)//self.stride)* self.stride + K
        #         W_effective = ((W-K)//self.stride)* self.stride + K
        #         X_data_effective = X.cached_data[:,0:H_effective, 0:W_effective,:]
        #         X = Tensor.make_const(X_data_effective, requires_grad=False) # weiz: TODO: i am not sure what the impact of requires_grad=False for Hessian is yet!!!
        # weiz 2024-10-02 handle stride case

        X_perm = permute(X, (3,1,2,0))
        out_grad_perm = permute(out_grad, (1,2,0,3))   
        #f_grad_perm = conv(X_perm, out_grad_perm, padding=self.padding)    
        f_grad_perm = conv(X_perm, out_grad_perm, padding=0)  # weiz 2024-10-09 usef effective A then we don't need any padding  
        f_grad = permute(f_grad_perm, (1,2,0,3))
        
        # step 2 calculate gradients w.r.t input X
        F_flip = flip(F, (0,1)) # flip KK axes
        F_flip_perm = transpose(F_flip) # transpose is the shortcut to permute the last two axes
        
        #x_grad = conv(out_grad, F_flip_perm, padding=K-self.padding-1) # when padding is no smaller than kernel, it will become a problem, weiz 2024-10-03
        x_grad = conv(out_grad, F_flip_perm, padding=K-1) # weiz 2024-10-09 when using Snuggy, don't need any padding, we just need align the x_grad in the last step

        real_x_grad_nd_array = NDArray.make(shape=tuple(node.inputs[0].shape), device=node.inputs[0].cached_data.device)
        real_x_grad_nd_array.fill(0)
        real_x_grad_nd_array[:,0:self.real_H_grad, 0:self.real_W_grad, :] = x_grad.cached_data[:,self.H_selector,self.W_selector,:]
        x_grad = Tensor.make_const(real_x_grad_nd_array)
        # weiz 2024-10-02 handle stride case
        # if(self.stride > 1):
        #     if((H-K)%self.stride !=0 ):
        #         assert((W-K)%self.stride !=0) # weiz we assume image is always a square, and kernel is always a square
        #         _,H_x_grad,W_x_grad,_ = x_grad.shape
        #         x_grad_cached_data = x_grad.realize_cached_data()
        #         x_grad_padded_ndarray = x_grad_cached_data.pad( ((0,0), (0, H-H_x_grad), (0, W-W_x_grad), (0,0)) )
        #         x_grad = Tensor.make_const(x_grad_padded_ndarray, requires_grad=False) # weiz: TODO: i am not sure what the impact of requires_grad=False for Hessian is yet!!!
        # weiz 2024-10-02 handle stride case
        return x_grad, f_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    if(stride == 1):
        return Conv(stride, padding)(a, b)
    else:
        N,H,W,C = a.shape
        K,_,I,O = b.shape
        H=H+2*padding
        W=W+2*padding
        if((H-K) % stride==0):
            assert((W-K)%stride ==0) # weiz 2024-10-08 I assume it is a square image
            return Conv(stride, padding)(a, b)
        else:
            return SnuggyConv(stride, padding)(a,b)
        
    
