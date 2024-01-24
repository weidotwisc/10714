from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if (axes is not None):
            if (type(axes) is not tuple):
                assert (type(axes) is int)
                self.axes = (axes,)  # make it like sum axes semantic in weiz needle (aka sum semantic in numpy) weiz 2023-01-23
            else:
                self.axes = axes
        else:
            self.axes = None

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True) # keep the annihilated axes as dimension 1, that is long time what I want
        z_minus_z_max = Z - max_z
        sum_exp_z = array_api.sum(array_api.exp(z_minus_z_max), axis=self.axes, keepdims=True) # similar to max, keep the annihiated axes as dimension 1
        assert(sum_exp_z.shape == max_z.shape)
        return array_api.sum(array_api.log(sum_exp_z) + max_z, axis=self.axes) # in order to make the semantic correspond to keepdims=False, the result need a reduction of axes, but the actual number stay the same
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

