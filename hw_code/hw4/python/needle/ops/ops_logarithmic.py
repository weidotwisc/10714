from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

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


class   LogSumExp(TensorOp):
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
        self.fwd_input_orig_shape = Z.shape
        max_z = array_api.max(Z, axis=self.axes, keepdims=True) # keep the annihilated axes as dimension 1, that is long time what I want
        #max_z_bcast = array_api.broadcast_to(max_z, Z.shape)
        z_minus_z_max = Z - max_z # because I keep the annihilated dimension as 1, so here max_z can be bcasted to Z properly
        f = array_api.exp(z_minus_z_max)
        sum_exp_z = array_api.sum(f, axis=self.axes, keepdims=True) # similar to max, keep the annihiated axes as dimension 1
        assert(sum_exp_z.shape == max_z.shape)
        self.grad_intermediate = f / sum_exp_z
        assert(self.grad_intermediate.shape == self.fwd_input_orig_shape)
        self.fwd_output_orig_shape = max_z.shape # bookeeping the fwd output right shape
        lse = array_api.squeeze(array_api.log(sum_exp_z) + max_z) # in order to make the semantic correspond to keepdims=False, the result need a reduction of axes, but the actual number stay the same
        if(lse.shape == ()): # weiz 2024-11-04 my own squeeze() is the same as numpy squeeze() so it will squeeze a scalar-value to shape (), but this course considers reduction result to be of shape (1,)
            lse = lse.reshape((1,)) # so I will do this hack to reshape it to (1,) for this special case
        return lse
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # step 1 reshape out_grad to original output shape
        # weiz 2024-10-22 add the following line to make sure out_grad is compact() so that it can be reshaped. but why ?
        #out_grad.cached_data = out_grad.cached_data.compact()
         # end of weiz 2024-10-22 add the following line to make sure out_grad is compact() so that it can be reshaped. but why ?
        out_grad = out_grad.reshape(self.fwd_output_orig_shape)
        # step 2 bcast out_grad to original input shape
        out_grad = broadcast_to(out_grad, self.fwd_input_orig_shape)
        # step 3 element-wise multiply out_grad with grad_intermediate
        return  out_grad * self.grad_intermediate
    
        # weiz 2024-10-22 make sure we use Tensor instead of NDArray to multiply with out_grad
        #return  out_grad * Tensor.make_const(self.grad_intermediate, dtype=out_grad.dtype, device=out_grad.device)
        ### END YOUR SOLUTION

class PKULogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes=(axes,)
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_original = Z.max(axis=self.axes, keepdims=True) 
        max_z_reduce = Z.max(axis=self.axes)
        return array_api.log(array_api.summation(array_api.exp(Z - max_z_original.broadcast_to(Z.shape)), axis=self.axes)) + max_z_reduce 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = Tensor(z.realize_cached_data().max(axis=self.axes, keepdims=True), device=z.device)
        exp_z = exp(z - max_z.broadcast_to(z.shape))
        sum_exp_z = summation(exp_z, axes=self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

