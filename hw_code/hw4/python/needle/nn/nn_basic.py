"""The module.
"""
from functools import reduce
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
from needle import broadcast_to, power_scalar, divide, reshape
from needle.init import *
import numpy as np
from needle.init import one_hot # weiz 2024-01-28 one-hot encoding for SoftmaxLoss calculation
from needle.ops import summation # weiz 2024-01-28 import summation for SoftmaxLoss calculation

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    # weiz 2024-10-20 add params coungting functionality
    def num_params(self):
        res = np.sum([np.prod(x.shape) for x in self.parameters()])
        return res

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(kaiming_uniform(in_features, out_features, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(kaiming_uniform(out_features, 1, requires_grad=True, device=device, dtype=dtype).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = X@self.weight
        if self.bias:
            Y = Y + self.bias.broadcast_to(Y.shape)
        return Y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape # weiz I am assuming X is the type of Tensor
        assert(len(shape) >=2)
        batch_size = shape[0]
        unit_size = reduce(lambda x, y: x * y, shape[1:], 1) # get the unit size, i.e, the product of all the elements in shape except the first one(the batch size dimension)
        return X.reshape((batch_size, unit_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

# weiz 2024-11-15 to support RNN
class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # weiz 2024-01-28. seems  from test cases logits shape always samples by classes, y is always an array of shape (samples,)
        assert(len(logits.shape)==2) # weiz assumption that the logits are always 2D tensors
        num_of_samples = logits.shape[0]
        num_of_cls = logits.shape[1]
        # target_logits = logits[array_api.arange(num_of_samples), y] # use advanced indexing in numpy not sure if ndl implements this
        LSE = ops.logsumexp(logits, axes=1)
        one_hot_encoding = one_hot(num_of_cls, y, requires_grad=False, device=logits.device, dtype=logits.dtype) # weiz 2024-10-21 add device, and dtype so that we won''t have type mismatch
        target_logits = summation(logits * one_hot_encoding, axes=1)
        loss_per_sample_vec = LSE - target_logits
        result_sum = summation(loss_per_sample_vec)
        result = result_sum / num_of_samples # as in HW0 and HW1, the softmax loss is average over a minibatch
        return result
        #raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(ones(1, dim), requires_grad=True, device=device, dtype=dtype) # weiz 2024-02-01, the fact that numpy array is almost a row vector when working with bcast make it okay to initialize it as an array as well
        self.bias = Parameter(zeros(1, dim), requires_grad=True, device=device, dtype=dtype)
        self.running_mean = zeros(dim, requires_grad=False, device=device, dtype=dtype)
        self.running_var = ones(dim, requires_grad=False, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            assert (len(x.shape) == 2)  # x should be always a 2D tensor
            m = x.shape[0]
            n = x.shape[1]
            E_X = summation(x, axes=0).reshape((1, n)) / m  # expection of X
            assert (E_X.shape == (1, n))
            C_X = x - broadcast_to(E_X, x.shape)  # centered X
            assert (C_X.shape == (m, n))
            Var_X = summation(C_X * C_X, axes=0).reshape((1, n)) / m # variance of X
            assert (Var_X.shape == (1, n))
            Std_X = power_scalar((Var_X + self.eps), 0.5)  # std of X
            assert (Std_X.shape == (1, n))
            Norm_X = divide(C_X, broadcast_to(Std_X, x.shape))  # normalized X
            assert (Norm_X.shape == (m, n))
            Y = broadcast_to(self.weight, x.shape) * Norm_X + broadcast_to(self.bias, x.shape)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * E_X.reshape(-1) # weiz crazy test case that requires running_mean must be an array
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var_X.reshape(-1) # weiz crazy test case that requires running_var must be an array
        else:
            Norm_X = (x - broadcast_to(self.running_mean, x.shape)) / broadcast_to((self.running_var + self.eps) ** (0.5), x.shape)
            Y =  self.weight.broadcast_to(x.shape) * Norm_X + broadcast_to(self.bias, x.shape)
        return Y
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(ones(1, dim), requires_grad=True, device=device, dtype=dtype)
        self.bias = Parameter(zeros(1,dim), requires_grad=True, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # weiz: x is 2D tensor, batch_size * feature_num
        assert(len(x.shape) == 2) # x should be always a 2D tensor
        m = x.shape[0]
        n = x.shape[1]
        E_X = summation(x, axes=1).reshape((m,1)) / n # expection of X
        assert(E_X.shape==(m,1))
        C_X = x - broadcast_to(E_X, x.shape) # centered X
        assert (C_X.shape == (m, n))
        Var_X = summation(C_X * C_X, axes=1).reshape((m, 1)) / n # variance of X
        assert(Var_X.shape == (m,1))
        Std_X = power_scalar((Var_X+self.eps), 0.5) # std of X
        assert (Std_X.shape == (m, 1))
        Norm_X = divide(C_X, broadcast_to(Std_X, x.shape)) # normalized X
        assert(Norm_X.shape == (m, n))
        Y = broadcast_to(self.weight, x.shape) * Norm_X + broadcast_to(self.bias, x.shape)
        return Y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if(self.training):
            # * is unpacking operator that converts tuple to sequence of numbers
            probs = randb(*x.shape, p=1-self.p, dtype=x.dtype, device=x.device) # p is the probability to be zeroed
                                                               # weiz 2024-11-04 need to set up the right dtype and device
            # and the randb impl is random_choice<=p,
            # thus randb(1-self.p) is the right percentage to be one-ed
            # e.g., p=0.2 means 20% will be zeroed,
            # thus 1/(1-0.2) ratio need be scaled
            return x * probs * (1/(1-self.p))
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

