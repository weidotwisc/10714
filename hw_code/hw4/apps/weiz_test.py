import numpy as np
import inspect
#from needle import backend_ndarray as nd
import sys
import pytest
sys.path.append("/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python")
import needle as ndl
from needle import backend_ndarray as nd
import torch



#print_ndarray_funcs()
def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    #c = np.random.randn(*out.shape)
    c = np.ones(out.shape) # weiz 2024-06-23 to make it reproducible
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size): # size is the number of elements in args[i]
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    print("numerical_grad ", numerical_grad )
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    print("bwd_grad ", backward_grad)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    print(error)
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]

SUMMATION_PARAMETERS = [((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2)
]
_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

#@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
#@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def weiztest_summation_backward(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.summation, A, axes=axes)

#weiztest_summation_backward((1,1,1), None, ndl.cpu())


#GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]
#@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def weiztest_tanh_backward(shape, device):
    #_A = np.random.randn(*shape).astype(np.float32)
    _A = np.ones(shape).astype(np.float32) # weiz 2024-06-23 to make it reproducible
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.tanh, A)

#weiztest_tanh_backward((1,), ndl.cpu())



STACK_PARAMETERS = [((5, 5), 0, 1),
    ((5, 5), 0, 2),
    ((1,5,7), 2, 5)]
@pytest.mark.parametrize("shape, axis, l", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
    A_t = [torch.Tensor(_A[i]) for i in range(l)]
    out = ndl.stack(A, axis=axis)
    out_t = torch.stack(A_t, dim=axis)
    np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)

#test_stack((5,5), 0, 1, ndl.cpu())

#@pytest.mark.parametrize("shape, axis, l", STACK_PARAMETERS)
#@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack_backward(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
    A_t = [torch.Tensor(_A[i]) for i in range(l)]
    for i in range(l):
        A_t[i].requires_grad = True
    ndl.stack(A, axis=axis).sum().backward()
    torch.stack(A_t, dim=axis).sum().backward()
    for i in range(l):
        np.testing.assert_allclose(A_t[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5)

test_stack_backward((5,5), 0, 1, ndl.cpu())