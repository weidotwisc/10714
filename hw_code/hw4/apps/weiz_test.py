import numpy as np
import inspect
#from needle import backend_ndarray as nd
import sys
import pytest
sys.path.append("/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python")
import needle as ndl
from needle import backend_ndarray as nd
import torch
import torch.nn as nn

from functools import reduce
import operator
import timeit
import statistics

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

#test_stack_backward((5,5), 0, 1, ndl.cpu())



def ndl_test():
    v1  = ndl.Tensor([0], dtype="float32")
    v2 = ndl.exp(v1)
    v3 = v2 + 1
    v4 = v2 * v3
    v4.backward()

# ndl_test()


def cifar10_dataset():
    data_path = "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/data/cifar-10-batches-py/test_batch"
    import pickle
    with open(data_path, 'rb') as fo:
        cifar_data_file = pickle.load(fo, encoding='bytes')
        print(cifar_data_file)

#cifar10_dataset()

flip_forward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 8), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (1,2)},
    {"shape": (3, 3, 6, 8), "axes": (1,2)},
    {"shape": (10, 32, 32, 8), "axes": (2,3)},
    {"shape": (3, 3, 6, 8), "axes": (2,3)},
    {"shape": (10, 32, 32, 8), "axes": (0,1,2,3)},
]

def weiztest_flip_forward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    num_of_elem = reduce(operator.mul, shape)
    _A = np.arange(num_of_elem).reshape(shape)
    #_A = np.random.randn(*shape)
    _B = np.flip(_A, axes)
    A = ndl.Tensor(_A, device=device)
    B = ndl.flip(A, axes=axes)
    assert np.linalg.norm(A.numpy() - _A) < 1e-4

device=ndl.cpu()
params={}
params['shape']=(3,2)
params['axes']=(0,)
#weiztest_flip_forward(params, device)


pad_params = [
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (2, 2), (2, 2), (0, 0) )},
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (0, 0), (0, 0), (0, 0) )},
]

def weiztest_pad_forward(params, device):
    np.random.seed(0)
    shape, padding = params['shape'], params['padding']
    #_A = np.random.randn(*shape)
    num_of_elem = reduce(operator.mul, shape)
    _A = np.arange(num_of_elem).reshape(shape)
    _B = np.pad(_A, padding)
    
    A = nd.NDArray(_A, device=device)
    B = A.pad(padding)

    assert np.linalg.norm(A.numpy() - _A) < 1e-4

device=ndl.cpu()
params={}
params['shape']=(10, 32, 32, 8)
params['padding']=( (0, 0), (2, 2), (2, 2), (0, 0) )
#weiztest_pad_forward(params, device)


def weiztest_ndarray_indexing():
    _A = np.arange(24).reshape(2,3,4)
    A= nd.NDArray(_A, device=device)
    x=A[:]
    print(x)

def weiztest_dilate():
    _A = np.arange(4).reshape(2,2)
    A = nd.NDArray(_A, device=ndl.cpu())
    dilated_A = A.dilate(axes=(0,1), dilation=1)
    print(dilated_A)
    print(dilated_A.undilate(axes=(0,1), dilation=1))

#weiztest_dilate()


# weiz 2024-07-16 play w/ Zico's convolution code
# taken from convolution_implementation.ipynb
def conv_reference_ndl(Z, weight):
    # NHWC -> NCHW
    Z_torch = torch.tensor(Z).permute(0,3,1,2)
    
    # KKIO -> OIKK
    W_torch = torch.tensor(weight).permute(3,2,0,1)
    
    # run convolution
    out = nn.functional.conv2d(Z_torch, W_torch)
    
    # NCHW -> NHWC
    return out.permute(0,2,3,1).contiguous().numpy()

def conv_naive_ndl(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    
    out = np.zeros((N,H-K+1,W-K+1,C_out));
    for n in range(N):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for y in range(H-K+1):
                    for x in range(W-K+1):
                        for i in range(K):
                            for j in range(K):
                                out[n,y,x,c_out] += Z[n,y+i,x+j,c_in] * weight[i,j,c_in,c_out]
    return out


# Needle formatting (aka tensorflow formatting)
# Input Z: NHWC
# Weights (Kernels): KKIO
def diff_conv_ndl():
    Z = np.random.randn(10,32,32,8) # NHWC
    W = np.random.randn(3,3,8,16) 


    # weiz code
    Z= np.arange(10*32*32*8).reshape((10,32,32,8))
    W = np.arange(3*3*8*16).reshape((3,3,8,16))

    _N=1
    _C=8
    _H=9
    _W=9
    _O=16
    _I=_C
    _K=3
    
  
    Z = np.arange(_N*_I*_H*_W).reshape((_N, _I, _H, _W)).astype(int)
    Z = Z.transpose((0,2,3,1))
    
    Weights = np.arange(_O*_I*_K*_K).reshape((_O, _I, _K, _K)).astype(int)
    Weights = Weights.transpose(2,3,1,0)
    W = Weights
    # weiz code 


    out = conv_reference_ndl(Z,W)
    out2 = conv_naive_ndl(Z,W)
    print(out.shape)
    print(out2.shape)
    print(np.linalg.norm(out-out2))
    #timer =timeit.Timer(lambda: conv_naive_ndl(Z,W))
    timer =timeit.Timer(lambda: conv_reference_ndl(Z,W))
    times = timer.repeat(repeat=10, number=1)
    mean_time = statistics.mean(times)
    variance_time = statistics.variance(times)

    print(f"Execution times: {times}")
    print(f"Mean execution time: {mean_time} seconds")
    print(f"Variance in execution time: {variance_time} seconds^2")

    print("-------")
    print(out[0,4,6,12])
    print(out2[0,4,6,12])
    out = out.transpose((0,3,1,2))
    print(out[0,12,4,6])
    print("ndl diff is done")
    
#diff_conv_ndl()

# pyt conv:
# Input: NCHW
# Weights: OIKK
def conv_reference_pyt(Z, W):
    out = nn.functional.conv2d(torch.Tensor(Z), torch.Tensor(W))
    return out.contiguous().numpy()

# Z: NCHW
# Weights: OIKK
# out_indices: (n,c_out,h,w)
def debug_conv_naive_pyt(Z, Weights, out_indices:tuple):
    assert(len(out_indices) == 4)
    K = Weights.shape[-1]
    n, c_out, h, w = out_indices
    kernels = Weights[c_out]
    inputs = Z[n,:, h:h+K, w:w+K]
    result = np.sum(kernels*inputs)
    print(result)
    return result



def conv_naive_pyt(Z, Weights):
    # Z: NCHW
    # W: OIKK
    N, C, H, W = Z.shape
    O, I, K, _ = Weights.shape
    assert(C==I)
    out = np.zeros((N,O,H+1-K, W+1-K))
    # Out: NOH'W'
    for n in range(N):
        for c_in in range(I):
            for c_out in range(O):
                for h in range (H+1-K):
                    for w in range (W+1-K):
                        #print(n,c_out,h,w, " c_in:", c_in)
                        #assert(out[n,c_out,h,w] == 0)
                        for k_h in range (K):
                            for k_w in range(K):
                                #out[n][c_out][h][w] += Z[n][c_in][h+k_h][w+k_w] * Weights[c_out][c_in][k_h][k_w]
                                out[n,c_out,h,w] += Z[n,c_in,h+k_h,w+k_w] * Weights[c_out,c_in,k_h,k_w]
    return out

import numpy as np




# Pyt formatting
# Input Z: NCHW
# Weights (Kernels): OIKK
def diff_conv_pyt():
    # _N=10
    # _C=8
    # _H=32
    # _W=32
    # _O=16
    # _I=_C
    # _K=3

    _N=1
    _C=8
    _H=9
    _W=9
    _O=16
    _I=_C
    _K=3
    
    Z = np.random.randn(_N, _I, _H, _W)
    Weights = np.random.randn(_O, _I, _K, _K)
    Z = np.arange(_N*_I*_H*_W).reshape((_N, _I, _H, _W)).astype(int)
    #Z = np.ones((_N, _I, _H, _W))
    Weights = np.arange(_O*_I*_K*_K).reshape((_O, _I, _K, _K)).astype(int)
    #Weights = np.ones((_O, _I, _K, _K))
    #Weights = np.ones(_O*_I*_K*_K).reshape((_O, _I, _K, _K)) / (_K*_K) # an averaging operator
    out = conv_reference_pyt(Z,Weights)
    out2 = conv_naive_pyt(Z,Weights)
    out2 = conv2d_batch_chatgpt(Z, Weights)
    print(out.shape)
    print(out2.shape)
    print(np.linalg.norm(out-out2))
    #print(out - out2)
    diff = np.nonzero(out-out2)
    print("diff.len: ")
    print(len(diff))
    idx_lst=[]
    for _diff in diff:
        if(len(_diff) > 0):
            print(_diff[0])
            idx_lst.append(_diff[0])
    print(idx_lst)
    print(out[tuple(idx_lst)])
    print(out2[tuple(idx_lst)])
    print("-----")
    debug_conv_naive_pyt(Z, Weights, tuple(idx_lst))
    #print(out[diff[0]])
    #print(out2[diff[0]])
        #print(len(_diff))
        #print(out[_diff])
        #print(out2[_diff])

    #print(np.nonzero(out-out2))
    #print(out)

    #print(out2)
    #timer =timeit.Timer(lambda: conv_naive_pyt(Z,Weights))
    #timer =timeit.Timer(lambda: conv_reference_pyt(Z,Weights))
    #times = timer.repeat(repeat=3, number=1)
    #mean_time = statistics.mean(times)
    #variance_time = statistics.variance(times)

    #print(f"Execution times: {times}")
    #print(f"Mean execution time: {mean_time} seconds")
    #print(f"Variance in execution time: {variance_time} seconds^2")



def conv2d_batch_chatgpt(input_array, kernel):
    print("hi from chatgpt")
    # Get the dimensions of the input array and the kernel
    batch_size, in_channels, input_h, input_w = input_array.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape
    
    # Calculate the dimensions of the output array
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    
    # Initialize the output array
    output_array = np.zeros((batch_size, out_channels, output_h, output_w))
    
    # Perform the convolution for each sample in the batch
    for b in range(batch_size):
        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(output_h):
                    for j in range(output_w):
                        output_array[b, oc, i, j] += np.sum(
                            input_array[b, ic, i:i+kernel_h, j:j+kernel_w] * kernel[oc, ic]
                        )
    
    return output_array

#diff_conv_pyt()


def weiz_test_op_conv(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    _W = np.random.randn(*W_shape)*5
    _W = _W.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    W = ndl.Tensor(_W, device=device)
    y = ndl.conv(Z, W, padding=padding, stride=stride)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad=True
    out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)

Z_shape, W_shape, stride, padding = ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 0 )
backward = True
device = ndl.cpu()
weiz_test_op_conv(Z_shape, W_shape, stride, padding, backward, device)