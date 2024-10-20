"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, BatchNorm2d, ReLU, Sequential, Residual


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # weiz 2024-10-14 implement the initialization
        # step 1 initialize filter Parmeters
        self.weight = Parameter(init.kaiming_uniform(0,0, (kernel_size, kernel_size, in_channels, out_channels)), 
                                device=device, dtype=dtype, requires_grad=True) # we don't really need to pass meaningful fan_in and fan_out, as shape will take care of it 

        # step 2 initialize bias Parameters
        # Per instruction, Initialize the (o,) bias tensor using uniform initialization on the interval  ± 1.0/(in_channels * kernel_size**2)**0.5
        #  def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
        bound_abs_val = 1 / (in_channels * kernel_size**2 )**0.5
        low = -1 * bound_abs_val
        high = 1 * bound_abs_val
        self.bias = Parameter(init.rand(self.out_channels, low=low,high=high, 
                                        device=device, dtype=dtype, requires_grad=True)) if bias else None

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # weiz 2024-10-15, x is of NCHW, which is the PyT format, and in Conv ops, we use NHWC, which is the TF format
        n,cin,h,w = x.shape
        x_nhwc = ops.permute(x, (0,2,3,1))
        assert((self.kernel_size-1) % 2 == 0) # in order to make sure when stride=1, we get the same input and output dimension
        padding = (self.kernel_size - 1 ) // 2
        y = ops.conv(x_nhwc, self.weight, stride=self.stride, padding = padding) # y now in nhwc format
        if self.bias:
            y = y + ops.broadcast_to(self.bias.reshape((1,1,1,self.out_channels)), y.shape) # weiz 2024-10-15, we need to make array of (out_channel,) to a tensor and then broadcast to y
        y = ops.permute(y, (0,3,1,2))
        return y

        ### END YOUR SOLUTION


class ConvBN(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # weiz 2024-10-18 conv->batchnorm2d->relu
        self.conv = Conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device, dtype=dtype)
        self.bn = BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))
    
class ResNetBasicBlock(Module):
    # conv_params_tuple is a tuple of ConvBN parameters, which is a tuple of (in_channels, out_channels, kernel_size and stride)
    def __init__(self, conv_params_tuple:tuple, bias=True, device=None, dtype="float32"):
        convbn_list=[]
        for conv_params in conv_params_tuple:
            in_channels, out_channels, kernel_size, stride = conv_params
            convbn = ConvBN(in_channels, out_channels, kernel_size, stride = stride, bias=bias, device=device,dtype=dtype)
            convbn_list.append(convbn)
        self.main_path = Sequential(*convbn_list)
        self.residual_block = Residual(self.main_path)

    def forward(self, x:Tensor) -> Tensor:
        return self.residual_block(x)