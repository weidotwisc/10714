import math
from .init_basic import *

# weiz 2024-10-09 directly copied from hw2 not being tested in hw4
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6/(fan_in+fan_out))
    data = rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    return data
    ### END YOUR SOLUTION

# weiz 2024-10-09 directly copied from hw2 not being tested in hw4
def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in+fan_out))
    data = randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    return data
    ### END YOUR SOLUTION


# weiz 2024-10-09 kaiming_uniform is the only function that we need to immplement in this howmework, all the other functions are copied from hw2
def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape==None:
        gain = math.sqrt(2.0)
        bound = gain * math.sqrt(3 / fan_in)
        data = rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
        return data
    else:
        K,_,I,O = shape
        fan_in = K * K * I
        fan_out = K * K * O
        gain = math.sqrt(2.0)
        bound = gain * math.sqrt(3 / fan_in)
        data = rand(*shape, low=-bound, high=bound, **kwargs)
        return data
    ### END YOUR SOLUTION

# weiz 2024-10-09 directly copied from hw2 not being tested in hw4
def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan_in)
    data = randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    return data
    ### END YOUR SOLUTION


