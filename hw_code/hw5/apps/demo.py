import sys
import needle as ndl
import numpy as np
import torch 
from needle.nn import default_device
# demo tensor
def demo_tensor():
    bs=8
    in_features = 16    
    #model = torch.nn.Linear(10, 5) 
    X_np = np.random.randn(bs,in_features).astype("float32")
    X_pyt = torch.tensor(X_np, dtype=torch.float32)
    X_ndl = ndl.Tensor(X_np, device=default_device(), dtype="float32")
    print(X_ndl)

def demo_sum_gradient():
    bs = 1
    in_features = 4
    X_np = np.ones((bs,in_features)).astype("float32")
    X_ndl = ndl.Tensor(X_np, device=default_device(), dtype="float32")
    sum = ndl.summation(X_ndl)
    g = sum.backward()
    print(sum)
    print(sum.grad)
    print(sum.inputs[0].grad)


def demo_matmul_gradient():
    m = 4
    n = 2
    o =3 
    X_np = np.random.randn(m,n).astype("float32")
    Y_np = np.random.randn(n,o).astype("float32")
    X_ndl = ndl.Tensor(X_np, device=default_device(), dtype="float32")
    Y_ndl = ndl.Tensor(Y_np, device=default_device(), dtype="float32")
    Z_ndl = X_ndl @ Y_ndl 
    sum_ndl = ndl.summation(Z_ndl)
    sum_ndl.backward()
    print(X_ndl.grad)

    X_pyt = torch.tensor(X_np, dtype=torch.float32, requires_grad=True)
    Y_pyt = torch.tensor(Y_np, dtype=torch.float32, requires_grad=True)
    Z_pyt = X_pyt @ Y_pyt
    sum_pyt = torch.sum(Z_pyt)
    sum_pyt.backward()
    print(X_pyt.grad)


#demo_tensor()
#demo_sum_gradient()
demo_matmul_gradient()