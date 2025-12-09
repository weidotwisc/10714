import sys
import needle as ndl
import numpy as np
import torch 
from needle.nn import default_device
# demo tensor
def demo_tensor():
    bs=8
    in_features = 16    
    model = torch.nn.Linear(10, 5) 
    X_np = np.random.randn(bs,in_features).astype("float32")
    X_pyt = torch.tensor(X_np, dtype=torch.float32)
    X_ndl = ndl.Tensor(X_np, device=default_device(), dtype="float32")
    print(X_ndl)


demo_tensor()