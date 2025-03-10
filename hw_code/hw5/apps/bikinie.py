import torch
import numpy as np
from utils import set_pyt_seed
# logit is of shape (1,num_classes)
max_x = None
logits_shifted = None


def submax(x:torch.Tensor, max_tracking:bool):
    global max_x
    max_x, _ = torch.max(x, dim=-1, keepdim=True)
    max_x = max_x.broadcast_to(x.shape)
    if(max_tracking is False):
        max_x = max_x.detach()
    else:
        max_x.retain_grad()
    return x - max_x

def softmax(logits:torch.Tensor):
    probs = torch.exp(logits)
    denom = probs.sum(dim=-1)
    return probs / denom

def cross_entropy(prob:torch.Tensor, labels:torch.Tensor):
    one_hot = torch.nn.functional.one_hot(labels, num_classes=prob.shape[-1])
    loss = -1 * torch.sum(torch.log(prob) * one_hot)
    return loss

def cross_entropy_max_tracking(logits:torch.Tensor, labels:torch.Tensor, max_tracking:bool=False):
    global logits_shifted
    logits_shifted = submax(logits, max_tracking)
    logits_shifted.retain_grad()
    probs = softmax(logits_shifted)
    return cross_entropy(probs, labels)


def try_cross_entropy_softmax_track_max():
    global max_x
    global logits_shifted
    n = 3
    #logits_np = np.random.rand(1, n)
    #labels_np = np.random.randint(0, n, size=(1,))
    logits_np = np.arange(n).reshape(1,n )+ 1
    labels_np = array = np.array([n-1])

    # ground truth
    logits_pyt = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    labels_pyt = torch.tensor(labels_np, dtype=torch.long)

    
    # Step 1: Define PyTorch CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()
    # Calculate the loss
    loss = criterion(logits_pyt, labels_pyt)
    loss.backward()
    print("Cross-entropy loss(ground_truth):", loss.item())
    print(logits_pyt.grad.data)

    # Step 2: max not tracking 
    max_tracking = False
    logits_pyt = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    labels_pyt = torch.tensor(labels_np, dtype=torch.long)
    loss = cross_entropy_max_tracking(logits_pyt, labels_pyt, max_tracking)
    loss.backward()
    print("Cross-entropy loss(max not tracking):", loss.item())
    print(logits_pyt.grad.data)
    print(logits_shifted.grad.data)
    # max tracking
    max_tracking = True
    logits_pyt = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    labels_pyt = torch.tensor(labels_np, dtype=torch.long)
    loss = cross_entropy_max_tracking(logits_pyt, labels_pyt, max_tracking)
    loss.backward()
    print("Cross-entropy loss(max tracking):", loss.item())
    print(logits_pyt.grad.data)
    print(max_x.grad.data)
    print(logits_shifted.grad.data)

    

def simple_max_test():
    n = 2
    bs = 1
    #logits_np = np.random.rand(bs, n)
    logits_np = np.arange(2).reshape(bs,n) + 1
    logits_pyt = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    max_val, _ = torch.max(logits_pyt, dim=-1, keepdim=True)
    max_val.retain_grad()
    y = logits_pyt - max_val
    y.retain_grad()
    (y).sum().backward()
    print(max_val.grad.data)
    print(y.grad.data)
    print(logits_pyt.grad.data)
    print("*******")
    print("not tracking")
    logits_pyt = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    max_val, _ = torch.max(logits_pyt, dim=-1, keepdim=True)
    max_val = max_val.detach()
    y = logits_pyt - max_val
    y.retain_grad()
    (y).sum().backward()
    print(y.grad.data)
    print(logits_pyt.grad.data)
    #print(max_val.grad.data)


#set_pyt_seed(42)
# try_cross_entropy_softmax_track_max()
#simple_max_test()


