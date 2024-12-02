import torch
import torch.nn as nn
from models import *
from needle.nn.nn_sequence import RNN, LSTM, Embedding
from needle.nn import default_device
from needle.nn import Parameter
import needle as ndl
import numpy as np
import random
'''
This file implements
(1) converter methods that convert modules (e.g., Embedding, RNN-based language modeling) from PyT to Needle so that I can test parity
'''


def set_pyt_seed(seed:int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    np.random.seed(seed) # taken from hw4
    random.seed(seed)


def embedding_converter(src_model: torch.nn.Embedding, device=None, dtype=None):
    vocab_size = src_model.num_embeddings
    emb_dim  = src_model.embedding_dim
    result = Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, device=device)
    result.weight = Parameter(src_model.weight.detach().numpy(), device=device, dtype=dtype, requires_grad=True)
    return result


def embedding_parity(seq_len, bs, vocab_size, emb_dim, device, dtype):
    src_model = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
    dst_model = embedding_converter(src_model, device=device, dtype=dtype)
    np.testing.assert_allclose(src_model.weight.detach().numpy(), dst_model.weight.detach().numpy(), atol=1e-5, rtol=1e-5)
    assert(src_model.weight.grad is None)
    assert(hasattr(dst_model.weight, "grad") is False)
    # X of shape (seq_len, bs)
    x = np.random.randint(0, vocab_size, size=(seq_len, bs)).astype(np.float32)
    x_pyt = torch.Tensor(x).to(torch.long)
    x_ndl = ndl.Tensor(x, device=device, dtype=dtype)
    src_model(x_pyt).sum().backward()
    dst_model(x_ndl).sum().backward()
    #np.testing.assert_allclose(model.rnn_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)
    print(np.linalg.norm(src_model.weight.grad.detach().numpy(), ord=2))
    print(np.linalg.norm(dst_model.weight.grad.detach().numpy(), ord=2))
    np.testing.assert_allclose(src_model.weight.grad.detach().numpy(), dst_model.weight.grad.detach().numpy(), atol=1e-5, rtol=1e-5)



def rnn_converter(pyt_model: torch.nn.RNN, device=None, dtype="float32"):
    num_layers = pyt_model.num_layers
    input_size = pyt_model.input_size
    hidden_size = pyt_model.hidden_size
    ndl_model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, device=device, dtype=dtype)

    '''
     Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`
    '''
    weight_ih_list = [getattr(pyt_model, f'weight_ih_l{i}') for i in range(num_layers)]
    weight_hh_list = [getattr(pyt_model, f'weight_hh_l{i}') for i in range(num_layers)] 
    bias_ih_list  =  [getattr(pyt_model, f'bias_ih_l{i}') for i in range(num_layers)] 
    bias_hh_list = [getattr(pyt_model, f'bias_hh_l{i}') for i in range(num_layers)] 
    assert(len(weight_ih_list)==len(weight_hh_list) == len(bias_ih_list) == len(bias_hh_list) == num_layers)
    for i in range(num_layers):
        ndl_model.rnn_cells[i].W_ih = Parameter(weight_ih_list[i].detach().numpy().transpose(), device=device, dtype=dtype, requires_grad=True) # weiz 2024-12-01 notice the transpose is important as PyT uses a column-major notation 
        ndl_model.rnn_cells[i].W_hh = Parameter(weight_hh_list[i].detach().numpy().transpose(), device=device, dtype=dtype, requires_grad=True)
        ndl_model.rnn_cells[i].bias_ih = Parameter(bias_ih_list[i].detach().numpy(), device=device, dtype=dtype, requires_grad=True)
        ndl_model.rnn_cells[i].bias_hh = Parameter(bias_hh_list[i].detach().numpy(), device=device, dtype=dtype, requires_grad=True) 
    return ndl_model
    

def rnn_parity(input_size, hidden_size, num_layers):
    '''
    By default, use Tanh for non-linearity and Bias is True
    '''
    # Direction 1: from PyT to NDL
    pyt_model = torch.nn.RNN(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers)
    ndl_model = rnn_converter(pyt_model, device=default_device(), dtype="float32")
    seq_len = 10
    bs = 16
    X_np = np.random.randn(seq_len,bs,input_size).astype("float32")
    X_pyt = torch.tensor(X_np, dtype=torch.float32)
    X_ndl = ndl.Tensor(X_np, device=default_device(), dtype="float32")
    logits_pyt, h_pyt = pyt_model(X_pyt)
    logits_ndl, h_ndl = ndl_model(X_ndl)
    np.testing.assert_allclose(logits_pyt.detach().numpy(), logits_ndl.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(h_pyt.detach().numpy(), h_ndl.detach().numpy(), atol=1e-5, rtol=1e-5)
    logits_pyt.sum().backward()
    logits_ndl.sum().backward()
    np.testing.assert_allclose(pyt_model.weight_ih_l0.grad.detach().numpy().transpose(), ndl_model.rnn_cells[0].W_ih.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.weight_hh_l0.grad.detach().numpy().transpose(), ndl_model.rnn_cells[0].W_hh.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.bias_ih_l0.grad.detach().numpy(), ndl_model.rnn_cells[0].bias_ih.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.bias_hh_l0.grad.detach().numpy(), ndl_model.rnn_cells[0].bias_hh.grad.detach().numpy(), atol=1e-5, rtol=1e-5)


#rnn_parity(input_size=30, hidden_size=10, num_layers=3)



def linear_converter(pyt_model : torch.nn.Linear = None, ndl_model: ndl.nn.Linear = None, device=default_device(), dtype="float32"):
    assert ((pyt_model is not None)^(ndl_model is not None))
    if pyt_model is not None:
        in_features = pyt_model.in_features
        out_features = pyt_model.out_features
        ndl_model = ndl.nn.Linear(in_features=in_features, out_features=out_features, device=device, dtype=dtype)
        ndl_model.weight = ndl.nn.Parameter(pyt_model.weight.detach().numpy().transpose(), device=device, dtype=dtype, requires_grad=True) # weiz 2024-12-01 torch.nn.Linear is of shape (out_features, in_features)
        ndl_model.bias = ndl.nn.Parameter(pyt_model.bias.detach().numpy(), device=device, dtype=dtype, requires_grad=True)
        return ndl_model
    else:
        in_features = ndl_model.in_features
        out_features = ndl_model.out_features
        pyt_model = torch.nn.Linear(in_features=in_features, out_features=out_features, dtype=torch.float32)
        pyt_model.weight = torch.nn.Parameter(torch.tensor(ndl_model.weight.detach().numpy().transpose(), dtype=torch.float32))
        pyt_model.bias = torch.nn.Parameter(torch.tensor(ndl_model.bias.detach().numpy(), dtype=torch.float32))
        return pyt_model
    
def linear_parity(pyt_model: torch.nn.Linear = None, ndl_model: ndl.nn.Linear = None):
    '''
    Given two models, test if they are the same
    '''
    assert(pyt_model is not None and ndl_model is not None)
    # step 1 fwd pass a tensor X
    bs=16
    in_features=pyt_model.in_features
    X_np = np.random.randn(bs,in_features).astype("float32")
    X_pyt = torch.tensor(X_np, dtype=torch.float32)
    X_ndl = ndl.Tensor(X_np, device=default_device(), dtype="float32")
    logits_pyt = pyt_model(X_pyt)
    logits_ndl= ndl_model(X_ndl)
    np.testing.assert_allclose(logits_pyt.detach().numpy(), logits_ndl.detach().numpy(), atol=1e-5, rtol=1e-5)

    # step 2 backward and compares gradients
    logits_pyt.sum().backward()
    logits_ndl.sum().backward()
    np.testing.assert_allclose(pyt_model.weight.grad.detach().numpy().transpose(), ndl_model.weight.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.bias.grad.detach().numpy(), ndl_model.bias.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    
    
def test_linear_parity(bs=16, in_features=30, out_features=10, device=default_device(), dtype="float32"):
    # Direction 1: src:pyt dest:ndl
    pyt_model = torch.nn.Linear(in_features=in_features, out_features=out_features)
    ndl_model = linear_converter(pyt_model, None, device=default_device(), dtype="float32")
    linear_parity(pyt_model, ndl_model)

    # Direction 2: dest:ndl src:pyt 
    ndl_model = ndl.nn.Linear(in_features=in_features, out_features=out_features)
    pyt_model = linear_converter(None, ndl_model=ndl_model, device=default_device(), dtype="float32")
    linear_parity(pyt_model, ndl_model)


test_linear_parity(bs=16, in_features=30, out_features=10, device=default_device(), dtype="float32")

class RNNLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(RNNLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=False) # weiz make batch_first false so that it complies with my RNN impl
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
from models import LanguageModel

def rnn_lm_converter(src_model: RNNLanguageModel, device=None, dtype="float32"):
    embedding_size = src_model.embedding_dim
    vocab_size = src_model.vocab_size
    hidden_size = src_model.hidden_size
    num_layers  = src_model.num_layers
    seq_model ="rnn" # weiz hard-coded rnn for now
    model = LanguageModel(embedding_size=embedding_size, vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, seq_model=seq_model, device=device)
    # step 1 convert the embedding layer
    

#embedding_parity(10, 16, 100, 30, device=default_device(), dtype="float32")