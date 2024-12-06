import torch
import torch.nn as nn
from models import *
from needle.nn.nn_sequence import RNN, LSTM, Embedding
from needle.nn import default_device
from needle.nn import Parameter
import needle as ndl
import numpy as np
import random
from models import LanguageModel
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
    
def linear_parity(pyt_model: torch.nn.Linear = None, ndl_model: ndl.nn.Linear = None, bs=16):
    '''
    Given two models, test if they are the same
    '''
    assert(pyt_model is not None and ndl_model is not None)
    # step 1 fwd pass a tensor X
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
    
    
def test_linear_parity(in_features=30, out_features=10, bs=16, device=default_device(), dtype="float32"):
    # Direction 1: src:pyt dest:ndl
    pyt_model = torch.nn.Linear(in_features=in_features, out_features=out_features)
    ndl_model = linear_converter(pyt_model, None, device=device, dtype="float32")
    linear_parity(pyt_model, ndl_model, bs)

    # Direction 2: dest:ndl src:pyt 
    ndl_model = ndl.nn.Linear(in_features=in_features, out_features=out_features)
    pyt_model = linear_converter(None, ndl_model=ndl_model, device=device, dtype="float32")
    linear_parity(pyt_model, ndl_model, bs)


#test_linear_parity(in_features=30, out_features=10, bs=16, device=default_device(), dtype="float32")

def embedding_converter(pyt_model: torch.nn.Embedding = None, ndl_model: ndl.nn.Embedding = None, device=None, dtype=None):
    assert ((pyt_model is not None)^(ndl_model is not None))
    if(pyt_model is not None):
        vocab_size = pyt_model.num_embeddings
        emb_dim  = pyt_model.embedding_dim
        result = ndl.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, device=device)
        result.weight = ndl.nn.Parameter(pyt_model.weight.detach().numpy(), device=device, dtype=dtype, requires_grad=True)
        return result
    else:
        vocab_size = ndl_model.num_embeddings
        emb_dim = ndl_model.embedding_dim
        result = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        result.weight = torch.nn.Parameter(torch.tensor(ndl_model.weight.detach().numpy(), dtype=torch.float32))
        return result
    



def embedding_parity(pyt_model: torch.nn.Embedding = None, ndl_model: ndl.nn.Embedding = None, seq_len=10, bs=16):
   
    assert(pyt_model is not None and ndl_model is not None)
    vocab_size = pyt_model.num_embeddings
    # X of shape (seq_len,bs)
    x = np.random.randint(0, vocab_size, size=(seq_len, bs)).astype(np.float32)
    x_pyt = torch.Tensor(x).to(torch.long)
    x_ndl = ndl.Tensor(x, device=ndl_model.device, dtype=ndl_model.dtype)
    
    # step 1 test forward
    embeddings_pyt = pyt_model(x_pyt)
    embeddings_ndl = ndl_model(x_ndl)
    np.testing.assert_allclose(embeddings_pyt.detach().numpy(), embeddings_ndl.detach().numpy(), atol=1e-5, rtol=1e-5)
    
    # step 2 test backward
    embeddings_pyt.sum().backward()
    embeddings_ndl.sum().backward()
    #np.testing.assert_allclose(model.rnn_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)
    print(np.linalg.norm(pyt_model.weight.grad.detach().numpy(), ord=2))
    print(np.linalg.norm(ndl_model.weight.grad.detach().numpy(), ord=2))
    np.testing.assert_allclose(pyt_model.weight.grad.detach().numpy(), ndl_model.weight.grad.detach().numpy(), atol=1e-5, rtol=1e-5)


def test_embedding_parity(vocab_size, emb_dim, seq_len, bs, device, dtype):
    # Direction 1: src:pyt dest:ndl
    pyt_model = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)  
    ndl_model = embedding_converter(pyt_model=pyt_model, ndl_model=None, device=device, dtype=dtype)   
    embedding_parity(pyt_model=pyt_model, ndl_model=ndl_model, seq_len=seq_len, bs=bs)
    

    # Direction 2: dest:ndl src:pyt 
    pyt_model = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)  
    ndl_model = embedding_converter(pyt_model=pyt_model, ndl_model=None, device=device, dtype=dtype)   
    embedding_parity(pyt_model=pyt_model, ndl_model=ndl_model, seq_len=seq_len, bs=bs)
    

#test_embedding_parity(vocab_size=100, emb_dim=30, seq_len=10, bs=16, device=default_device(), dtype="float32")
    

def rnn_converter(pyt_model: torch.nn.RNN = None, ndl_model:ndl.nn.RNN = None, device=None, dtype="float32"):
    assert ((pyt_model is not None)^(ndl_model is not None))
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
    if (pyt_model is not None):
        num_layers = pyt_model.num_layers
        input_size = pyt_model.input_size
        hidden_size = pyt_model.hidden_size
        ndl_model = ndl.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, device=device, dtype=dtype)

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
    else:
        num_layers = ndl_model.num_layers
        input_size = ndl_model.input_size
        hidden_size = ndl_model.hidden_size
        pyt_model = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dtype=torch.float32)
        for i in range(num_layers):
            # Extract weights and biases from ndl_model
            W_ih = ndl_model.rnn_cells[i].W_ih.detach().numpy().transpose()  # Reverse transpose
            W_hh = ndl_model.rnn_cells[i].W_hh.detach().numpy().transpose()  # Reverse transpose
            bias_ih = ndl_model.rnn_cells[i].bias_ih.detach().numpy()
            bias_hh = ndl_model.rnn_cells[i].bias_hh.detach().numpy()

            # Update pyt_model weights and biases using Parameter
            setattr(pyt_model, f'weight_ih_l{i}', torch.nn.Parameter(torch.tensor(W_ih, dtype=pyt_model.weight_ih_l0.dtype)))
            setattr(pyt_model, f'weight_hh_l{i}', torch.nn.Parameter(torch.tensor(W_hh, dtype=pyt_model.weight_hh_l0.dtype)))
            setattr(pyt_model, f'bias_ih_l{i}', torch.nn.Parameter(torch.tensor(bias_ih, dtype=pyt_model.bias_ih_l0.dtype)))
            setattr(pyt_model, f'bias_hh_l{i}', torch.nn.Parameter(torch.tensor(bias_hh, dtype=pyt_model.bias_hh_l0.dtype)))

        return pyt_model

    

def rnn_parity(pyt_model: torch.nn.RNN = None, ndl_model: ndl.nn.RNN = None, seq_len=10, bs=16):
    '''
    By default, use Tanh for non-linearity and Bias is True
    '''
    assert (pyt_model is not None and ndl_model is not None)
    input_size = pyt_model.input_size
    hidden_size = pyt_model.hidden_size
    num_layers = pyt_model.num_layers
    # Step 1 run forward
    X_np = np.random.randn(seq_len,bs,input_size).astype("float32")
    X_pyt = torch.tensor(X_np, dtype=torch.float32)
    X_ndl = ndl.Tensor(X_np, device=ndl_model.device, dtype="float32")
    logits_pyt, h_pyt = pyt_model(X_pyt)
    logits_ndl, h_ndl = ndl_model(X_ndl)
    np.testing.assert_allclose(logits_pyt.detach().numpy(), logits_ndl.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(h_pyt.detach().numpy(), h_ndl.detach().numpy(), atol=1e-5, rtol=1e-5)

    # Step 2 run backward()
    logits_pyt.sum().backward()
    logits_ndl.sum().backward()
    np.testing.assert_allclose(pyt_model.weight_ih_l0.grad.detach().numpy().transpose(), ndl_model.rnn_cells[0].W_ih.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.weight_hh_l0.grad.detach().numpy().transpose(), ndl_model.rnn_cells[0].W_hh.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.bias_ih_l0.grad.detach().numpy(), ndl_model.rnn_cells[0].bias_ih.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.bias_hh_l0.grad.detach().numpy(), ndl_model.rnn_cells[0].bias_hh.grad.detach().numpy(), atol=1e-5, rtol=1e-5)


def test_rnn_parity(input_size=30, hidden_size=10, num_layers=3, seq_len=10, bs=16, device=default_device(), dtype="float32"):
    # direction 1: src: pyt dest: ndl
    pyt_model = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dtype=torch.float32)
    ndl_model = rnn_converter(pyt_model=pyt_model, ndl_model=None, device=device, dtype="float32")
    rnn_parity(pyt_model=pyt_model, ndl_model=ndl_model, seq_len=seq_len, bs=bs)
    # direction 2: src: ndl dest: pyt
    ndl_model = ndl.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dtype=dtype, device=device)
    pyt_model = rnn_converter(pyt_model=None, ndl_model=ndl_model, device=device, dtype=dtype)
    rnn_parity(pyt_model=pyt_model, ndl_model=ndl_model, seq_len=seq_len, bs=bs)

#test_rnn_parity(input_size=30, hidden_size=10, num_layers=3, seq_len=10, bs=16, device=default_device(), dtype="float32")


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

def rnnlm_converter(pyt_model: RNNLanguageModel = None , ndl_model: LanguageModel = None, device=default_device(), dtype="float32"):
    assert ((pyt_model is not None)^(ndl_model is not None))

    if (pyt_model is not None):
        embedding_size = pyt_model.embedding_dim
        vocab_size = pyt_model.vocab_size
        hidden_size = pyt_model.hidden_size
        num_layers  = pyt_model.num_layers
        seq_model ="rnn" # weiz hard-coded rnn for now
        ndl_model = LanguageModel(embedding_size=embedding_size, output_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, seq_model=seq_model, device=device)
        # step 1 convert the embedding layer
        ndl_model.embedding_layer = embedding_converter(pyt_model=pyt_model.embedding, ndl_model=None, device=device, dtype=dtype)
        # step 2 convert the rnn layer
        ndl_model.seq_model = rnn_converter(pyt_model=pyt_model.rnn, ndl_model=None, device=device, dtype=dtype)
        # step 3 convert the linear layer
        ndl_model.linear_layer = linear_converter(pyt_model=pyt_model.fc, ndl_model=None, device=device, dtype=dtype)
        return ndl_model
    else:
        embedding_size = ndl_model.embedding_size
        vocab_size = ndl_model.vocab_size
        hidden_size = ndl_model.hidden_size
        num_layers = ndl_model.num_layer
        pyt_model = RNNLanguageModel(vocab_size=vocab_size, embedding_dim=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
        # step 1 convert the embedding layer
        pyt_model.embedding = embedding_converter(pyt_model=None, ndl_model=ndl_model.embedding_layer, device=device, dtype=dtype)
        # step 2 convert the rnn layer
        pyt_model.rnn = rnn_converter(pyt_model=None, ndl_model=ndl_model.seq_model, device=device, dtype=dtype)
        # step 3 convert the linear layer
        pyt_model.fc = linear_converter(pyt_model=None, ndl_model=ndl_model.linear_layer, device=device, dtype=dtype)
        return pyt_model 

def rnnlm_parity(pyt_model: RNNLanguageModel = None, ndl_model: LanguageModel = None, seq_len=10, bs=16):
    assert(pyt_model is not None and ndl_model is not None)
    hidden_size = pyt_model.hidden_size
    num_layers = pyt_model.num_layers
    vocab_size = pyt_model.vocab_size
    # Step 1 run forward
    X_np = np.random.randint(0, vocab_size, size=(seq_len, bs)).astype(np.float32)
    X_pyt = torch.tensor(X_np, dtype=torch.long)
    X_ndl = ndl.Tensor(X_np, device=ndl_model.device, dtype="float32")
    init_hidden_np = np.zeros((num_layers, bs, hidden_size)).astype("float32")
    init_hidden_pyt = torch.tensor(init_hidden_np, dtype=torch.float32)
    init_hidden_ndl = ndl.Tensor(init_hidden_np, device=ndl_model.device, dtype="float32")
    logits_pyt, h_pyt = pyt_model(X_pyt, init_hidden_pyt)
    logits_ndl, h_ndl = ndl_model(X_ndl, init_hidden_ndl)
    np.testing.assert_allclose(logits_pyt.detach().numpy().reshape(seq_len*bs, -1), logits_ndl.detach().numpy(), atol=1e-5, rtol=1e-5) # notice in pyt_model, the output is (seq_len, bs, vocab_size) ndl_model the output is (seq_len*bs, vocab), so we need to reshape so that they are consistent
    np.testing.assert_allclose(h_pyt.detach().numpy(), h_ndl.detach().numpy(), atol=1e-5, rtol=1e-5)

    # Step 2 run backward()
    logits_pyt.sum().backward()
    logits_ndl.sum().backward()
    np.testing.assert_allclose(pyt_model.rnn.weight_ih_l0.grad.detach().numpy().transpose(), ndl_model.seq_model.rnn_cells[0].W_ih.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.rnn.weight_hh_l0.grad.detach().numpy().transpose(), ndl_model.seq_model.rnn_cells[0].W_hh.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.rnn.bias_ih_l0.grad.detach().numpy(), ndl_model.seq_model.rnn_cells[0].bias_ih.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.rnn.bias_hh_l0.grad.detach().numpy(), ndl_model.seq_model.rnn_cells[0].bias_hh.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.embedding.weight.grad.detach().numpy(), ndl_model.embedding_layer.weight.grad.detach().numpy(), atol=1e-5, rtol=1e-5) 

def rnnlm_parity_multi_seq(pyt_model: RNNLanguageModel = None, ndl_model: LanguageModel = None, seq_len=10, bs=16):
    assert(pyt_model is not None and ndl_model is not None)
    hidden_size = pyt_model.hidden_size
    num_layers = pyt_model.num_layers
    vocab_size = pyt_model.vocab_size
    # Step 1 run forward
    X_np_seq1 = np.random.randint(0, vocab_size, size=(seq_len, bs)).astype(np.float32)
    X_pyt_seq1 = torch.tensor(X_np_seq1, dtype=torch.long)
    X_ndl_seq1 = ndl.Tensor(X_np_seq1, device=ndl_model.device, dtype="float32")
    X_np_seq2 = np.random.randint(0, vocab_size, size=(seq_len, bs)).astype(np.float32)
    X_pyt_seq2 = torch.tensor(X_np_seq2, dtype=torch.long)
    X_ndl_seq2 = ndl.Tensor(X_np_seq2, device=ndl_model.device, dtype="float32")

    init_hidden_np = np.zeros((num_layers, bs, hidden_size)).astype("float32")
    init_hidden_pyt = torch.tensor(init_hidden_np, dtype=torch.float32)
    init_hidden_ndl = ndl.Tensor(init_hidden_np, device=ndl_model.device, dtype="float32")
    logits_pyt, h_pyt = pyt_model(X_pyt_seq1, init_hidden_pyt)
    logits_ndl, h_ndl = ndl_model(X_ndl_seq1, init_hidden_ndl)
    h_pyt.detach()
    h_ndl.detach()
    logits_pyt, h_pyt = pyt_model(X_pyt_seq2, h_pyt)
    logits_ndl, h_ndl = ndl_model(X_ndl_seq2, h_ndl)
    np.testing.assert_allclose(logits_pyt.detach().numpy().reshape(seq_len*bs, -1), logits_ndl.detach().numpy(), atol=1e-5, rtol=1e-5) # notice in pyt_model, the output is (seq_len, bs, vocab_size) ndl_model the output is (seq_len*bs, vocab), so we need to reshape so that they are consistent
    np.testing.assert_allclose(h_pyt.detach().numpy(), h_ndl.detach().numpy(), atol=1e-5, rtol=1e-5)

    # Step 2 run backward()
    logits_pyt.sum().backward()
    logits_ndl.sum().backward()
    np.testing.assert_allclose(pyt_model.rnn.weight_ih_l0.grad.detach().numpy().transpose(), ndl_model.seq_model.rnn_cells[0].W_ih.grad.detach().numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(pyt_model.rnn.weight_hh_l0.grad.detach().numpy().transpose(), ndl_model.seq_model.rnn_cells[0].W_hh.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.rnn.bias_ih_l0.grad.detach().numpy(), ndl_model.seq_model.rnn_cells[0].bias_ih.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.rnn.bias_hh_l0.grad.detach().numpy(), ndl_model.seq_model.rnn_cells[0].bias_hh.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.embedding.weight.grad.detach().numpy(), ndl_model.embedding_layer.weight.grad.detach().numpy(), atol=1e-5, rtol=1e-5) 
       

def test_rnnlm_parity(vocab_size=100, input_size=30, hidden_size=10, num_layers=3, seq_len=10, bs=16, device=default_device(), dtype="float32"):
    # direction 1: src: pyt dest: ndl
    pyt_model = RNNLanguageModel(vocab_size=vocab_size, embedding_dim=input_size, hidden_size=hidden_size, num_layers=num_layers)
    ndl_model = rnnlm_converter(pyt_model=pyt_model, ndl_model=None, device=device, dtype=dtype)
    rnnlm_parity(pyt_model=pyt_model, ndl_model=ndl_model, seq_len=seq_len, bs=bs)
    pyt_model.zero_grad() # weiz 2024-12-06 to clear out gradients is important for pytorch models, as otherwise the gradients are accumulated
    rnnlm_parity_multi_seq(pyt_model=pyt_model, ndl_model=ndl_model, seq_len=seq_len, bs=bs)
    pyt_model.zero_grad()
    # direction 2: src: ndl dest: pyt
    ndl_model = LanguageModel(embedding_size=input_size, output_size=vocab_size, hidden_size=hidden_size,num_layers=num_layers, seq_model="rnn", device=device, dtype=dtype)
    pyt_model = rnnlm_converter(pyt_model=None, ndl_model=ndl_model, device=device, dtype=dtype)
    rnnlm_parity(pyt_model=pyt_model, ndl_model=ndl_model, seq_len=seq_len, bs=bs)
    pyt_model.zero_grad()
    rnnlm_parity_multi_seq(pyt_model=pyt_model, ndl_model=ndl_model, seq_len=seq_len, bs=bs)
    pyt_model.zero_grad()

#set_pyt_seed(42)
test_rnnlm_parity(vocab_size=100, input_size=30, hidden_size=10, num_layers=3, seq_len=10, bs=16, device=default_device(), dtype="float32")