"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
from .nn_basic import ReLU, Tanh

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # weiz 2024-11-17, notice sigmoid(x) = 1 / (1 + exp(-1)), we have AddScalar, DivideScalar, but we don't have DividedByScalar, thus we need **(-1) to represent reciprocal.
        return (1 + ops.exp(-x)) **(-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        sqrt_k = np.sqrt(1 / self.hidden_size)
        # notice that Parameter is Tensor subclass and Module has a list of Parameters
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=(-sqrt_k), high = sqrt_k, device=device, dtype=dtype), 
                              device=device, dtype=dtype, requires_grad=True)
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=(-sqrt_k), high = sqrt_k, device=device, dtype=dtype), 
                              device=device, dtype=dtype, requires_grad=True)
        if(bias):
            self.bias_ih = Parameter(init.rand(hidden_size, low=(-sqrt_k), high = sqrt_k, device=device, dtype=dtype), 
                              device=device, dtype=dtype, requires_grad=True)
            self.bias_hh = Parameter(init.rand(hidden_size, low=(-sqrt_k), high = sqrt_k, device=device, dtype=dtype), 
                              device=device, dtype=dtype, requires_grad=True)
        if nonlinearity == "tanh":
            self.act_func = Tanh()
        elif nonlinearity == "relu":
            self.act_func = ReLU()
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        x_proj_to_h = X @ self.W_ih 
        if h is not None:
            h_proj_to_h = h @ self.W_hh
            cell_linear_proj = x_proj_to_h + h_proj_to_h
        else:
            cell_linear_proj = x_proj_to_h
        if self.bias:
            # notice : (1) my __add__ for tensor doesn't support implict bcast, so I would need to bcast 
            # (2) my bcast supports from smaller rank to larger rank following numpy bcast rule, so i can do (hidden_size,) bcast to (bs, hidden_size)       
            cell_linear_proj = cell_linear_proj + self.bias_hh.broadcast_to(cell_linear_proj.shape) + self.bias_ih.broadcast_to(cell_linear_proj.shape)
        y = self.act_func(cell_linear_proj)
        return y
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION  
        layers =[]
        layer_1 = RNNCell(input_size=input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)
        layers.append(layer_1)
        for i in range(num_layers - 1):
            layer_i = RNNCell(input_size=hidden_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)
            layers.append(layer_i)
        self.rnn_cells = layers
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        
        if h0 is not None:
            h0_splits = ops.split(h0, axis=0) # h0_splits is now a TensorTuple of (bs, hidden_size), tuple size is num_layers
        input_splits = ops.split(X, axis=0) # input_splits is now a TensorTuple of (bs, input_size), tuple size is seq_len

        final_state_list = []
        for l,rnn_cell in enumerate(self.rnn_cells):
            if h0 is None:
                _h_t = None
            else:
                _h_t = h0_splits[l] 
            next_layer_input_splits=[]
            for t in range(seq_len):
                x = input_splits[t] # weiz 2024-11-18 the only reason that indexing t works because input_splits is a TensorTuple, which implements def __getitem__(self, index: int)
                _h_t = rnn_cell(x, _h_t)
                next_layer_input_splits.append(_h_t)
            final_state_list.append(_h_t)
            input_splits = next_layer_input_splits
        Y = ops.stack(tuple(input_splits), axis=0)
        final_states = ops.stack(tuple(final_state_list), axis=0)
        return Y, final_states
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # learnable parameters
        sqrt_k = np.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size*4, low=(-sqrt_k), high = sqrt_k, device=device, dtype=dtype), 
                              device=device, dtype=dtype, requires_grad=True)
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size*4, low=(-sqrt_k), high = sqrt_k, device=device, dtype=dtype), 
                              device=device, dtype=dtype, requires_grad=True)
        if(bias):
            self.bias_ih = Parameter(init.rand(hidden_size*4, low=(-sqrt_k), high = sqrt_k, device=device, dtype=dtype), 
                              device=device, dtype=dtype, requires_grad=True)
            self.bias_hh = Parameter(init.rand(hidden_size*4, low=(-sqrt_k), high = sqrt_k, device=device, dtype=dtype), 
                              device=device, dtype=dtype, requires_grad=True)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        x_proj_to_h = X @ self.W_ih 
        if h is not None:
            h_0, c_0 = h
            h_proj_to_h = h_0 @ self.W_hh
            cell_linear_proj = x_proj_to_h + h_proj_to_h # bs, hidden_size*4
        else:
            h_0 = None
            c_0 = None
            cell_linear_proj = x_proj_to_h
        if self.bias:
            # notice : (1) my __add__ for tensor doesn't support implict bcast, so I would need to bcast 
            # (2) my bcast supports from smaller rank to larger rank following numpy bcast rule, so i can do (hidden_size,) bcast to (bs, hidden_size)       
            cell_linear_proj = cell_linear_proj + self.bias_hh.broadcast_to(cell_linear_proj.shape) + self.bias_ih.broadcast_to(cell_linear_proj.shape) # bs, hidden_size*4
        splits_columns = ops.split(cell_linear_proj, axis=1) # split_columns is a TensorTuple of unit element, each element is bs by 1. there are 4*hidden_size columns
        splits_columns = tuple(splits_columns) # weiz 2024-11-17, this is critical, as ops.stack() will require iterables, split() returns TensorTuple, but it didn't support slicing properly in __getitem__ call
                                               # unfortuntely. So we need to convert it to a tuple first, lucily TensorTuple at least implements __getitem__ for individual index, so it can be converted to a tuple.
                                               # This really is just because we didnt support split to even size tensors, but rather we can only split to many many unit size 1 tensor.
        it = ops.stack(splits_columns[0:self.hidden_size], axis=1)
        ft = ops.stack(splits_columns[self.hidden_size: self.hidden_size*2], axis=1)
        gt = ops.stack(splits_columns[self.hidden_size*2:self.hidden_size*3], axis=1)
        ot = ops.stack(splits_columns[self.hidden_size*3:self.hidden_size*4], axis=1)
        it = self.sigmoid(it)
        ft = self.sigmoid(ft)
        gt = self.tanh(gt)
        ot = self.sigmoid(ot)
        if c_0 is None:
            ct = it * gt
        else:
            ct = c_0 * ft + it * gt
        ht = self.tanh(ct) * ot
        return ht, ct
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        layers =[]
        layer_1 = LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias,device=device, dtype=dtype)
        layers.append(layer_1)
        for i in range(num_layers - 1):
            layer_i = LSTMCell(input_size=hidden_size, hidden_size=hidden_size, bias=bias, device=device, dtype=dtype)
            layers.append(layer_i)
        self.lstm_cells = layers
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        
        if h is not None:
            h0, c0 = h
            h0_splits = ops.split(h0, axis=0) # h0_splits is now a TensorTuple of (bs, hidden_size), tuple size is num_layers
            c0_splits = ops.split(c0, axis=0) # c0_splits is now a TensorTuple of (bs, hidden_size), tuple size is num_layers
        else:
            h0 = None 
            c0 = None
        input_splits = ops.split(X, axis=0) # input_splits is now a TensorTuple of (bs, input_size), tuple size is seq_len

        final_state_list = []
        final_cell_list = []
        for l,lstm_cell in enumerate(self.lstm_cells):
            if h0 is None:
                _h_t = None
                _c_t = None
            else:
                _h_t = h0_splits[l] 
                _c_t = c0_splits[l]
            next_layer_input_splits=[]
            for t in range(seq_len):
                x = input_splits[t] # weiz 2024-11-18 the only reason that indexing t works because input_splits is a TensorTuple, which implements def __getitem__(self, index: int)
                if _h_t is None:
                    assert(_c_t is None)
                    _h_t, _c_t = lstm_cell(x, None)
                else:
                    _h_t, _c_t = lstm_cell(x, (_h_t, _c_t))
                next_layer_input_splits.append(_h_t)
            final_state_list.append(_h_t)
            final_cell_list.append(_c_t)
            input_splits = next_layer_input_splits
        Y = ops.stack(tuple(input_splits), axis=0)
        final_states = ops.stack(tuple(final_state_list), axis=0)
        final_cells = ops.stack(tuple(final_cell_list), axis=0)
        return Y, (final_states, final_cells)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION