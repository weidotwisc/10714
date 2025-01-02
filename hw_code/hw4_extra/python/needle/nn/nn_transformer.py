from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """

        # weiz 2024-12-28 
        # a: [N,H,T,d], b:[N,H,d,T], b_transpose:[N,H,T,d]
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:]) # a_shape: [N,H,T,1,d]
        a = a.reshape(a_shape) # a: [N,H,T,1,d]

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:]) # b_transpose_shape: [N,H,1,T,d]
        b_transpose = b_transpose.reshape(b_transpose_shape) # b: [N,H,1,T,d]

        broadcast_shape = list(a_shape) # broadcast_shape [N,H,T,1,d]
        broadcast_shape[-2] = b_transpose_shape[-2] # broadcast_shape: [N,H,T,T,d]
        a = a.broadcast_to(broadcast_shape) # a: [N,H,T,1,d] --bcast--> [N,H,T,T,d] 

        broadcast_shape = list(b_transpose_shape) # broadcast_shape: [N,H,1,T,d]
        broadcast_shape[-3] = a_shape[-3] # broadcast_shape: [N,H,T,T,d]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)# b_transpose: [N,H,1,T,d] --bcast--> [N,H,T,T,d]

        return (a * b_transpose).sum(len(a.shape) - 1) # a*b_transpose: [N,H,T,T,d]; .sum(len(a.shape)-1): [N,H,T,T], notice ops.sum() will let keepdims=False when call into ndarray.sum
                                                       # N,H,T,T is the final shape

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        qk_t = self.matmul(q, k) / np.sqrt(v_dim) # weiz 2024-12-29, was supposed to be qk^T, but matmul expect 2nd argument to be in the transpose format already
        if(self.causal):
            qk_t = qk_t + Tensor(self.create_causal_mask(queries_len, keys_values_len, self.device).broadcast_to((batch_size,num_head, queries_len,keys_values_len)), 
                                 dtype=self.dtype, device=self.device, requires_grad=False) # weiz 2024-12-29, the create_causal_mask result need to be broadcast to the first two dimensions too
        probs = self.softmax(qk_t)
        probs = self.dropout(probs)
        result = self.matmul(probs, v.transpose()) # matmul() expects 2nd argument to be in transpose format, also Tensor transpose just transpose the last two dimensions, which matches perfectly what we need
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        # step 1 layer norm and project Q,K,V
        q_flatten = ops.reshape(q, (batch_size*queries_len, q_dim))
        q_flatten_prenormed = self.prenorm_q(q_flatten)
        q_flatten_prenormed_projected = self.q_projection(q_flatten_prenormed) # q_flatten_prenormed_projected is of shape (B*T, inner_dim)

        k_flatten = ops.reshape(k, (batch_size*queries_len, k_dim))
        k_flatten_pernormed = self.prenorm_k(k_flatten)
        k_flatten_prenormed_projected = self.k_projection(k_flatten_pernormed)

        v_flatten = ops.reshape(v, (batch_size*queries_len, v_dim))
        v_flatten_prenormed = self.prenorm_v(v_flatten)
        v_flatten_prenormed_projected = self.v_projection(v_flatten_prenormed)


        # step 2 split and permute q, k, v
        q_flatten_prenormed_projected_reshaped = ops.reshape(q_flatten_prenormed_projected, (batch_size, queries_len,self.num_head, self.dim_head)) # (B,T,H,D)
        q_bthd = ops.permute(q_flatten_prenormed_projected_reshaped, (0,2,1,3)) # (B, H, T, D), in the right shape to be used by attntion calculation

        k_flatten_prenormed_projected_reshaped = ops.reshape(k_flatten_prenormed_projected, (batch_size, queries_len, self.num_head, self.dim_head))
        k_bthd = ops.permute(k_flatten_prenormed_projected_reshaped, (0,2,1,3))

        v_flatten_prenormed_projected_reshaped = ops.reshape(v_flatten_prenormed_projected, (batch_size, queries_len, self.num_head, self.dim_head))
        v_bthd = ops.permute(v_flatten_prenormed_projected_reshaped, (0,2,1,3))
        
        # step 3  compute multi-head attention activation
        x_bhtd, probs = self.attn(q_bthd, k_bthd, v_bthd)
        x_bthd = ops.permute(x_bhtd, (0,2,1,3))
        x = ops.reshape(x_bthd, (batch_size*queries_len, -1))
        
        # step 4 project to the output space
        _result = self.out_projection(x)
        result = ops.reshape(_result, (batch_size, queries_len, -1))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)