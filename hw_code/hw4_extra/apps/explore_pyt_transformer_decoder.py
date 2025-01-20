import torch
import numpy as np
import needle as ndl

### weiz 2025-01-13 experiment with torch.nn.TransformerEncoderLayer class, espeically play with post-norm (default) and pre-norm 

# Copy  the post-norm code from lectures
def softmax(Z):
    Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return Z / Z.sum(axis=-1, keepdims=True)

def layer_norm(Z, eps):
    return (Z - Z.mean(axis=-1, keepdims=True)) / np.sqrt(Z.var(axis=-1, keepdims=True) + eps)
    
def relu(Z):
    return np.maximum(Z,0)

def multihead_attention(X, mask, heads, W_KQV, W_out):
    N,T,d = X.shape
    K,Q,V = np.split(X@W_KQV, 3, axis=-1)
    K,Q,V = [a.reshape(N,T,heads,d//heads).swapaxes(1,2) for a in (K,Q,V)]
    
    attn = softmax(K@Q.swapaxes(-1,-2) / np.sqrt(d//heads) + mask)
    return (attn@V).swapaxes(1,2).reshape(N,T,d) @ W_out, attn

def transformer_post_norm(X, mask, heads, W_KQV, W_out, W_ff1, W_ff2, eps):
    Z = layer_norm(multihead_attention(X, mask, heads, W_KQV, W_out)[0] + X, eps)
    return layer_norm(Z + relu(Z@W_ff1)@W_ff2, eps)


def transformer_pre_norm(X, mask, heads, W_KQV, W_out, W_ff1, W_ff2, eps):
    X1 = multihead_attention(layer_norm(X, eps), mask, heads, W_KQV, W_out)[0] + X
    X2 = layer_norm(X1, eps)
    return (X1 + (relu(X2@W_ff1))@W_ff2)

T, d = 100, 256
heads = 4
N = 10
M = torch.triu(-float("inf")*torch.ones(T,T),1)
X = torch.randn(N,T,d)


def parity_transform_post_norm_numpy():
    # Test post-norm as in lecture
    trans = torch.nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=0.0, batch_first=True)
    trans.self_attn.in_proj_bias.data.zero_() # notice since in needle AttentionLayer, we don't have bias terms, just to play safe we remove self_attn's bias terms in pytorch module
                                          # though bias in pytorch linear layer is intialized with very small values.
                                          # weiz 2025-01-14 notice that in lecture, Zico didn't explicitly set the self_attn bias to be zero, probably he knew pytorch made them very small anyway
    trans.linear1.bias.data.zero_()
    trans.linear2.bias.data.zero_()
    Y_ = trans(X, M)
    
    Y = transformer_post_norm(X.numpy(), M.numpy(), heads,
                trans.self_attn.in_proj_weight.detach().numpy().T, 
                trans.self_attn.out_proj.weight.detach().numpy().T,
                trans.linear1.weight.detach().numpy().T,
                trans.linear2.weight.detach().numpy().T,
                trans.norm1.eps)
    print(np.linalg.norm(Y - Y_.detach().numpy()))
    np.testing.assert_allclose(Y, Y_.detach().numpy(), atol=1e-5, rtol=1e-5)


# Test pre-norm my own figuring-out, via numpy
def parity_transform_pre_norm_numpy():
    trans = torch.nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=0.0, batch_first=True, norm_first=True)
    trans.self_attn.in_proj_bias.data.zero_()
    trans.linear1.bias.data.zero_()
    trans.linear2.bias.data.zero_()
    Y_ = trans(X, M)
    Y = transformer_pre_norm(X.numpy(), M.numpy(), heads,
                trans.self_attn.in_proj_weight.detach().numpy().T, 
                trans.self_attn.out_proj.weight.detach().numpy().T,
                trans.linear1.weight.detach().numpy().T,
                trans.linear2.weight.detach().numpy().T,
                trans.norm1.eps)
    print(np.linalg.norm(Y - Y_.detach().numpy()))
    np.testing.assert_allclose(Y, Y_.detach().numpy(), atol=1e-5, rtol=1e-5)


def parity_transform_pre_norm_needle():
    # Test pre-norm in pyt and against my impl in ndl
    ffn_dim=128
    trans = torch.nn.TransformerEncoderLayer(d, heads, dim_feedforward=ffn_dim, dropout=0.0, batch_first=True, norm_first=True)
    trans.self_attn.in_proj_bias.data.zero_() # notice that since needle self_attention didn't implement any bias, this  unfortunately is a must-do
    #trans.linear1.bias.data.zero_() # notice since we support bias in ndl transformer linear layers, we didn't have to comment these out
    #trans.linear2.bias.data.zero_()
    q_proj_np, k_proj_np, v_proj_np = np.split(trans.self_attn.in_proj_weight.detach().numpy().T, 3 ,axis=-1)
    w_out_proj_np = trans.self_attn.out_proj.weight.detach().numpy().T
    linear1_np = trans.linear1.weight.detach().numpy().T
    linear1_bias_np = trans.linear1.bias.detach().numpy() # it is of shape (out_features,) i.e. just a numpy array
    linear2_np = trans.linear2.weight.detach().numpy().T
    linear2_bias_np = trans.linear2.bias.detach().numpy()

    Y_ = trans(X, M)

    X_ndl = ndl.Tensor(X.detach().numpy(), device=ndl.default_device(), dtype="float32")
    trans_ndl = ndl.nn.TransformerLayer(q_features=d, num_head=heads, dim_head=(d//heads), hidden_size=ffn_dim, dropout=0,causal=True, device=ndl.default_device(), dtype="float32")
    attn_layer_ndl: ndl.nn.AttentionLayer = trans_ndl.attn_residual.fn.modules[0]
    attn_layer_ndl.q_projection.weight = ndl.nn.Parameter(q_proj_np, device=ndl.default_device(), dtype="float32", requires_grad=True)
    attn_layer_ndl.k_projection.weight = ndl.nn.Parameter(k_proj_np, device=ndl.default_device(), dtype="float32", requires_grad=True)
    attn_layer_ndl.v_projection.weight = ndl.nn.Parameter(v_proj_np, device=ndl.default_device(), dtype="float32", requires_grad=True)
    attn_layer_ndl.out_projection.weight = ndl.nn.Parameter(w_out_proj_np,device=ndl.default_device(), dtype="float32", requires_grad=True)
    linear1_ndl: ndl.nn.Linear = trans_ndl.mlp_residual.fn.modules[1]
    linear1_ndl.weight = ndl.nn.Parameter(linear1_np, device=ndl.default_device(), dtype="float32", requires_grad=True)
    linear1_ndl.bias = ndl.nn.Parameter(linear1_bias_np, device=ndl.default_device(), dtype="float32", requires_grad=True)
    linear2_ndl: ndl.nn.Linear = trans_ndl.mlp_residual.fn.modules[4]
    linear2_ndl.weight = ndl.nn.Parameter(linear2_np, device=ndl.default_device(), dtype="float32", requires_grad=True)
    linear2_ndl.bias = ndl.nn.Parameter(linear2_bias_np, device=ndl.default_device(), dtype="float32", requires_grad=True)


    # step 1 test forward
    Y_ndl = trans_ndl(X_ndl)
    print(np.linalg.norm(Y_ndl.detach().numpy() - Y_.detach().numpy()))
    np.testing.assert_allclose(Y_ndl.detach().numpy(), Y_.detach().numpy(), atol=1e-5, rtol=1e-5)

    # step 2 test backward
    Y_.sum().backward()
    Y_ndl.sum().backward()
    pyt_model = trans.linear2
    ndl_model = linear2_ndl
    np.testing.assert_allclose(pyt_model.weight.grad.detach().numpy().transpose(), ndl_model.weight.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyt_model.bias.grad.detach().numpy().reshape(-1), ndl_model.bias.grad.detach().numpy().reshape(-1), atol=1e-5, rtol=1e-5)
    pyt_model = trans.linear1
    ndl_model = linear1_ndl
    np.testing.assert_allclose(pyt_model.weight.grad.detach().numpy().transpose(), ndl_model.weight.grad.detach().numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(pyt_model.bias.grad.detach().numpy().reshape(-1), ndl_model.bias.grad.detach().numpy().reshape(-1), atol=1e-4, rtol=1e-4)



    pyt_model = trans.self_attn
    ndl_model = attn_layer_ndl
    w_out_proj_grad_pyt_np = pyt_model.out_proj.weight.grad.detach().numpy().T
    w_out_proj_grad_ndl_np = ndl_model.out_projection.weight.grad.detach().numpy()
    np.testing.assert_allclose(w_out_proj_grad_pyt_np, w_out_proj_grad_ndl_np, atol=1e-4, rtol=1e-4)
    q_proj_grad_pyt_np, k_proj_grad_pyt_np, v_proj_grad_pyt_np = np.split(pyt_model.in_proj_weight.grad.detach().numpy().T, 3 ,axis=-1)
    q_proj_grad_ndl_np = ndl_model.q_projection.weight.grad.detach().numpy()
    k_proj_grad_ndl_np = ndl_model.k_projection.weight.grad.detach().numpy()
    v_proj_grad_ndl_np = ndl_model.v_projection.weight.grad.detach().numpy()
    np.testing.assert_allclose(q_proj_grad_pyt_np, q_proj_grad_ndl_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(k_proj_grad_pyt_np, k_proj_grad_ndl_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(v_proj_grad_pyt_np, v_proj_grad_ndl_np, atol=1e-4, rtol=1e-4)


parity_transform_post_norm_numpy()
parity_transform_pre_norm_numpy()
parity_transform_pre_norm_needle()