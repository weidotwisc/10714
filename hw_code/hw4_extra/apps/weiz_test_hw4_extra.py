import sys
sys.path.append('./python')
sys.path.append('./apps')
import numpy as np
import pytest
import torch
import itertools
import mugrade
import os

import needle as ndl
import needle.nn as nn

from simple_ml import *
from models import LanguageModel


def test_attention_layer(batch_size, seq_len, input_dim, num_head, dim_head, causal, dropout, device):

    np.random.seed(19943)

    q = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)
    k = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)
    v = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)

    layer = nn.AttentionLayer(
        input_dim, num_head, dim_head, 
        dropout=dropout, causal=causal, device=device)

    result = layer(
        ndl.Tensor(q, device=device),
        ndl.Tensor(k, device=device),
        ndl.Tensor(v, device=device),
    )

    result = result.numpy()
        
    current_input_id = "-".join([str(x) for x in (
        batch_size, seq_len, input_dim, num_head, dim_head, causal, dropout, device
    )])

    labels_path = (
        "./tests/hw4_extra/data/" + 
        "test_attention_layer-{}.npy"
        .format(current_input_id))

    with open(labels_path, 'rb') as f:
        label_result = np.load(f)

    np.testing.assert_allclose(result, label_result, atol=1e-5, rtol=1e-5)


# @pytest.mark.parametrize("batch_size", [4, 8])
# @pytest.mark.parametrize("seq_len", [5, 11])
# @pytest.mark.parametrize("input_dim", [27])
# @pytest.mark.parametrize("num_head", [8])
# @pytest.mark.parametrize("dim_head", [32])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("dropout", [0.0, 0.1])
# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])




#test_attention_layer(batch_size=4, seq_len=5, input_dim=27, num_head=8, dim_head=32, causal=False, dropout=0.0, device=ndl.cpu())



def test_transformer_layer(batch_size, seq_len, input_dim, num_head, dim_head, hidden_size, causal, dropout, device):
    
    np.random.seed(19943)

    x = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)
    ndl_x = ndl.Tensor(x, device=device)

    layer = nn.TransformerLayer(
        input_dim, num_head, dim_head, hidden_size,
        dropout=dropout, causal=causal, device=device)

    result = layer(
        ndl_x
    )

    result = result.numpy()
        
    current_input_id = "-".join([str(x) for x in (
        batch_size, seq_len, input_dim, num_head, dim_head, hidden_size, causal, dropout, device
    )])

    labels_path = (
        "./tests/hw4_extra/data/" + 
        "test_transformer_layer-{}.npy"
        .format(current_input_id))

    with open(labels_path, 'rb') as f:
        label_result = np.load(f)

    np.testing.assert_allclose(result, label_result, atol=1e-5, rtol=1e-5)

# @pytest.mark.parametrize("batch_size", [2, 4])
# @pytest.mark.parametrize("seq_len", [5, 11])
# @pytest.mark.parametrize("input_dim", [27])
# @pytest.mark.parametrize("num_head", [8])
# @pytest.mark.parametrize("dim_head", [32])
# @pytest.mark.parametrize("hidden_size", [64])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("dropout", [0.0, 0.1])
# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])


#test_transformer_layer(batch_size=2, seq_len=5, input_dim=27, num_head=8, dim_head=32, hidden_size=64, causal=False, dropout=0.0, device=ndl.cpu())




def test_transformer_model(
        batch_size, seq_len, input_dim,
        hidden_size, num_layers,
        num_head, dim_head,
        causal, dropout, device):
        
    np.random.seed(19943)

    x = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)
    ndl_x = ndl.Tensor(x, device=device)

    model = nn.Transformer(
        input_dim, hidden_size, num_layers,
        num_head=num_head,
        dim_head=dim_head,
        dropout=dropout,
        causal=causal,
        device=device,
        batch_first=True,
    )

    result, _ = model(ndl_x)

    result = result.numpy()
        
    current_input_id = "-".join([str(x) for x in (
        batch_size, seq_len, input_dim,
        hidden_size, num_layers,
        num_head, dim_head,
        causal, dropout, device
    )])

    labels_path = (
        "./tests/hw4_extra/data/" + 
        "test_transformer_model-{}.npy"
        .format(current_input_id))

    with open(labels_path, 'rb') as f:
        label_result = np.load(f)

    np.testing.assert_allclose(result, label_result, atol=1e-5, rtol=1e-5)


# @pytest.mark.parametrize("batch_size", [8])
# @pytest.mark.parametrize("seq_len", [5, 11])
# @pytest.mark.parametrize("input_dim", [27])
# @pytest.mark.parametrize("hidden_size", [64])
# @pytest.mark.parametrize("num_layers", [2, 4])
# @pytest.mark.parametrize("num_head", [8])
# @pytest.mark.parametrize("dim_head", [32])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("dropout", [0.0, 0.1])
# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])

test_transformer_model(
        batch_size=8, seq_len=5, input_dim=27,
        hidden_size=64, num_layers=2,
        num_head=8, dim_head=32,
        causal=False, dropout=0.0, device=ndl.cpu())


test_transformer_model(
        batch_size=8, seq_len=5, input_dim=27,
        hidden_size=64, num_layers=4,
        num_head=8, dim_head=32,
        causal=False, dropout=0.1, device=ndl.cpu())