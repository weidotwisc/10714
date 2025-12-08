import sys

from needle import backend_ndarray
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

def test_transformer_language_model():
    device = backend_ndarray.ndarray.default_device()
    #device = ndl.cuda()
    corpus = ndl.data.Corpus("data/ptb")
    train_data = ndl.data.batchify(corpus.train, batch_size=256, device=device, dtype="float32")
    # step 1 define model
    embedding_dim = 64
    num_head = 8
    dim_head = 8 # notice in PyTorch TransformerDecoderLayer, one has to make sure embedding_dim = num_head x dim_head, in needle impl, we don't have such constraints
    vocab_size = len(corpus.dictionary)
    hidden_size = 32
    num_layers = 1
    seq_len = 20
    model = LanguageModel(embedding_size=embedding_dim, output_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, seq_model='transformer',
                           seq_len=seq_len, num_head=num_head, dim_head=dim_head, device=device)
    
    # step 2 define optimizer
    lr = 0.003
    n_epochs = 10
    train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device, lr=lr, optimizer=ndl.optim.Adam)
    evaluate_ptb(model, train_data, seq_len=seq_len, device=device)

test_transformer_language_model()
