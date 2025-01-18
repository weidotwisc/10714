import torch
import numpy as np
import needle as ndl
from utils import *
import os
dlsys_home=os.getenv('DLSYS_HOME')
assert(dlsys_home is not None)
sys.path.append(os.path.join(dlsys_home, "hw4_extra"))
sys.path.append(os.path.join(dlsys_home, "hw4_extra/python"))

class PyTTransformerLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, seq_len, num_heads, dropout, device=None):
        super(PyTTransformerLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, device=device)
        self.pos_embedding = torch.nn.Embedding(seq_len, embedding_dim, device=device)
        self.M = torch.triu(-float("inf")*torch.ones(seq_len,seq_len),1).to(device)
        self.device = device
        transformer_layers = []
        for i in range(num_layers):
            transformer_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, 
                                                                 nhead=num_heads, 
                                                                 dim_feedforward=hidden_size, 
                                                                 dropout=dropout, 
                                                                 batch_first=True,
                                                                norm_first=True, device=device)
            transformer_layers.append(transformer_layer)
        self.transformer_layers = torch.nn.Sequential(*transformer_layers)
        #self.trans
        self.fc = torch.nn.Linear(embedding_dim, vocab_size, device=device)

    def forward(self, x):
        bs, seq_len = x.shape
        assert(seq_len == self.seq_len)
        x_emb = self.embedding(x)
        x_pos = self.pos_embedding(torch.arange(self.seq_len, device=self.device).unsqueeze(0)) # weiz x_pos is important, here x_pos is of (1,seq_len, embed_dim) shape
        #assert(x_emb.shape == x_pos.shape)
        #bs,seq_len, embedding_dim = x_pos.shape
        #assert(embedding_dim == self.embedding_dim)
        x_real = x_emb + x_pos # weiz x_pos is bcasted
        x_real = self.transformer_layers(x_real)
        out = self.fc(x_real) # TODO: need to flatten!
        return out


def test_pyt_language_model_training():
    set_pyt_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu") 
    embedding_dim = 64
    seq_len = 20
    batch_size = 256
    hidden_size = 32
    num_layers = 1
    num_heads=8
    n_epochs=10
    corpus = ndl.data.Corpus(os.path.join(dlsys_home, "hw4", "data/ptb"))
    vocab_size = len(corpus.dictionary)
    train_data = ndl.data.batchify(corpus.train, batch_size=batch_size, device=None, dtype="float32")


    # taking the same hyper-params from hw4
    #model = LanguageModel(20, len(corpus.dictionary), hidden_size=32, num_layers=1, seq_model='transformer', seq_len=20, device=device)
    

    # weiz get from hw4
    
    
    
    #def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, seq_len, num_heads, dropout, device=torch.cuda()):
    model = PyTTransformerLanguageModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, 
                                        num_layers=num_layers,
                                        seq_len=seq_len,
                                        num_heads=num_heads,
                                        dropout=0.0, 
                                        device=device)
    
    lr=0.003
    weight_decay = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    

    ds = ndl.data.PTBDataset(train_data, seq_len=seq_len, dtype="float32", device=None)
    
   

    for epoch in range(n_epochs):
        total_loss = 0
        total_samples = 0
        for sequences, targets in ds:
            sequences = np.transpose(sequences, (-1,-2)) # weiz 2025-01-17, sequences was in seq_len x bs integers, now make it bs x seq_len, notice they should be only integers now!!! as model embedding will make the extra embed_dim!!!
            sequences = torch.Tensor(sequences.numpy()).to(torch.long).to(device=device)
            targets = torch.Tensor(targets.numpy()).to(torch.long).to(device=device)
            optimizer.zero_grad()
        
            # Detach the hidden state to avoid backpropagating through the entire sequence history
            outputs = model(sequences)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            print(loss.item())
            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

        print(f"Training Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / total_samples:.4f}")
    total_loss = 0
    total_samples = 0
    for sequences, targets in ds:
        sequences = np.transpose(sequences, (-1,-2)) # weiz 2025-01-18, sequences was in seq_len x bs integers, now make it bs x seq_len, notice they should be only integers now!!! as model embedding will make the extra embed_dim!!!
        sequences = torch.Tensor(sequences.numpy()).to(torch.long).to(device=device)
        targets = torch.Tensor(targets.numpy()).to(torch.long).to(device=device)
        outputs = model(sequences)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        total_loss += loss.item() * len(targets)
        total_samples += len(targets)
    print(f"Eval Loss: {total_loss / total_samples:.4f}")
    
   
test_pyt_language_model_training()



