import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        layer1 = nn.ConvBN(3,16,7,stride=4, device=device, dtype=dtype)
        print("layer 1 #params, ", layer1.num_params())
        layer2 = nn.ConvBN(16,32,3,stride=2, device=device, dtype=dtype)
        print("layer 2 #params, ", layer2.num_params())
        layer34 = nn.ResNetBasicBlock(((32,32,3,1),(32,32,3,1)),device=device, dtype=dtype)
        print("layer 34 #params, ", layer34.num_params())
        layer5 = nn.ConvBN(32,64,3,2, device=device, dtype=dtype) 
        print("layer 5 #params, ", layer5.num_params())
        layer6 = nn.ConvBN(64,128,3,2, device=device, dtype=dtype) # weiz 2024-10-20 hw4 figure has an error, should be 64 instead of 32
        print("layer 6 #params, ", layer6.num_params())
        layer78 = nn.ResNetBasicBlock(((128,128,3,1), (128,128,3,1)), device=device, dtype=dtype)
        print("layer 78 #params, ", layer78.num_params())
        layer9 = nn.Sequential(nn.Flatten(), nn.Linear(128,128, device=device, dtype=dtype),nn.ReLU(), nn.Linear(128,10, device=device, dtype=dtype))
        print("layer 9 #params, ", layer9.num_params())
        self.model = nn.Sequential(layer1,
                                   layer2,
                                   layer34,
                                   layer5,
                                   layer6,
                                   layer78,
                                   layer9
                                )
        print("ResNet9 #params, ", self.num_params())
        
    
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)