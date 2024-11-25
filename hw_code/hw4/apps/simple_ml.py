"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
import sys
import os

dlsys_home=os.getenv('DLSYS_HOME')
assert(dlsys_home is not None)
sys.path.append(os.path.join(dlsys_home, "hw4"))
sys.path.append(os.path.join(dlsys_home, "hw4/python"))


import needle as ndl

import needle.nn as nn
from apps.models import *
import time
import argparse
from needle.backend_selection import default_device
from needle.data.datasets import *
#device = ndl.cpu() # weiz 2024-11-01 comment this out, as from hw4 i can use NEEDLE_BACKEND=nd, nd_cuda or np to control the backend
device = default_device() # weiz 2024-11-09 get default device

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(label_filename, 'rb') as lbpath:
        magic, n = struct.unpack('>ii', lbpath.read(8)) # > means big-endian, i means int, two iis mean we need to read two numbers
        #print(magic, n)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8) # use np.frombuffer, apparently the previous lbapth.read(8) already moves the pointer to the proper data region
        assert(len(labels) == n)
        y = labels
        #print(np.max(labels), np.min(labels)) # labels from 0 to 9
    with gzip.open(image_filesname, 'rb') as imgpath:
        magic, n, rows, cols = struct.unpack('>iiii', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
        assert(len(images) == n)
        #print(images.shape)
        X = images.astype(np.float32) / 255
    return X,y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    Z_exp = ndl.exp(Z)  # element-wise exp()
    Z_exp_sum_col = (ndl.summation(Z_exp, axes=1)).reshape((-1,1))  # make a column vector, element component is the sum of each row in Z_exp
                                                                       # note that, I have to use axes=(1,) not axes=(1), as (1) is not a tuple
    A = Z_exp / ndl.broadcast_to(Z_exp_sum_col, Z_exp.shape)  # bcast (explicitly in needl) and normalize Z_exp to get Activation A
    log_A = ndl.log(A)  # recall cross entropy was <ground_truth, -log(prediction)>
    total_loss = ndl.summation(log_A * y_one_hot) * (-1)  # element-wise multiply and then sum, same as sum(A@Y.transpose())
    return total_loss / (Z).shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    start_idx = 0
    end_idx = start_idx + batch
    num_classes = W2.shape[1]
    while start_idx < len(X):
        # step 1: retrieve data
        X_batch = ndl.Tensor(X[start_idx:end_idx])
        y_batch = y[start_idx:end_idx]
        start_idx = start_idx + batch
        end_idx = end_idx + batch
        # step 2: fwd computation:
        Z1 = ndl.matmul(X_batch, W1)
        A1 = ndl.relu(Z1)  # relu
        Z2 = ndl.matmul(A1, W2)
        Y = np.eye(num_classes)[y_batch]  # create 1-hot-encoding, create an ID matrix of num_classes x num_classes, and use each element in y to index this ID matrix to get the corresponding
        y_one_hot = ndl.Tensor(Y) # weiz, use numpy to generate really data logic and use ndl to wrap this tensor
        loss = softmax_loss(Z2, y_one_hot)
        print("loss: ", loss)
        # step 3: bwd computation:
        loss.backward()
        #W1.data = W1.data - lr * W1.grad.data  # in-place update, not working , because datatype in data() not matching
        #W2.data = W2.data - lr * W2.grad.data
        W1.cached_data = W1.cached_data - lr * W1.grad.realize_cached_data() # in-place update
        W2.cached_data = W2.cached_data - lr * W2.grad.realize_cached_data()
        #W1 = W1 - lr * W1.grad # this will add additional compute nodes in the graph
        #W2 = W2 - lr * W2.grad
    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if (opt is not None):
        model.train()
        correct, total_loss = 0, 0
        total_sample_num = 0
        for batch in dataloader:
            opt.reset_grad()
            X, y = batch
            X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn(out, y)
            print("training loss ", loss)
            total_loss += loss.data.numpy() * y.shape[0]
            total_sample_num += y.shape[0]
            loss.backward()
            opt.step()
        return correct/(total_sample_num), total_loss/(total_sample_num)
    else:
        model.eval()
        correct, total_loss = 0, 0
        total_sample_num = 0
        for batch in dataloader:
            X, y = batch
            X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn(out, y)
            print("eval loss ", loss)
            total_loss += loss.data.numpy() * y.shape[0]
            total_sample_num += y.shape[0]
        return correct/(total_sample_num), total_loss/(total_sample_num)

    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    #total_acc, total_loss = 0.0, 0.0
    for i in range(n_epochs):
        acc, loss = epoch_general_cifar10(dataloader=dataloader, model=model, loss_fn = loss_fn(), opt = opt)
        #total_acc += acc
        #total_loss += loss
        print(f"Epoch {i+1}: train_loss ={loss}, train_acc = {acc}")
    return acc, loss # weiz 2024-11-23 realized the question asks for only the last epoch acc and loss
    #return total_acc / n_epochs, total_loss / n_epochs
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(dataloader=dataloader, model=model, loss_fn=loss_fn())
    print(f"eval_loss = {avg_loss}, eval_acc = {avg_acc}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def cifar10_resnet9():

    dataset = ndl.data.CIFAR10Dataset(os.path.join(dlsys_home, "hw4", "data/cifar-10-batches-py"), train=True)
    dataloader = ndl.data.DataLoader(dataset=dataset,batch_size=128,shuffle=True)
    model = ResNet9(device=device, dtype="float32")
    train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,lr=0.001, weight_decay=0.001)
    evaluate_cifar10(model, dataloader)


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    ds = PTBDataset(batchified_data=data, seq_len=seq_len, device=device, dtype=dtype)
    h = None
    if (opt is not None):
        #dataloader = ndl.data.DataLoader(dataset=ds, batch_size=seq_len, shuffle=False)
        model.train()
        correct, total_loss = 0, 0
        total_sample_num = 0
        for batch in ds:
            opt.reset_grad()
            X, y = batch
            #X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out, h = model(X,h)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn(out, y)
            print("training loss ", loss)
            total_loss += loss.data.numpy() * y.shape[0]
            total_sample_num += y.shape[0]
            loss.backward()
            opt.step()
        return correct/(total_sample_num), total_loss/(total_sample_num)
    else:
        #dataloader = ndl.data.DataLoader(dataset=ds, batch_size=seq_len, shuffle=False)
        model.eval()
        correct, total_loss = 0, 0
        total_sample_num = 0
        for batch in ds:
            X, y = batch
            #X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out, h = model(X, h)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn(out, y)
            print("eval loss ", loss)
            total_loss += loss.data.numpy() * y.shape[0]
            total_sample_num += y.shape[0]
        return correct/(total_sample_num), total_loss/(total_sample_num)
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    #total_acc, total_loss = 0.0, 0.0
    for i in range(n_epochs):
        acc, loss = epoch_general_ptb(data=data, model=model, seq_len=seq_len, loss_fn=loss_fn(), opt=opt, clip=clip, device=device, dtype=dtype)
        #total_acc += acc
        #total_loss += loss
        print(f"Epoch {i+1}: train_loss ={loss}, train_acc = {acc}")
    return acc, loss # weiz 2024-11-23 realized the question asks for only the last epoch acc and loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data=data, model=model, seq_len=seq_len, loss_fn=loss_fn(), opt=None, clip=None, device=device, dtype=dtype)
    print(f"eval_loss = {avg_loss}, eval_acc = {avg_acc}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT




############# Below is added on 2024-11-01 to launch real training run ###############
def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy().squeeze(), np.mean(h.numpy().argmax(axis=1) != y) # weiz 2024-11-02 add the squeeze() to make sure we get a scalar value (e.g., numpy array of shape () instead of (1,) to make hw1 tests happy)


def weiz_nn_mnist():
    X, y = parse_mnist(
        os.path.join(dlsys_home, "hw4", "data/train-images-idx3-ubyte.gz"), 
        os.path.join(dlsys_home, "hw4", "data/train-labels-idx1-ubyte.gz")
    )
    X_te, y_te = parse_mnist(os.path.join(dlsys_home, "hw4","data/t10k-images-idx3-ubyte.gz"),
                             os.path.join(dlsys_home, "hw4", "data/t10k-labels-idx1-ubyte.gz"))
    np.random.seed(0)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 400).astype(np.float32) / np.sqrt(400))
    W2 = ndl.Tensor(np.random.randn(400, 10).astype(np.float32) / np.sqrt(10))
    for i in range(20):
        W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)

    print("training loss, training err: ", loss_err(ndl.relu(ndl.Tensor(X) @ W1) @ W2, y))
    print("testing loss, testing err: ", loss_err(ndl.relu(ndl.Tensor(X_te) @ W1) @ W2, y_te))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greet the user.")
    parser.add_argument("--app", type=str, help="Which app to run", default="hw1")
    args = parser.parse_args()
    if args.app == "hw1":
        weiz_nn_mnist() 
    elif args.app == 'cifar10':
        cifar10_resnet9()
    #weiz_explore_gradient_of_gradient()
    #weiz_explore_hessian()
    #weiz_test2()