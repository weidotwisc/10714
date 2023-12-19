import os.path
import struct
import numpy as np
import gzip
import time
try:
    from simple_ml_ext import *
except:
    pass
import sys
sys.path.append("../")
print(sys.path)
def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x+y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    print(os.path.dirname(__file__))
    with gzip.open(label_filename, 'rb') as lbpath:
        magic, n = struct.unpack('>ii', lbpath.read(8)) # > means big-endian, i means int, two iis mean we need to read two numbers
        #print(magic, n)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8) # use np.frombuffer, apparently the previous lbapth.read(8) already moves the pointer to the proper data region
        assert(len(labels) == n)
        y = labels
        #print(np.max(labels), np.min(labels)) # labels from 0 to 9
    with gzip.open(image_filename, 'rb') as imgpath:
        magic, n, rows, cols = struct.unpack('>iiii', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
        assert(len(images) == n)
        #print(images.shape)
        X = images.astype(np.float32) / 255
    return X,y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    num_classes = np.max(y)+1
    Z_exp = np.exp(Z) # element-wise exp()
    Z_exp_sum_col = (np.sum(Z_exp, axis=1)).reshape(-1,1) # make a column vector, element component is the sum of each row in Z_exp
    A = Z_exp / Z_exp_sum_col # bcast and normalize Z_exp to get Activation A
    log_A = np.log(A) # recall cross entropy was <ground_truth, -log(prediction)>
    Y = np.eye(num_classes)[y] # create 1-hot-encoding, create an ID matrix of 10x10, and use each element in y to index this ID matrix to get the corresponding
                      # 1-hot encoding for that sample
    total_loss = np.sum(log_A * Y) *(-1) # element-wise multiply and then sum, same as sum(A@Y.transpose())
    return total_loss / len(Z)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    start_idx=0
    end_idx = start_idx + batch
    n_features = theta.shape[0]
    num_classes = theta.shape[1]
    while start_idx < len(X):
        # step 1: retrieve data
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        start_idx = start_idx + batch
        end_idx = end_idx + batch

        # step2: forward propagation
        Z = X_batch @ theta
        Z_exp = np.exp(Z)  # element-wise exp()
        Z_exp_sum_col = (np.sum(Z_exp, axis=1)).reshape(-1,1)  # make a column vector, element component is the sum of each row in Z_exp
        A = Z_exp / Z_exp_sum_col  # bcast and normalize Z_exp to get Activation A
        log_A = np.log(A)  # recall cross entropy was <ground_truth, -log(prediction)>
        Y = np.eye(num_classes)[y_batch]  # create 1-hot-encoding, create an ID matrix of 10x10, and use each element in y to index this ID matrix to get the corresponding
                                          # 1-hot encoding for that sample
        #total_loss = np.sum(log_A * Y) * (-1)  # element-wise multiply and then sum, same as sum(A@Y.transpose())
        #avg_loss = total_loss / len(Z)
        #print("loss: ", avg_loss)
        # step3: back propagation
        grad = X_batch.transpose() @ (A - Y.astype(np.float32)) / float(batch) # note the lecture in this class require the loss as the average loss, so the gradients here need to be averaged across batch size
        theta -= (lr*grad)



    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    start_idx = 0
    end_idx = start_idx + batch
    num_classes = W2.shape[1]
    while start_idx < len(X):
        # step 1: retrieve data
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        start_idx = start_idx + batch
        end_idx = end_idx + batch
        # step 2: fwd computation:
        Z1 = X_batch @ W1
        A1 = np.maximum(Z1, 0) # relu
        Z2 = A1 @ W2
        Z2_exp = np.exp(Z2)
        Z2_exp_sum_col = (np.sum(Z2_exp, axis=1)).reshape(-1,1)  # make a column vector, element component is the sum of each row in Z2_exp
        A2 = Z2_exp / Z2_exp_sum_col  # bcast and normalize Z_exp to get Activation A
        log_A2 = np.log(A2)  # recall cross entropy was <ground_truth, -log(prediction)>
        Y = np.eye(num_classes)[y_batch]  # create 1-hot-encoding, create an ID matrix of num_classes x num_classes, and use each element in y to index this ID matrix to get the corresponding
        # 1-hot encoding for that sample
        total_loss = np.sum(log_A2 * Y) * (-1)  # element-wise multiply and then sum, same as sum(A@Y.transpose())
        avg_loss = total_loss / len(Z2)
        #print("loss: ", avg_loss)
        # step 3: bwd computation:
        grad_z2 = A2 - Y
        grad_w2 = A1.transpose() @ grad_z2 / float(batch)
        # take element-wise derivative of Z1
        element_wise_z1_derivative = np.zeros(Z1.shape)
        element_wise_z1_derivative[Z1>0]=1
        grad_z1 = grad_z2 @ (W2.transpose()) * element_wise_z1_derivative
        grad_w1 = X_batch.transpose() @ grad_z1 / float(batch)
        W1 -= lr * grad_w1
        W2 -= lr * grad_w2


    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 400 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=400, epochs=20, lr = 0.2)


    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")



    start_time = time.time()
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.2, batch=100, cpp=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    # X = np.arange(6).reshape(2, 3)
    # Y = np.arange(6).reshape(3, 2)
    # Z = X @ Y
    # result = np.zeros(Z.shape).astype(np.float32)
    # matmul_cpp(X.astype(np.float32), Y.astype(np.float32), result)
    # print(Z == result)
    # print(result)
    # m=20
    # k=10
    # A = np.random.randn(m,k)
    # y = np.random.randint(0, k, m) # 0 to (k-1) is the range, m is the number of random numbers generated
    # l = cross_entropy_loss_cpp(A.astype(np.float32), y.astype(np.uint8))
    # print(l)
