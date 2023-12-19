import numpy as np
import sys
import numdifftools as nd
sys.path.append("./src")
sys.path.append("../src")
sys.path.append("../src/simple_ml_ext")
import mugrade

from simple_ml import *
try:
    from simple_ml_ext import *
except:
    pass


##############################################################################
### TESTS/SUBMISSION CODE FOR add()
def test_add():
    assert add(5,6) == 11
    assert add(3.2,1.0) == 4.2
    assert type(add(4., 4)) == float
    np.testing.assert_allclose(add(np.array([1,2]), np.array([3,4])),
                               np.array([4,6]))

def submit_add():
    mugrade.submit(add(1,2))
    mugrade.submit(add(4.5, 3.2))
    mugrade.submit(type(add(3,2)))
    mugrade.submit(add(np.array([1.,2.]), np.array([5,6])))


##############################################################################
### TESTS/SUBMISSION CODE FOR parse_mnist()

def test_parse_mnist():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (60000,784)
    assert y.shape == (60000,)
    np.testing.assert_allclose(np.linalg.norm(X[:10]), 27.892084)
    np.testing.assert_allclose(np.linalg.norm(X[:1000]), 293.0717,
        err_msg="""If you failed this test but not the previous one,
        you are probably normalizing incorrectly. You should normalize
        w.r.t. the whole dataset, _not_ individual images.""", rtol=1e-6)
    np.testing.assert_equal(y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])


def submit_parse_mnist():
    X,y = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                      "data/t10k-labels-idx1-ubyte.gz")
    mugrade.submit(X.dtype)
    mugrade.submit(y.dtype)
    mugrade.submit(X.shape)
    mugrade.submit(y.shape)
    mugrade.submit(np.linalg.norm(X[:10]))
    mugrade.submit(y[:10])


##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_loss()

def test_softmax_loss():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)

    Z = np.zeros((y.shape[0], 10))
    np.testing.assert_allclose(softmax_loss(Z,y), 2.3025850)
    Z = np.random.randn(y.shape[0], 10)
    np.testing.assert_allclose(softmax_loss(Z,y), 2.7291998)


def submit_softmax_loss():
    X,y = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                      "data/t10k-labels-idx1-ubyte.gz")
    np.random.seed(0)
    mugrade.submit(softmax_loss(np.zeros((y.shape[0], 10)),y))
    mugrade.submit(softmax_loss(np.random.randn(y.shape[0], 10),y))


##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_regression_epoch()

def test_softmax_regression_epoch():
    # test numeical gradient
    np.random.seed(0)
    X = np.random.randn(50,5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    Theta = np.zeros((5,3), dtype=np.float32)
    dTheta = -nd.Gradient(lambda Th : softmax_loss(X@Th.reshape(5,3),y))(Theta)
    softmax_regression_epoch(X,y,Theta,lr=1.0,batch=50)
    np.testing.assert_allclose(dTheta.reshape(5,3), Theta, rtol=1e-4, atol=1e-4)


    # test multi-steps on MNIST
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
    softmax_regression_epoch(X[:100], y[:100], theta, lr=0.1, batch=10)
    np.testing.assert_allclose(np.linalg.norm(theta), 1.0947356, 
                               rtol=1e-5, atol=1e-5)


def submit_softmax_regression_epoch():
    X,y = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                      "data/t10k-labels-idx1-ubyte.gz")

    theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
    softmax_regression_epoch(X[:100], y[:100], theta, lr=0.2, batch=100)
    mugrade.submit(np.linalg.norm(theta))

    theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
    softmax_regression_epoch(X, y, theta, lr=0.1, batch=200)
    mugrade.submit(np.linalg.norm(theta))
    mugrade.submit(loss_err(X@theta, y))

##############################################################################
### TESTS/SUBMISSION CODE FOR nn_epoch()

def test_nn_epoch():

    # test nn gradients
    np.random.seed(0)
    X = np.random.randn(50,5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
    W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
    dW1 = nd.Gradient(lambda W1_ : 
        softmax_loss(np.maximum(X@W1_.reshape(5,10),0)@W2, y))(W1)
    dW2 = nd.Gradient(lambda W2_ : 
        softmax_loss(np.maximum(X@W1,0)@W2_.reshape(10,3), y))(W2)
    W1_0, W2_0 = W1.copy(), W2.copy()
    nn_epoch(X, y, W1, W2, lr=1.0, batch=50)
    np.testing.assert_allclose(dW1.reshape(5,10), W1_0-W1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dW2.reshape(10,3), W2_0-W2, rtol=1e-4, atol=1e-4)

    # test full epoch
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)
    W1 = np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100)
    W2 = np.random.randn(100, 10).astype(np.float32) / np.sqrt(10)
    nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
    np.testing.assert_allclose(np.linalg.norm(W1), 28.437788, 
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(W2), 10.455095, 
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(loss_err(np.maximum(X@W1,0)@W2, y),
                               (0.19770025, 0.06006667), rtol=1e-4, atol=1e-4)


def submit_nn_epoch():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")

    np.random.seed(1)
    W1 = np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100)
    W2 = np.random.randn(100, 10).astype(np.float32) / np.sqrt(10)
    nn_epoch(X[:100], y[:100], W1, W2, lr=0.1, batch=100)
    mugrade.submit(np.linalg.norm(W1))
    mugrade.submit(np.linalg.norm(W2))

    np.random.seed(1)
    W1 = np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100)
    W2 = np.random.randn(100, 10).astype(np.float32) / np.sqrt(10)
    nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
    mugrade.submit(np.linalg.norm(W1))
    mugrade.submit(np.linalg.norm(W2))
    mugrade.submit(loss_err(np.maximum(X@W1,0)@W2, y))


##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_regression_epoch_cpp()

def test_matmul_cpp():
    X = np.arange(6).reshape(2,3)
    Y = np.arange(6).reshape(3,2)
    Z = X@Y
    result = np.zeros(Z.shape).astype(np.float32)
    matmul_cpp(X.astype(np.float32), Y.astype(np.float32), result)
    np.testing.assert_equal(Z, result)
    X = np.random.randn(2, 3)
    Y = np.random.randn(3, 2)
    Z = X @ Y
    result = np.zeros(Z.shape).astype(np.float32)
    matmul_cpp(X.astype(np.float32), Y.astype(np.float32), result)
    np.testing.assert_allclose(Z, result, rtol=1e-6, atol=1e-6)

def test_softmax_cpp():
    Z = np.random.randn(20,10)
    Z_exp = np.exp(Z)  # element-wise exp()
    Z_exp_sum_col = (np.sum(Z_exp, axis=1)).reshape(-1, 1)  # make a column vector, element component is the sum of each row in Z_exp
    A = Z_exp / Z_exp_sum_col  # bcast and normalize Z_exp to get Activation A
    result = np.zeros(Z.shape).astype(np.float32)
    softmax_cpp(Z.astype(np.float32), result)
    np.testing.assert_allclose(A, result, rtol=1e-6, atol=1e-6)

def test_cross_entropy_loss_cpp():
    m = 20 # batch size
    k = 10 # number of classes
    Z = np.random.randn(m, k)
    Z_exp = np.exp(Z)  # element-wise exp()
    Z_exp_sum_col = (np.sum(Z_exp, axis=1)).reshape(-1, 1)  # make a column vector, element component is the sum of each row in Z_exp
    A = Z_exp / Z_exp_sum_col  # bcast and normalize Z_exp to get Activation A
    y = np.random.randint(0, k, m)  # 0 to (k-1) is the range, m is the number of random numbers generated
    Y = np.eye(k)[y]
    avg_loss = np.sum(np.log(A) * Y)*(-1) / m
    l = cross_entropy_loss_cpp(A.astype(np.float32), y.astype(np.uint8))
    print("py: ", avg_loss)
    print("cpp: ", l)
    np.testing.assert_allclose(avg_loss, l, rtol=1e-6, atol=1e-6)

def test_X_transpose_matmul_Y_cpp():
    m = 20
    n = 30
    k =15
    X = np.random.randn(m, n)
    Y = np.random.randn(m, k)
    Z = X.transpose() @ Y
    result = np.zeros(Z.shape).astype(np.float32)
    X_transpose_matmul_Y_cpp(X.astype(np.float32), Y.astype(np.float32), result)
    np.testing.assert_allclose(Z, result, rtol=1e-6, atol=1e-6)


def test_softmax_regression_epoch_cpp():
    # test numeical gradient
    np.random.seed(0)
    X = np.random.randn(50,5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    Theta = np.zeros((5,3), dtype=np.float32)
    dTheta = -nd.Gradient(lambda Th : softmax_loss(X@Th.reshape(5,3),y))(Theta)
    softmax_regression_epoch_cpp(X,y,Theta,lr=1.0,batch=50)
    np.testing.assert_allclose(dTheta.reshape(5,3), Theta, rtol=1e-4, atol=1e-4)


    # test multi-steps on MNIST
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
    softmax_regression_epoch_cpp(X[:100], y[:100], theta, lr=0.1, batch=10)
    print("weiz: softmax_regression_epoch_cpp is called")
    np.testing.assert_allclose(np.linalg.norm(theta), 1.0947356, 
                               rtol=1e-5, atol=1e-5)


def submit_softmax_regression_epoch_cpp():
    X,y = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                      "data/t10k-labels-idx1-ubyte.gz")

    theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
    softmax_regression_epoch_cpp(X[:100], y[:100], theta, lr=0.2, batch=100)
    mugrade.submit(np.linalg.norm(theta))

    theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
    softmax_regression_epoch_cpp(X, y, theta, lr=0.1, batch=200)
    mugrade.submit(np.linalg.norm(theta))
    mugrade.submit(loss_err(X@theta, y))
