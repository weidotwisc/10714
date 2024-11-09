import sys

sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
import gc
np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    l1 = nn.Linear(in_features=dim, out_features=hidden_dim)
    norm1 = norm(hidden_dim) # don't forget hidden_dim in constructor param
    relu = nn.ReLU() # relu doesn't need any constructor param
    dropout = nn.Dropout(drop_prob) # don't forget drop_prob in constructor param
    l2 = nn.Linear(in_features=hidden_dim, out_features=dim)
    norm2 = norm(dim) # don't forget dim in consstructor param
    F_module = nn.Sequential(l1, norm1, relu, dropout, l2, norm2)
    residual = nn.Residual(F_module) # residual module takes F module as input
    relu_last = nn.ReLU()
    return nn.Sequential(residual, relu_last)
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    l1 = nn.Linear(in_features=dim, out_features=hidden_dim)
    relu = nn.ReLU()
    residual_blocks = [ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for i in range(num_blocks)]
    l_last = nn.Linear(in_features=hidden_dim, out_features=num_classes)
    return nn.Sequential(l1, relu, *residual_blocks, l_last)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if(opt is None):
        model.eval()
    else:
        model.train()
    loss_fn = nn.SoftmaxLoss()
    total_loss = 0.0
    total_err = 0
    for batch in dataloader:
        train_tensor, label_tensor = batch
        flatten_feature_size = np.prod(train_tensor.shape[1:])
        #print(flatten_feature_size)
        train_tensor = train_tensor.reshape((train_tensor.shape[0], flatten_feature_size)) # I need to flatten here to make the linear layer happy, thou
                                                 # previous dataset/dataloader ask for (28,28,1) shape, here we need to flatten it so that the linear layer
                                                 # will work with 784 as input dimension, instead of (28,28,1) tuple
        assert(train_tensor.requires_grad == False)
        logits = model(train_tensor)
        loss = loss_fn(logits, label_tensor)
        error = np.sum(np.argmax(logits.numpy(), axis=1) != label_tensor.numpy())
        total_err += error
        total_loss += loss.numpy().item() * train_tensor.shape[0] # softmax loss is average
        if(model.training):
            opt.reset_grad()
            loss.backward()
            opt.step()
        gc.collect()
    return total_err / len(dataloader.dataset), total_loss / len(dataloader.dataset)


    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # step 1 create dataset and dataloader
    print(data_dir)
    data_dir = os.path.realpath(data_dir)
    mnist_train_dataset = ndl.data.MNISTDataset(
        data_dir+"/train-images-idx3-ubyte.gz", data_dir+"/train-labels-idx1-ubyte.gz"
    )
    mnist_train_dataloader = ndl.data.DataLoader(
        dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True
    )
    mnist_test_dataset = ndl.data.MNISTDataset(
        data_dir+"/t10k-images-idx3-ubyte.gz", data_dir+"/t10k-labels-idx1-ubyte.gz"
    )
    mnist_test_dataloader = ndl.data.DataLoader(
        dataset=mnist_test_dataset, batch_size=batch_size, shuffle=False
    )
    # step 2 create model
    dim = np.prod(mnist_train_dataset[0][0].shape)
    model = MLPResNet(dim=dim, hidden_dim=hidden_dim)
    # step 3 create optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    # step 4 start training
    for i in range(epochs):
        avg_train_err, avg_train_loss = epoch(mnist_train_dataloader, model, opt)
        print(f"Epoch{i+1} train_err {avg_train_err}, train_loss {avg_train_loss}")
        avg_test_err, avg_test_loss = epoch(mnist_test_dataloader, model, None)
        print(f"Epoch{i + 1} test_err {avg_test_err}, test_loss {avg_test_loss}")
    return avg_train_err, avg_train_loss, avg_test_err, avg_test_loss # note even in hw2 description it asked for accuracy, the test cases actually compare against test error
    ### END YOUR SOLUTION


#### 2024-01-14, added by weiz
def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")
def linear_backward(lhs_shape, rhs_shape):
    np.random.seed(199)
    f = ndl.nn.Linear(*lhs_shape)
    f.bias.data = get_tensor(lhs_shape[-1])
    x = get_tensor(*rhs_shape)
    (f(x) ** 2).sum().backward()
    return x.grad.cached_data
def test_nn_linear_backward_1():
    np.testing.assert_allclose(
        linear_backward((10, 5), (1, 10)),
        np.array(
            [
                [
                    20.61148,
                    6.920893,
                    -1.625556,
                    -13.497676,
                    -6.672813,
                    18.762121,
                    7.286628,
                    8.18535,
                    2.741301,
                    5.723689,
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )

if __name__ == "__main__":
    DLSYS_HOME = os.getenv("DLSYS_HOME")
    train_mnist(data_dir=os.path.join(DLSYS_HOME, "hw4", "data"))
    #print("weiz hw2")
    #test_nn_linear_backward_1()
