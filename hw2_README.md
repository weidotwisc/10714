# Section 1 Prep
git clone https://github.com/dlsys10714/hw2.git

# Section 2 Notes
#TODO

# Section 3 Tests

# Section 3.1 Unit Tests
```bash
python3 -m pytest -v -k "test_init"
python3 -m pytest -v -k "test_nn_linear"
python3 -m pytest -v -k "test_nn_relu"
python3 -m pytest -v -k "test_nn_sequential"
python3 -m pytest -v -k "test_op_logsumexp" 
python3 -m pytest -v -k "test_nn_softmax_loss"
python3 -m pytest -v -k "test_nn_layernorm"
python3 -m pytest -v -k "test_nn_flatten"
python3 -m pytest -v -k "test_nn_batchnorm"
python3 -m pytest -v -k "test_nn_dropout"
python3 -m pytest -v -k "test_nn_residual"
python3 -m pytest -v -k "test_optim_sgd"
python3 -m pytest -v -k "test_optim_adam"
python3 -m pytest -v -k "flip_horizontal"
python3 -m pytest -v -k "random_crop"
python3 -m pytest -v -k "test_mnist_dataset"
python3 -m pytest -v -k "test_dataloader"
python3 -m pytest -v -k "test_mlp"
```


# Section 3.2 Full Functionality Tests
For the final mlp_resnet test, I get
```
Epoch10 train_err 0.014016666666666667, train_loss 0.04696077882467459
Epoch10 test_err 0.0265, test_loss 0.08526339300791733
```
