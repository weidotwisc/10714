# working directory
hw_code/hw4

# Step 1 Prep
```bash
pip3 install --upgrade --no-deps git+https://github.com/dlsyscourse/mugrade.git # might need
pip3 install numdifftools # might need
git clone https://github.com/dlsys10714/hw4.git
cd hw4
# get cifar10 data
mkdir data
cd data
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar vxzf cifar-10-python.tar.gz
cd ..
# end of get cifar10 data
source ../dlsys.profile
make # MUST DO so that all the backends are properly built
```

# Step 2 Unit test
```bash
source ../dlsys.profile
export NEEDLE_BACKEND=nd
export PYTHONPATH=/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python
python3 -m pytest ./tests/hw4 -l -v -k "nd_backend"
python3 -m pytest ./tests/hw4 -l -v -k "test_cifar10"
python3 -m pytest ./tests/hw4 -l -v -k "pad_forward"
python3 -m pytest ./tests/hw4 -l -v -k "flip"
python3 -m pytest ./tests/hw4 -l -v -k "dilate"
python3 -m pytest ./tests/hw4 -l -v -k "op_conv and forward" # weiz 2024-09-16
python3 -m pytest ./tests/hw4  -l -v -k "op_conv and backward" # weiz first all passed on 2024-09-29, because Zico's test cases cover little. Then reall passed on 2024-10-09, with proper dilate filter and more test cases by my own.
python3 -m pytest ./tests/hw4  -l -v -k "kaiming_uniform" # weiz 2024-10-09
python3 -m pytest ./tests/hw4  -l -v -k "nn_conv_forward" # weiz 2024-10-15
python3 -m pytest ./tests/hw4  -l -v -k "nn_conv_backward" # weiz 2024-10-15
python3 -m pytest ./tests/hw4  -l -v -k "resnet9" # weiz 2024-10-21
python3 -m pytest ./tests/hw4 -l -v -k "train_cifar10" # weiz 2024-10-30
python3 -m pytest ./tests/hw4 -v -k "test_rnn_cell" # weiz 2024-11-16
python3 -m pytest ./tests/hw4 -l -v -k "test_rnn" # weiz 2024-11-16
python3 -m pytest ./tests/hw4/ -l -v -k "test_lstm_cell" # weiz 2024-11-17
python3 -m pytest ./tests/hw4/ -l -v -k "test_lstm" # weiz 2024-11-18
python3 -m pytest ./tests/hw4 -l -v -k "ptb" # weiz 2024-11-20
python3 -m pytest ./tests/hw4 -l -v -k "language_model_implementation" # weiz 2024-11-28
```

# Step3 test all
## All unit tests
```bash
test_all.sh
```
## App test
```bash
python apps/simple_ml.py --app cifar10
```

# Step4 test previous homeworks
This is unique step that I try to use hw4 to test hw1, hw2 and hw3 tests.
## HW1
```bash
# HW1 (under $DLSYS_HOME/hw4 directory)
$ test_hw1.sh
$ export NEEDLE_BACKEND=nd_cuda
$ python apps/simple_ml.py --app hw1
training loss, training err:  (array([0.00915339], dtype=float32), 0.0010166666666666666)
testing loss, testing err:  (array([0.06506477], dtype=float32), 0.0192)
real    0m48.044s
user    0m41.041s
sys     0m13.264s
$ export NEEDLE_BACKEND=nd
$ python apps/simple_ml.py --app hw1
training loss, training err:  (array([0.0091638], dtype=float32), 0.0010166666666666666)
testing loss, testing err:  (array([0.06474191], dtype=float32), 0.0198)       
real    26m40.875s
user    26m38.333s
sys     0m6.141s
$ export NEEDLE_BACKEND=np
$ python apps/simple_ml.py --app hw1
training loss, training err:  (0.009235965007539178, 0.0009833333333333332)
testing loss, testing err:  (0.06463945428130624, 0.0196)
real    1m10.756s
user    31m31.516s
sys     43m2.866s
```
As we can see the numpy() matmul impl is very good close enough to my naive CUDA code. In the meantime the CPU backend nd is very slow.

## HW2
```bash
$ ./test_hw2.sh
export NEEDLE_BACKEND=nd
$ python apps/mlp_resnet.py
Epoch1 train_err 0.12138333333333333, train_loss 0.39724682342261075
Epoch1 test_err 0.0458, test_loss 0.1506293749110773
Epoch2 train_err 0.04781666666666667, train_loss 0.1562193478271365
Epoch2 test_err 0.0378, test_loss 0.11920421850401908
Epoch3 train_err 0.03225, train_loss 0.10838141564900676
Epoch3 test_err 0.0354, test_loss 0.10827904904843308
Epoch4 train_err 0.025516666666666667, train_loss 0.08492666484011958
Epoch4 test_err 0.0328, test_loss 0.10682388768880628
Epoch5 train_err 0.0212, train_loss 0.0705799465548868
Epoch5 test_err 0.0288, test_loss 0.09647917824448086
Epoch6 train_err 0.018266666666666667, train_loss 0.06110667303359757
Epoch6 test_err 0.0284, test_loss 0.09368669569084886
Epoch7 train_err 0.01675, train_loss 0.055382042803491155
Epoch7 test_err 0.0272, test_loss 0.09539421081542969
Epoch8 train_err 0.016566666666666667, train_loss 0.05268322633113712
Epoch8 test_err 0.0311, test_loss 0.10642983846657444
Epoch9 train_err 0.015516666666666666, train_loss 0.049933036343039326
Epoch9 test_err 0.0309, test_loss 0.09836121287720744
Epoch10 train_err 0.013316666666666666, train_loss 0.04581303940620273
Epoch10 test_err 0.0275, test_loss 0.0924831946811173

$ export NEEDLE_BACKEND=np
$ time python apps/mlp_resnet.py
Using numpy backend
/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/data
Epoch1 train_err 0.12176666666666666, train_loss 0.39788831365605193
Epoch1 test_err 0.0483, test_loss 0.15307226228062062
Epoch2 train_err 0.047966666666666664, train_loss 0.15646728590751688
Epoch2 test_err 0.0394, test_loss 0.1266390545316972
Epoch3 train_err 0.032483333333333336, train_loss 0.10807666268820564
Epoch3 test_err 0.033, test_loss 0.10674781305366196
Epoch4 train_err 0.0254, train_loss 0.08552930859072755
Epoch4 test_err 0.0327, test_loss 0.10924756327527575
Epoch5 train_err 0.02178333333333333, train_loss 0.07118703774176538
Epoch5 test_err 0.0318, test_loss 0.10522429438948165
Epoch6 train_err 0.019533333333333333, train_loss 0.06374976551936319
Epoch6 test_err 0.0287, test_loss 0.0970231155824149
Epoch7 train_err 0.01721666666666667, train_loss 0.05683380129824703
Epoch7 test_err 0.0285, test_loss 0.09395950592064764
Epoch8 train_err 0.016033333333333333, train_loss 0.052890151179550836
Epoch8 test_err 0.0271, test_loss 0.09135522606200538
Epoch9 train_err 0.015266666666666666, train_loss 0.04985853557319691
Epoch9 test_err 0.0255, test_loss 0.0900757042632904
Epoch10 train_err 0.014316666666666667, train_loss 0.04677191001130268
Epoch10 test_err 0.026, test_loss 0.08474011503567454
real    3m10.168s
user    53m46.711s
sys     115m29.270s

$ export NEEDLE_BACKEND=nd_cuda
$ time python apps/mlp_resnet.py

Epoch1 train_err 0.12113333333333333, train_loss 0.3974067923799157
Epoch1 test_err 0.0477, test_loss 0.15360124985221774
Epoch2 train_err 0.0473, train_loss 0.15497828715170422
Epoch2 test_err 0.0389, test_loss 0.11916258395067417
Epoch3 train_err 0.032966666666666665, train_loss 0.10892302753714224
Epoch3 test_err 0.0344, test_loss 0.10798864909331314
Epoch4 train_err 0.025766666666666667, train_loss 0.08488478493255873
Epoch4 test_err 0.0319, test_loss 0.10179146365087945
Traceback (most recent call last):
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/apps/mlp_resnet.py", line 156, in <module>                                                             
    train_mnist(data_dir=os.path.join(DLSYS_HOME, "hw4", "data"))
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/apps/mlp_resnet.py", line 111, in train_mnist                                                          
    avg_train_err, avg_train_loss = epoch(mnist_train_dataloader, model, opt)
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/apps/mlp_resnet.py", line 70, in epoch                                                                 
    loss.backward()
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/autograd.py", line 297, in backward                                                      
    compute_gradient_of_variables(self, out_grad)
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/autograd.py", line 399, in compute_gradient_of_variables                                 
    partial_adjoints = node.op.gradient_as_tuple(node.grad, node) # weiz 2023-12-30, note, we need to use node.op not input_node.op, this is most IMPORTANT!!!   
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/autograd.py", line 67, in gradient_as_tuple                                              
    output = self.gradient(out_grad, node)
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/ops/ops_mathematic.py", line 129, in gradient                                            
    grad_a = (b**(-1)) * out_grad
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/autograd.py", line 327, in __pow__                                                       
    return needle.ops.PowerScalar(other)(self)
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/autograd.py", line 80, in __call__                                                       
    return Tensor.make_from_op(self, args)
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/autograd.py", line 242, in make_from_op                                                  
    tensor.realize_cached_data()
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/autograd.py", line 107, in realize_cached_data                                           
    self.cached_data = self.op.compute(
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/ops/ops_mathematic.py", line 103, in compute                                             
    return a**self.scalar # weiz 2024 __pow__ is available in NDArray.py
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/backend_ndarray/ndarray.py", line 538, in __pow__                                        
    return self.ewise_or_scalar(other, self.device.ewise_power, self.device.scalar_power)                                                                        
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/backend_ndarray/ndarray.py", line 500, in ewise_or_scalar                                
    out = NDArray.make(self.shape, device=self.device)
  File "/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python/needle/backend_ndarray/ndarray.py", line 157, in make                                           
    array._handle = array.device.Array(prod(shape))
RuntimeError: out of memory
```

## HW3
```bash
./test_hw3.sh
```
