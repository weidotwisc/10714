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
python3 -m pytest ./tests -l -v -k "nd_backend"
python3 -m pytest ./tests -l -v -k "test_cifar10"
python3 -m pytest ./tests -l -v -k "pad_forward"
python3 -m pytest ./tests -l -v -k "flip"
python3 -m pytest ./tests -l -v -k "dilate"
python3 -m pytest  ./tests -l -v -k "op_conv and forward"
python3 -m pytest ./tests -l -v -k "op_conv and backward"
python3 -m pytest ./tests  -l -v -k "kaiming_uniform"
python3 -m pytest ./tests  -l -v -k "nn_conv_forward"
python3 -m pytest ./tests  -l -v -k "nn_conv_backward" # weiz 2024-10-15
python3 -m pytest ./tests  -l -v -k "resnet9" # weiz 2024-10-21
python3 -m pytest ./tests -l -v -k "train_cifar10" # weiz 2024-10-30
```

# Step3 test all
```bash
test_all.sh
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

