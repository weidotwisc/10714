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
```

# Step3 test all
```bash
test_all.sh
```
