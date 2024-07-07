source ../dlsys.profile
export NEEDLE_BACKEND=nd
export PYTHONPATH=/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python
python3 -m pytest -l -v -k "nd_backend"
python3 -m pytest -l -v -k "test_cifar10"
