# working directory
hw_code/hw4

# Step 1 Prep
```bash
git clone https://github.com/dlsys10714/hw4.git
```

# Step 2 Unit test
```bash
source ../dlsys.profile
export NEEDLE_BACKEND=nd
export PYTHONPATH=/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python
python3 -m pytest -l -v -k "nd_backend"
```

# Step3 test all
```bash
test_all.sh
```
