# Section 1 Prep
```bash
git clone https://github.com/dlsys10714/hw3.git
pip install cmake --upgrade # too old cmake will not work for cuda
```
In CMakeLists.txt add the following the specify the right python to use
```bash
set(Python_EXECUTABLE /mnt/nfs/d3nvme0/userhomes/weiz/venvs/bleeding/bin/python) # weiz 2024-03-18 specify the python that I want
```

# Section 2 Notes
#TODO

# Section 3 Tests
```bash
export PYTHONPATH=$PYTHONPATH:/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw3/python
export NEEDLE_BACKEND=nd
```

# Section 3.1 Unit Tests
```bash
python3 -m pytest -v -k "reshape"
```


# Section 3.2 Full Functionality Tests

