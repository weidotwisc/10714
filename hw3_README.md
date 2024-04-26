# Section 1 Prep
```bash
git clone https://github.com/dlsys10714/hw3.git
pip install cmake --upgrade # too old cmake will not work for cuda
```
In CMakeLists.txt add the following the specify the right python to use
```bash
set(Python_EXECUTABLE /mnt/nfs/d3nvme0/userhomes/weiz/venvs/bleeding/bin/python) # weiz 2024-03-18 specify the python that I want
```
Then do 
```bash
cmake #(?? to test again on CCC)
make
```

# Section 2 Project directory 
Note this homework must be built on dyce machine (or GPU machines)!
On dyce, the directory is at 

```/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw3```

# Section 3 Tests
```bash
export PYTHONPATH=$PYTHONPATH:/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw3/python
export NEEDLE_BACKEND=nd
```

# Section 3.1 Unit Tests
```bash
python3 -m pytest -v -k "reshape"
python3 -m pytest -v -k "permute"
python3 -m pytest -v -k "test_broadcast_to" # for some reason if i only specify broadcast, it will also test compact(), which i haven't implemented yet
python3 -m pytest -v -k "getitem and cpu and not compact"
python3 -m pytest -v -k "compact and cpu"
```


# Section 3.2 Full Functionality Tests

