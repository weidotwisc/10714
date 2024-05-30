# workding directory
```hw_code/hw3```

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
python3 -m pytest -v -k "setitem and cpu"
python3 -m pytest -v -k "(ewise_fn or ewise_max or log or exp or tanh or (scalar and not setitem)) and cpu"
python3 -m pytest -v -k "reduce and cpu"
python3 -m pytest -s -v -k "matmul and cpu"
python3 -m pytest -v -k "matmul and cpu or matmul_tiled" # see https://github.com/dlsyscourse/hw3/issues/7
python3 -m pytest -v -k "(compact or setitem) and cuda"
python3 -m pytest -v -k "(ewise_fn or ewise_max or log or exp or tanh or (scalar and not setitem)) and cuda"
python3 -m pytest -v -k "reduce and cuda"
python3 -m pytest -v -k "matmul and cuda"
```


# Section 3.2 Full Functionality Tests
```
source ../dlsys.profile
./test_all.sh
```

# Update: 2024-05-29
I have added [dlsys.profile](./hw_code/dlsys.profile) so that ```PYTHON_EXECUTABLE_PATH``` is defined for both ccc and f5 and then [CMakeLists.txt](./hw_code/hw3/CMakeLists.txt) can find the right python.

