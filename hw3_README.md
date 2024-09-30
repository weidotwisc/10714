# workding directory
```hw_code/hw3```

# Section 1 Prep 
```bash
git clone https://github.com/dlsys10714/hw3.git
pip install cmake --upgrade # too old cmake will not work for cuda
pip install pybind11
pip install pytest
```
Notice that I have edited [dlsys.profile](./hw_code/dlsys.profile) so that different cluster (dyce, CCC, Vela-LSF) can all find the right ```PYTHON_EXECUTABLE_PATH``` path.

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
Note that in ```test_all.sh```, I have added the logic to set up the right ```PYTHONPATH``` and ```NEEDLE_BACKEND```.

# Update: 2024-05-29
I have added [dlsys.profile](./hw_code/dlsys.profile) so that ```PYTHON_EXECUTABLE_PATH``` is defined for both ccc and f5 and then [CMakeLists.txt](./hw_code/hw3/CMakeLists.txt) can find the right python.
# Update: 2024-09-30
I have added support for Vela LSF cluster as well in [dlsys.profile](./hw_code/dlsys.profile)

