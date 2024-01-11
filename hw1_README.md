# Section 1 Prep
git clone https://github.com/dlsys10714/hw1.git

# Section 2 Notes
#TODO

# Section 3 Tests

# Section 3.1 Unit Tests
```bash
# assume at /Users/weiz/courses/cmu/dl_sys/10714/hw_code/hw1
python3 -m pytest -v -k "forward"
python3 -m pytest -l -v -k "backward"
python3 -m pytest -k "topo_sort"
python3 -m pytest -k "compute_gradient"
python3 -m pytest -k "softmax_loss_ndl"
python3 -m pytest -l -k "nn_epoch_ndl"
```

# Section 3.2 Full Functionality Tests
```bash
# assume at /Users/weiz/courses/cmu/dl_sys/10714/hw_code/hw1
python apps/simple_ml.py
```

Results:
```
training loss err:  (0.009224053184869952, 0.0009833333333333332)
testing loss err:  (0.06452358790197966, 0.0196)
```
