# Section 1 Prep
git clone https://github.com/dlsys10714/hw1.git

# Section 2 Notes
See See my overleaf ML self-study notes Section 8.2

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
$ time python apps/simple_ml.py
```

Results:
```
training loss, training err:  (0.009235965007539178, 0.0009833333333333332)    
testing loss, testing err:  (0.06463945428130624, 0.0196)                      
real    1m19.296s
user    34m7.397s
sys     49m36.290s
```
