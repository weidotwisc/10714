Homework 0
# Section 1 Prep
```bash
git clone https://github.com/dlsyscourse/hw0.git
pip3 install --upgrade --no-deps git+https://github.com/dlsyscourse/mugrade.git
pip3 install pybind11
pip3 install numdifftools
```

# Section 2 Notes
See my overleaf ML self-study notes Section 8.1

# Section 3 Tests
## Section 3.1 unit tests
```bash
# assuming in the home directory .../10714/hw_code/hw0
python3 -m pytest -k "add"
python3 -m pytest -k "parse_mnist"
python3 -m pytest -k "softmax_loss"
python3 -m pytest -k "softmax_regression_epoch and not cpp"
python3 -m pytest -k "nn_epoch"
python3 -m pytest -k "softmax_regression_epoch_cpp"
```
## Section 3.2 full functionality tests

