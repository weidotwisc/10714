# working directory
hw_code/hw5 (Notice this at the beginning of carbon copy of hw4_extra

# Step 1 Prep
```bash # on lsf00
source ~/envs/fms.profile 
pyenv which python
pip3 install --upgrade --no-deps git+https://github.com/dlsyscourse/mugrade.git
pip3 install pybind11
pip3 install numdifftools
pip install cmake --upgrade 
pip install pybind11
pip install pytest
pip install argparse
pip install pytest
```
The exclude.txt can be found here [exclude.txt](./hw_code/exclude.txt)
# Step 2 Compile and make sure all the old tests can pass
## Step 2.1 unit tests
```bash
# assuming at 10714/hw_code
source ../dlsys.profile
export PYTHONPATH=$DLSYS_HOME/hw5/python
cd hw5
make
./test_hw1.sh # hw1
./test_hw2.sh # hw2
./test_hw3.sh # hw3
./test_hw4.sh # hw4
./test_hw4_Extra.sh # hw4_extra
```
## Step 2.2 hw4_extra transformer app test 
```bash
python apps/utils.py # this is to test AttentionLayer parity
python apps/explore_pyt_transformer_decoder.py
```
## Step 2.3 Run the Transformer based language model (PyTorch vs Needle)
```bash
python apps/pyt_lm_transformer.py # 10 epochs
python apps/ndl_lm_transformer.py # 10 epochs
```
The results look like this
```bash
python apps/ndl_lm_transformer.py
(1 epoch)
eval_loss = [6.07330316], eval_acc = 0.12936269854972376
(10 epoch)
eval_loss = [4.70171681], eval_acc = 0.22103655904696132 # (the exact same result as old python env)
python apps/pyt_lm_transformer.py
Eval Loss: 4.8383 Eval correctness:  0.1998 (10 epoch)
```
weiz 2025-03-10 ( i actually finished this on 2025-03-02 but forgot to update the README)
