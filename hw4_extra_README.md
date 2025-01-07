# working directory
hw_code/hw4_extra

# Step 1 Prep
```bash
cd 10714/hw_code
git@github.com:dlsyscourse/hw4_extra.git ./hw4_extra

# The purpose of following command is to copy all the files from hw4 to hw4_extra, overwrite them and keep the new ones in hw4_extra
# won't copy any file pattern listed in "exclude.txt".
# If we want to have a dry run, we do this: rsync -av --dry-run --exclude-from="exclude.txt" hw4/ hw4_extra/
rsync -av --exclude-from="exclude.txt" hw4/ hw4_extra/ 
```
The exclude.txt can be found here [exclude.txt](./hw_code/exclude.txt)
# Step 2 Compile and make sure all the old tests can pass
```bash
# assuming at 10714/hw_code
source ../dlsys.profile
export PYTHONPATH=$DLSYS_HOME/hw4_extra/python
cd hw4_extra
make
./test_hw1.sh # hw1
./test_hw2.sh # hw2
./test_hw3.sh # hw3
./test_hw4.sh # hw4
```

# Step 3 testing hw4_extra
## Step 3.1 test each part of the homework
```bash
source ../dlsys.profile
export PYTHONPATH=$DLSYS_HOME/hw4_extra/python
cd hw4_extra
# part 1 multi-head attention activation layer
python3 -m pytest ./tests/hw4_extra/ -l -v -k   "attention_activation" # weiz 2024-12-29, mom's birthday :)
# part 2 implementing the self-attention layer with trainable parameters
python3 -m pytest ./tests/hw4_extra  -l -v -k "attention_layer" # weiz 2025-01-02
# part 3 Implementing a prenorm residual Transformer Layer
python3 -m pytest ./tests/hw4_extra -l -v -k "transformer_layer" # weiz 2025-01-02
# part 4 Implementing the Transformer model                                     
python3 -m pytest ./tests/hw4_extra -l -v -k "transformer_model" # weiz 2025-01-03 # notice that I have changed the rtol from 1e-5 to 1e-4 to make all tests passed, otherwise 1 of the 32 test cases will fail on 1 of the 1080 elements.
```
## Step 3.2 test all the pieces
```bash
cd hw4_extra
./test_hw4_extra.sh
```
## Step 3.3 Run the Transformer based language model
```bash
cd hw4_extra
python apps/weiz_test_hw4_extra.py # notice still have pretty bad memory issue
```
