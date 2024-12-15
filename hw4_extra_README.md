# working directory
hw_code/hw4_extra

# Step 1 Prep
```bash
cd hw_code
git@github.com:dlsyscourse/hw4_extra.git ./hw4_extra

# The purpose of following command is to copy all the files from hw4 to hw4_extra, overwrite them and keep the new ones in hw4_extra
# won't copy any file pattern listed in "exclude.txt".
# If we want to have a dry run, we do this: rsync -av --dry-run --exclude-from="exclude.txt" hw4/ hw4_extra/
rsync -av --exclude-from="exclude.txt" hw4/ hw4_extra/ 
```
# Step 2 Compile and make sure all the old tests can pass
```bash
source ../dlsys.home
export PYTHONPATH=$DLSYS_HOME/hw4_extra/python
make
./test_hw1.sh # hw1
./test_hw2.sh # hw2
./test_hw3.sh # hw3
./test_hw4.sh # hw4
```
