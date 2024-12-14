# working directory
hw_code/hw4_extra

# Step 1 Prep
```bash
cd hw_code
git@github.com:dlsyscourse/hw4_extra.git ./hw4_extra
rsync -av --exclude-from="exclude.txt" hw4/ hw4_extra/ # if we want to have a dry run, we do this: rsync -av --dry-run --exclude-from="exclude.txt" hw4/ hw4_extra/
```
# Step 2 Compile and make sure all the old tests can pass
```bash
./test_all.sh # hw4
```