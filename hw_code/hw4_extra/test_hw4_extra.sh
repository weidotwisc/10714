source ../dlsys.profile
export PYTHONPATH=$DLSYS_HOME/hw4_extra/python
# part 1 multi-head attention activation layer
python3 -m pytest ./tests/hw4_extra/ -l -v -k   "attention_activation" # weiz 2024-12-29, mom's birthday :)
# part 2 implementing the self-attention layer with trainable parameters
python3 -m pytest ./tests/hw4_extra  -l -v -k "attention_layer" # weiz 2025-01-02
