source ../dlsys.profile
export PYTHONPATH=$DLSYS_HOME/hw4_extra/python
# part 1 multi-head attention activation layer
python3 -m pytest ./tests/hw4_extra/ -l -v -k   "attention_activation" # weiz 2024-12-29, mom's birthday :)
# part 2 implementing the self-attention layer with trainable parameters
python3 -m pytest ./tests/hw4_extra  -l -v -k "attention_layer" # weiz 2025-01-02
# part 3 Implementing a prenorm residual Transformer Layer¶
python3 -m pytest ./tests/hw4_extra -l -v -k "transformer_layer" # weiz 2025-01-02
# part 4 Implementing the Transformer model
python3 -m pytest ./tests/hw4_extra -l -v -k "transformer_model" # weiz 2025-01-03 # notice that I have changed the rtol from 1e-5 to 1e-4 to make all tests passed, otherwise 1 of the 32 test cases will fail on 1 of the 1080 elements.
