#!/usr/bin/env bash
source ../dlsys.profile
# Check if DLSYS_HOME is defined
if [ -z "$DLSYS_HOME" ]; then
    echo "Error: DLSYS_HOME environment variable is not defined."
    exit 1
fi
export PYTHONPATH=$PYTHONPATH:$DLSYS_HOME/hw4/python
export NEEDLE_BACKEND=nd
python3 -m pytest ./tests/hw4 -l -v -k "nd_backend"
python3 -m pytest ./tests/hw4 -l -v -k "test_cifar10"
python3 -m pytest ./tests/hw4 -l -v -k "pad_forward"
python3 -m pytest ./tests/hw4 -l -v -k "flip"
python3 -m pytest ./tests/hw4 -l -v -k "dilate"
python3 -m pytest ./tests/hw4 -l -v -k "op_conv and forward" # weiz 2024-09-16
python3 -m pytest ./tests/hw4  -l -v -k "op_conv and backward" # weiz first all passed on 2024-09-29, because Zico's test cases cover little. Then reall passed on 2024-10-09, with proper dilate filter and more test cases by my own.
python3 -m pytest ./tests/hw4  -l -v -k "kaiming_uniform" # weiz 2024-10-09
python3 -m pytest ./tests/hw4  -l -v -k "nn_conv_forward" # weiz 2024-10-15
python3 -m pytest ./tests/hw4  -l -v -k "nn_conv_backward" # weiz 2024-10-15
python3 -m pytest ./tests/hw4  -l -v -k "resnet9" # weiz 2024-10-21
python3 -m pytest ./tests/hw4 -l -v -k "train_cifar10" # weiz 2024-10-30
python3 -m pytest ./tests/hw4 -v -k "test_rnn_cell" # weiz 2024-11-16
python3 -m pytest ./tests/hw4 -l -v -k "test_rnn" # weiz 2024-11-16
python3 -m pytest ./tests/hw4/ -l -v -k "test_lstm_cell" # weiz 2024-11-17
python3 -m pytest ./tests/hw4/ -l -v -k "test_lstm" # weiz 2024-11-18
python3 -m pytest ./tests/hw4 -l -v -k "ptb" # weiz 2024-11-20
