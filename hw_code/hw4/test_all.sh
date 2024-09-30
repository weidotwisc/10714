#!/usr/bin/env bash
source ../dlsys.profile
# Check if DLSYS_HOME is defined
if [ -z "$DLSYS_HOME" ]; then
    echo "Error: DLSYS_HOME environment variable is not defined."
    exit 1
fi
export PYTHONPATH=$PYTHONPATH:$DLSYS_HOME/hw4/python
export NEEDLE_BACKEND=nd
python3 -m pytest -l -v -k "nd_backend"
python3 -m pytest -l -v -k "test_cifar10"
python3 -m pytest -l -v -k "pad_forward"
python3 -m pytest -l -v -k "flip"
python3 -m pytest -l -v -k "dilate"
python3 -m pytest  -l -v -k "op_conv and forward" # weiz 2024-09-16
python3 -m pytest -l -v -k "op_conv and backward" # weiz 2024-09-29
