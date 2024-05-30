#!/usr/bin/env bash
# Check if DLSYS_HOME is defined
if [ -z "$DLSYS_HOME" ]; then
    echo "Error: DLSYS_HOME environment variable is not defined."
    exit 1
fi
export PYTHONPATH=$PYTHONPATH:$DLSYS_HOME/hw3/python
export NEEDLE_BACKEND=nd
python3 -m pytest -v -k "reshape"
python3 -m pytest -v -k "permute"
python3 -m pytest -v -k "test_broadcast_to" # for some reason if i only specify broadcast, it will also test compact(), which i haven't implemented yet
python3 -m pytest -v -k "getitem and cpu and not compact"
python3 -m pytest -v -k "compact and cpu"
python3 -m pytest -v -k "setitem and cpu"
python3 -m pytest -v -k "(ewise_fn or ewise_max or log or exp or tanh or (scalar and not setitem)) and cpu"
python3 -m pytest -v -k "reduce and cpu"
python3 -m pytest -s -v -k "matmul and cpu"
python3 -m pytest -v -k "matmul and cpu or matmul_tiled" # see https://github.com/dlsyscourse/hw3/issues/7
python3 -m pytest -v -k "(compact or setitem) and cuda"
python3 -m pytest -v -k "(ewise_fn or ewise_max or log or exp or tanh or (scalar and not setitem)) and cuda"
python3 -m pytest -v -k "reduce and cuda"
python3 -m pytest -v -k "matmul and cuda"