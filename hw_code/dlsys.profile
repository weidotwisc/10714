#!/usr/bin/env bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export DLSYS_HOME=$DIR

# Get the hostname
HOSTNAME=$(hostname)

# Set PYTHON_EXECUTABLE_PATH based on hostname
if [ "$HOSTNAME" = "f5" ]; then
    export PYTHON_EXECUTABLE_PATH="/mnt/nfs/d3nvme0/userhomes/weiz/venvs/bleeding/bin/python"
elif [[ "$HOSTNAME" == ccc* ]]; then
    export PYTHON_EXECUTABLE_PATH="/dccstor/weiz/.pyenv/versions/3.11.0/envs/codenet/bin/python"
elif [[ "$HOSTNAME" == lsf* ]]; then
    export PYTHON_EXECUTABLE_PATH="/gpfs/users/weiz/.pyenv/versions/codenet/bin/python"
else
    echo "Hostname does not match any specified pattern."
    exit 1
fi
# Print the set value for verification
echo "PYTHON_EXECUTABLE_PATH set to: $PYTHON_EXECUTABLE_PATH"
