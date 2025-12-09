This section deals with verification. We are going to use hw5 as the working directory

# Build 
## In VScode, on LSF01
### Create ```/gpfs/users/weiz/10714/hw_code/hw5/.vscode``` direcotry, and add these two files

(1) `vscode_bashrc` (the file that vscode need to call from `settings.json`, see below)
```bash
# Project-local rcfile used only by VS Code

# Optional: keep your usual shell behavior if you want
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi

# Always load fms env for this VS Code workspace
if [ -f "$HOME/envs/fms.profile" ]; then
  source "$HOME/envs/fms.profile"
fi

# Load dlsys.profile dynamically
if [ -f "$HOME/10714/hw_code/dlsys.profile" ]; then
    source "$HOME/10714/hw_code/dlsys.profile"
fi
```

(2) `settings.json` (called when vscode is started and will load enviroments above properly)
```json
{
    // code static analysis can index ./python as library
    "python.analysis.extraPaths": [
        "${workspaceFolder}/python"
    ],
    // terminal will get correct PYTHONPATH
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "$PYTHONPATH:${workspaceFolder}/python"
    },
    "terminal.integrated.profiles.linux": {
        "fmsbash": {
            "path": "/bin/bash",
            "args": [
                "--rcfile",
                "${workspaceFolder}/.vscode/vscode_bashrc"
            ]
        }
    },
    "python.envFile": "${workspaceFolder}/apps/.env",
    "terminal.integrated.defaultProfile.linux": "fmsbash"
}
```
### Create ```/gpfs/users/weiz/10714/hw_code/hw5/apps/.env``` file 
This file defines all enviroment variables relavent to the running environment of VSCode and contains following content *note I cannot use export here, as it is not a syntax accpeted in vscode*
```bash
PYTHONPATH=$PYTHONPATH:/gpfs/users/weiz/10714/hw_code/hw5/python
```
Note this file is referred in the `settings.json` and only sourced during each dynamic run, so you won't see anything in vscode terminal even if you do ```echo $PYTHONPATH```. Also things like ```BACKEND``` (```nd | nd_cuda| np ```) should be defined here as well


