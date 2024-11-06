"""Logic for backend selection"""
import os


BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")


if BACKEND == "nd" or BACKEND == "nd_cuda": # weiz 2024-10-26 add nd_cuda backend
    print("Using needle backend")
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        #default_device,
        BackendDevice as Device,
    )
    from .backend_ndarray import default_device as nd_default_device
    NDArray = array_api.NDArray
elif BACKEND == "np":
    print("Using numpy backend")
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, Device # ,default_device
    from .backend_numpy import default_device as np_default_device
    NDArray = array_api.ndarray
else:
    raise RuntimeError("Unknown needle array backend %s" % BACKEND)


# weiz 2024-11-05 adding the correct device triage
def default_device():
    if BACKEND == "np":
        return np_default_device()
    else:
        return nd_default_device()