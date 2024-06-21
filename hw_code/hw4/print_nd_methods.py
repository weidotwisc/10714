import numpy as np
import inspect
#from needle import backend_ndarray as nd
import sys

sys.path.append("/mnt/nfs/d3nvme0/userhomes/weiz/10714/hw_code/hw4/python")
from needle.backend_ndarray import NDArray as nd
# Get the ndarray class
ndarray = nd

def print_ndarray_funcs():
    # List all attributes   
    all_attributes = dir(ndarray)

    # Filter out the functions and methods using inspect
    ndarray_functions = [
        func for func in all_attributes 
        if inspect.isfunction(getattr(ndarray, func)) or inspect.ismethod(getattr(ndarray, func))   
    ]

    # Print the list of functions and methods, including dunder methods
    print("Available functions in ndarray (including dunder methods):")
    for func in ndarray_functions:
        print(func)

print_ndarray_funcs()