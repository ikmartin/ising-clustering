from ctypes import c_int, c_bool, POINTER, c_double, CDLL, c_void_p
import numpy as np
from numpy.ctypeslib import ndpointer, as_array
from sys import path
import os
from pathlib import Path

# Statically load the interface so that the solver can be called.

#path.insert(1, '/opt/OpenBLAS')
lib = CDLL(os.path.join(
        str(Path(__file__).parent.absolute()),
        'solver/solver.so'
    ))

c_solver = lib.interface

c_solver.argtypes = [
    ndpointer(c_double, flags = "C_CONTIGUOUS"),
    c_int,
    c_int,
    c_int
]

c_solver.restype = POINTER(c_double)

c_free_ptr = lib.free_ptr


def call_my_solver(constraint_matrix):
    """
    Calls my Mehrotra Predictor-Corrector implementation. 

    This is more or less a general purpose LP solver, so it really just expects a constraint matrix. The RHS is locked as a vector of 1s for now.
    """
    num_rows, num_cols = constraint_matrix.shape
    num_workers = 1
    result = c_solver(constraint_matrix, num_rows, num_cols, num_workers)
    result_array = as_array(result, shape = (num_rows+num_cols,))
    objective = sum(result_array[num_cols:])
    c_free_ptr(result)

    return objective


