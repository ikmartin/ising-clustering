from ctypes import c_int, c_bool, POINTER, c_double, CDLL, c_void_p
import numpy as np
from numpy.ctypeslib import ndpointer
from sys import path
import os
from pathlib import Path

# Statically load the interface so that the solver can be called.

path.insert(1, '/opt/OpenBLAS')
lib = CDLL(os.path.join(
        str(Path(__file__).parent.absolute()),
        'solver/solver.so'
    ))

c_solver = lib.interface

c_solver.argtypes = [
    ndpointer(c_double),
    c_int,
    c_int,
    c_int
]

c_solver.restype = POINTER(c_double)


def call_my_solver(constraint_matrix):
    """
    Calls my Mehrotra Predictor-Corrector implementation. 

    This is more or less a general purpose LP solver, so it really just expects a constraint matrix. The RHS is locked as a vector of 1s for now.
    """
    num_rows, num_cols = constraint_matrix.shape
    num_workers = 1
    print(constraint_matrix)
    result = c_solver(constraint_matrix, num_rows, num_cols, num_workers)
    print(result)


    #c_free_ptr(result)


