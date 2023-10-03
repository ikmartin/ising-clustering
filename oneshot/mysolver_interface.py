from copy import deepcopy
from ctypes import c_int, c_bool, POINTER, c_double, CDLL, c_void_p, c_int8
import numpy as np
from numpy.ctypeslib import ndpointer, as_array
from sys import path
import os
from pathlib import Path
import torch
from math import comb

# Statically load the interface so that the solver can be called.
try:
    # path.insert(1, '/opt/OpenBLAS')
    lib = CDLL(os.path.join(str(Path(__file__).parent.absolute()), "solver/solver.so"))
except OSError:
    lib = CDLL("/home/ikmarti/Desktop/ising-clustering/oneshot/solver/solver.so")

c_solver = lib.interface
c_solver.argtypes = [
    ndpointer(c_double, flags="C_CONTIGUOUS"),
    c_int,
    c_int,
    c_int,
    c_double,
    c_int,
]
c_solver.restype = POINTER(c_double)

c_sparse_solver = lib.sparse_interface
c_sparse_solver.argtypes = [
    c_int,
    c_int,
    ndpointer(c_int8, flags="C_CONTIGUOUS"),
    ndpointer(c_int, flags="C_CONTIGUOUS"),
    ndpointer(c_int, flags="C_CONTIGUOUS"),
    c_int,
    c_double,
    c_int,
    c_double,
    c_double,
    c_int,
    ndpointer(c_double, flags = "C_CONTIGUOUS")
]
c_sparse_solver.restype = POINTER(c_double)

c_full_sparse_solver = lib.full_sparse_interface
c_full_sparse_solver.argtypes = [
    c_int,
    c_int,
    ndpointer(c_int8, flags="C_CONTIGUOUS"),
    ndpointer(c_int, flags="C_CONTIGUOUS"),
    ndpointer(c_int, flags="C_CONTIGUOUS"),
    ndpointer(c_int8, flags="C_CONTIGUOUS"),
    ndpointer(c_int, flags="C_CONTIGUOUS"),
    ndpointer(c_int, flags="C_CONTIGUOUS"),
    c_int,
    c_double,
    c_int,
]
c_full_sparse_solver.restype = POINTER(c_double)

c_get_initial_lam = lib.get_initial_lam
c_get_initial_lam.restype = POINTER(c_double)

c_imul_solver = lib.IMul_interface
c_imul_solver.argtypes = [
    c_int,
    c_int,
    ndpointer(c_int8, flags="C_CONTIGUOUS"),
    c_int,
    c_int,
    c_int,
    c_int,
    c_double,
    c_int,
]
c_imul_solver.restype = POINTER(c_double)

c_free_ptr = lib.free_ptr


def call_my_solver(CSC_constraints, tolerance=1e-8, max_iter=1000, fullreturn=False, RHS = None):
    """
    Calls my Mehrotra Predictor-Corrector implementation.

    This is more or less a general purpose LP solver, so it really just expects a constraint matrix. The RHS is locked as a vector of 1s for now.
    """


    num_rows, num_cols = CSC_constraints.size()

    if RHS is None:
        RHS = np.ones(num_rows)

    values = CSC_constraints.values().numpy().astype(np.int8)
    row_index = CSC_constraints.row_indices().numpy().astype(np.int32)
    col_ptr = CSC_constraints.ccol_indices().numpy().astype(np.int32)
    num_workers = 1
    result = c_sparse_solver(
        num_rows, num_cols, values, row_index, col_ptr, num_workers, tolerance, max_iter, 0.95, 0.15, int(fullreturn), RHS
    )
    result_array = as_array(result, shape=(num_rows + num_cols,))
    objective = sum(result_array[num_cols:])

    if fullreturn:
        # objective, answer (i.e. solution to LP), artifical variables
        result_array = deepcopy(result_array)
        saved_lam = c_get_initial_lam()
        saved_lam_array = deepcopy(as_array(saved_lam, shape=(num_rows + num_cols,)))
        c_free_ptr(saved_lam)
        c_free_ptr(result)
        return objective, result_array[:num_cols], result_array[num_cols:], saved_lam_array[:num_cols], saved_lam_array[num_cols:]

    c_free_ptr(result)
    return objective

def call_full_sparse(CSC_constraints, CSR_constraints, tolerance=1e-8, max_iter=200, fullreturn=False):
    """
    Calls my Mehrotra Predictor-Corrector implementation.

    This is more or less a general purpose LP solver, so it really just expects a constraint matrix. The RHS is locked as a vector of 1s for now.
    """
    num_rows, num_cols = CSC_constraints.size()
    csc_values = CSC_constraints.values().numpy().astype(np.int8)
    csc_row_index = CSC_constraints.row_indices().numpy().astype(np.int32)
    csc_col_ptr = CSC_constraints.ccol_indices().numpy().astype(np.int32)
    csr_values = CSR_constraints.values().numpy().astype(np.int8)
    csr_col_index = CSR_constraints.col_indices().numpy().astype(np.int32)
    csr_row_ptr = CSR_constraints.crow_indices().numpy().astype(np.int32)

    num_workers = 1
    result = c_full_sparse_solver(
        num_rows, num_cols, csc_values, csc_row_index, csc_col_ptr, csr_values, csr_col_index, csr_row_ptr, num_workers, tolerance, max_iter
    )
    result_array = as_array(result, shape=(num_rows + num_cols,))
    objective = sum(result_array[num_cols:])

    if fullreturn:
        # objective, answer (i.e. solution to LP), artifical variables
        result_array = deepcopy(result_array)
        c_free_ptr(result)
        return objective, result_array[:num_cols], result_array[num_cols:]

    c_free_ptr(result)
    return objective

def call_imul_solver(n1, n2, aux_array):
    num_aux = aux_array.shape[0]
    N = n1 + n2
    G = 2 * N + num_aux
    num_rows = int(2 ** (2 * N + num_aux) - 2 ** (N + num_aux))
    num_cols = int(G - N + comb(G, 2) - comb(N, 2))
    num_workers = 1
    tolerance = 1e-8
    max_iter = 200
    result = c_imul_solver(
        n1, n2, aux_array, num_aux, num_rows, num_cols, num_workers, tolerance, max_iter
    )
    result_array = as_array(result, shape=(num_rows + num_cols,))
    objective = sum(result_array[num_cols:])
    c_free_ptr(result)

    return objective
