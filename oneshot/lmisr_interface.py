from ctypes import c_int, c_bool, POINTER, c_double, CDLL, c_void_p
from numpy.ctypeslib import ndpointer, as_array
import numpy as np
from sys import path
from pathlib import Path
import os

path.insert(1, "/opt/OpenBLAS")

# Statically load the interface so that the solver can be called.
try:
    # path.insert(1, '/opt/OpenBLAS')
    lib = CDLL(os.path.join(str(Path(__file__).parent.absolute()), "lmisr/isingLPA.so"))
except OSError:
    lib = CDLL("/home/ikmarti/Desktop/ising-clustering/oneshot/lmisr/isingLPA.so")

c_free_ptr = lib.free_ptr
c_lmisr = lib.lmisr
c_lmisr.restype = POINTER(c_double)


def call_solver(N1: int, N2: int, aux_array: np.ndarray, fullreturn=False):
    """
    Calls Teresa's Mehrotra Predictor-Corrector implementation.

    The circuit which will be solved is IMul(N1, N2) with the specified auxilliary array.

    Aux array is expected to be nonempty.
    """

    num_aux_spins = aux_array.shape[0]

    aux_array = aux_array.astype(np.dtype("uint8"))
    c_formatted_aux_array = c_void_p(aux_array.ctypes.data)

    N = N1 + N2
    num_input_levels = 1 << N

    result = c_lmisr(
        c_int(N1),
        c_int(N2),
        c_int(num_aux_spins),
        c_int(N),
        c_int(num_input_levels),
        c_formatted_aux_array,
    )

    result_array = as_array(result, shape=((1 << N) + 2,))

    objective = result[0]

    c_free_ptr(result)

    if fullreturn:
        # sum of rho, my_frac, rho
        return result_array[0], result_array[1], result_array[2:]

    return objective
