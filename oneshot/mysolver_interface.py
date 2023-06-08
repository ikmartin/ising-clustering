from ctypes import c_int, c_bool, POINTER, c_double, CDLL, c_void_p
import numpy as np
from sys import path
import os
from pathlib import Path

path.insert(1, '/opt/OpenBLAS')
lib = CDLL(os.path.join(
        str(Path(__file__).parent.absolute()),
        'solver/solver.so'
    ))

c_solver = lib.interface
c_solver.restype = c_double


def call_my_solver(N1: int, N2: int, aux_array: np.ndarray):
    """
    Calls my Mehrotra Predictor-Corrector implementation. 

    The circuit which will be solved is IMul(N1, N2) with the specified auxilliary array.

    Aux array is expected to be nonempty.
    """

    num_aux_spins = aux_array.shape[0]

    aux_array = aux_array.astype(np.dtype('uint8'))
    c_formatted_aux_array = c_void_p(aux_array.ctypes.data)

    N = N1 + N2
    num_input_levels = 1 << N

    

    result = c_lmisr(
        c_int(N1),
        c_int(N2),
        c_int(num_aux_spins),
        c_int(N),
        c_int(num_input_levels),
        c_formatted_aux_array
    )


    objective = result[0]


    c_free_ptr(result)

    return objective < 1e-4
