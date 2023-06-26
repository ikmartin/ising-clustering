from new_constraints import constraints
from mysolver_interface import call_my_solver
import numpy as np
import torch

np.set_printoptions(precision = 2, suppress = True)

aux_array = None

for i in range(5):
    aux_tensor = torch.tensor(aux_array) if aux_array is not None else None
    M, terms, correct = constraints(3, 3, aux_tensor, 2, None, None, (2,), None, ising=True)
    M = M/2
    print(M)
    print(terms)
    objective, coeffs, rhos = call_my_solver(M.to_sparse_csc(), tolerance = 1e-8, fullreturn = True)
    print(objective)
    print(rhos)

    if objective < 1e-2:
        break

    new_aux_vec = np.sign(coeffs[0] + np.dot(coeffs[1:], correct[...,:-1].T))
    new_aux_vec[new_aux_vec == 0] = 1       # might happen, but numerical problems with solver should prvent it
    new_aux_vec = (1+new_aux_vec)/2
    new_aux_vec = np.expand_dims(new_aux_vec.astype(np.int8), axis=0)
    print(new_aux_vec)
    aux_array = new_aux_vec if aux_array is None else np.concatenate([aux_array, new_aux_vec])

