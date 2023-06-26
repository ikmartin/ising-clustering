from mysolver_interface import call_my_solver, call_imul_solver, call_full_sparse
from lmisr_interface import call_solver
from new_constraints import constraints
from ising import IMul
from solver import LPWrapper

import numpy as np
from time import perf_counter as pc

import torch

n1, n2, A = 3,3,7
my_time = 0
full_sparse_time = 0
glop_time = 0

aux_array = np.random.choice([-1, 1], (A, 2 ** (n1 + n2)), p=[0.5, 0.5])

aux_tensor = torch.tensor(aux_array).clamp(min=0)

start = pc()
M, keys, correct = constraints(n1, n2, aux_tensor, 2, None, None, None, None)
objective = call_my_solver(M.to_sparse_csc())
end = pc()
print(f'{objective} {end-start}')

start = pc()
M, keys, correct = constraints(n1, n2, aux_tensor, 2, None, None, None, None)
objective = call_full_sparse(M.to_sparse_csc(), M.to_sparse_csr())
end = pc()
print(f'{objective} {end-start}')

