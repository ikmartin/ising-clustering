from mysolver_interface import call_my_solver, call_imul_solver
from lmisr_interface import call_solver
from fast_constraints import fast_constraints, constraints_basin1, CSC_constraints
from ising import IMul
from solver import LPWrapper

import numpy as np
from time import perf_counter as pc

import torch

n1, n2, A = 4,4,6
my_time = 0
glop_time = 0

aux_array = np.random.choice([-1,1], (A, 2 ** (n1 + n2)), p = [0.5, 0.5])

aux_tensor = torch.tensor(aux_array).clamp(min=0)

start = pc()
constraints, keys = constraints_basin1(n1, n2, aux_tensor, 2)
objective = call_my_solver(constraints.to_sparse_csc(), tolerance = 1e-1)
end = pc()
my_time += end-start
myanswer = objective > 1
print(objective)

start = pc()
constraints, keys = constraints_basin1(n1, n2, aux_tensor, 2)
glop = LPWrapper(keys)
glop.add_constraints(constraints.to_sparse())
result = glop.solve()
end = pc()
glop_time += end-start

glopanswer = result is None

print(f'{my_time} {myanswer} {glop_time} {glopanswer}')