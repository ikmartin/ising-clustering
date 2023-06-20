from lmisr_interface import call_solver
from fast_constraints import constraints_building
import numpy as np
from solver import LPWrapper
import torch

np.set_printoptions(threshold = 50000000)
#aux_keys = [(0,9), (5,12), (1,13), (0,10), (3,11), (1,11), (0,11), (3,10)]
aux_keys = [(1,2), (4,10), (0,5), (0,8), (3,9), (0,9)]
_, _, correct = constraints_building(3, 3, None, 1, radius=None, mask=None, include=None)
correct = correct.numpy()

aux_vecs = np.concatenate([np.expand_dims(np.prod(correct[...,key], axis=-1), axis=0) for key in aux_keys]).astype(np.int8)
print(aux_vecs)

print("making constraints")
constraints, keys, _ = constraints_building(3, 3, torch.tensor(aux_vecs), 2, radius = None, mask = None, include = None)
print("making solver")
solver = LPWrapper(keys)
constraints = constraints.to_sparse()
print("loading constraints")
solver.add_constraints(constraints)
print("solving")
solution = solver.solve()
print(solution)

objective = call_solver(3, 3, aux_vecs)
print(objective)
