from lmisr_interface import call_solver
from fast_constraints import constraints_building
import numpy as np
from solver import LPWrapper
import torch

np.set_printoptions(threshold = 50000000)
n1 = 4
n2 = 4
# working 3x4
#aux_keys = [(0,9), (5,12), (1,13), (1,11), (0,11), (4,10), (3,10)]


aux_keys = [(2, 14), (6, 14), (5, 8), (1, 13), (5, 13), (3, 6), (0, 9), (0, 10), (4, 11), (1, 12), (0, 12), (2, 12), (5, 11)]

#aux_keys = [(1,2), (4,10), (0,5), (0,8), (3,9), (0,9)]
_, _, correct = constraints_building(n1,n2, None, 1, radius=None, mask=None, include=None)
correct = correct.numpy()

aux_vecs = np.concatenate([np.expand_dims(np.prod(correct[...,key], axis=-1), axis=0) for key in aux_keys]).astype(np.int8)
print(aux_vecs)
with open("saved-aux", "w") as FILE:
    FILE.write(str(aux_vecs.tolist()))

"""
print("making constraints")
constraints, keys, _ = constraints_building(n1,n2, torch.tensor(aux_vecs), 2, radius = None, mask = None, include = None)
print("making solver")
solver = LPWrapper(keys)
constraints = constraints.to_sparse()
print("loading constraints")
solver.add_constraints(constraints)
print("solving")
solution = solver.solve()
print(solution)
"""

objective = call_solver(n1,n2, aux_vecs)
print(objective)
