from constraints import traditional_constraints
import torch
from ising import IMul
from math import prod

degree = 3
circuit = IMul(3,3)
constraints = traditional_constraints(circuit, degree)
print(constraints)
print(constraints.shape)
total_elements = prod(constraints.shape)
print(constraints._nnz() / total_elements)
