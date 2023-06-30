from new_constraints import constraints
from mysolver_interface import call_my_solver
from itertools import product
from random import choice, choices, randint
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

n1, n2 = 3,3
desired = (0,1,2,3,4)
included = (5,)
num_aux = 4
num_inputs = n1 + n2 + (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1 + n2)
num_vars = num_inputs + num_outputs
radius = 1

#and_gate_list = list(product(range(num_inputs), range(num_inputs, num_vars)))
and_gate_list = []
for x, y in list(product(range(num_inputs), range(num_inputs, num_vars))):
    hyperplane = torch.zeros(num_vars)
    hyperplane[x] = 1
    hyperplane[y] = 1
    and_gate_list.append((hyperplane, 1.5))


weights = torch.randn(num_aux, num_inputs)
weights = F.normalize(weights, dim = 1)
bias = torch.randn(num_aux)

current_objective = 1000000
cache = {}

temperature = 0.1
iternum = 0
while True:
    iternum += 1
    #print(temperature)
    new_weights = F.normalize(temperature * torch.randn(num_aux, num_inputs) + weights)
    new_bias = temperature * torch.randn(num_aux) + bias
    
    planes = []
    for i in range(num_aux):
        planes.append((new_weights[i], new_bias[i].item()))

    M, _, _ = constraints(n1, n2, radius = radius, desired = desired, included = included, hyperplanes = planes)

    key = hash(M.numpy().data.tobytes())
    if key in cache:
        objective = cache[key]
    else:
        objective = call_my_solver(M.to_sparse_csc())
        cache[key] = objective

    if objective < current_objective:
        current_objective = objective
        temperature = max(0.01, objective / 300)
        weights = new_weights
        bias = new_bias
        print(f'{iternum} {objective}')
        while current_objective < 1e-2:
            if radius < num_outputs:
                radius += 1
                cache = {}
                M, _, _ = constraints(n1, n2, radius = radius, desired = desired, included = included, hyperplanes = planes)
                current_objective = call_my_solver(M.to_sparse_csc())
                print(f'radius now {radius} objective {current_objective}')
            else:
                print(planes)
                exit()
