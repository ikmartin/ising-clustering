from new_constraints import constraints
from mysolver_interface import call_my_solver
from itertools import product, combinations
from random import choice, choices, randint
import torch
import torch.nn.functional as F
from copy import deepcopy
from ast import literal_eval
import numpy as np
from tqdm import tqdm
from fast_constraints import all_answers_basic
import pandas as pd

n1, n2 = 3,3
desired = (0,1,2,3,4)
included = (5,)
num_aux = 4
num_inputs = n1 + n2 + (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1 + n2)
num_vars = num_inputs + num_outputs

M, _, _ = constraints(n1, n2, desired = desired, included = included)
reference_objective = call_my_solver(M.to_sparse_csc())

hyperplanes = []
loop = tqdm(range(10000), leave=True)
for i in loop:
    weights = torch.randn(num_vars)
    weights = F.normalize(weights, dim=0)
    bias = torch.rand(1).item()
    hyperplanes.append((weights, bias))

planes = []
results = []

all_states = all_answers_basic(2*(n1+n2))
all_states = torch.cat([all_states[...,:(n1+n2)], all_states[..., included], all_states[..., desired]], dim = -1)

loop = tqdm(hyperplanes, leave=True)
for plane in loop:
    M, _, correct = constraints(n1, n2, desired = desired, included = included, hyperplanes = [plane])
    
    objective = call_my_solver(M.to_sparse_csc())
    results.append(objective/reference_objective)

    pattern = torch.sign((F.linear(all_states.double(), plane[0].double()) - plane[1]))
    pattern = ((pattern + 1)/2).int()
    planes.append(pattern.tolist())

df = pd.DataFrame({'planes': planes, 'objective': results})
df.to_csv('3x3.csv')
