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
import matplotlib.pyplot as plt
from time import perf_counter as pc

n1, n2 = 3,4
desired = (0,1,2,3,4,5)
included = (6,)
num_aux = 7
num_inputs = n1 + n2 + (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1 + n2)
num_vars = num_inputs + num_outputs
radius = 1

thresh_power = 4

with open(f'dat/neutralizable_threshold_functions_dim{thresh_power}.dat', 'r') as FILE:
    functions = []
    for line in FILE.readlines():
        bool_line = literal_eval(line)
        if bool_line[0]:
            continue
        func_line = np.array([bool_line]).astype(np.int8)
        functions.append(func_line)
    neutrals = np.concatenate(functions)
    print(neutrals)

#and_gate_list = list(product(range(num_inputs), range(num_inputs, num_vars)))
and_gate_list = []
for func in neutrals:
    for indices in combinations(range(num_vars), thresh_power):
        and_gate_list.append((indices, func))

loop = tqdm(range(500), leave=True)
lams = []
initlams = []
full_time = 0
est_time = 0
for i in loop:
    current_gates = choices(and_gate_list, k=num_aux)
    M, _, _ = constraints(n1, n2, radius = radius, desired = desired, included = included, bool_funcs = current_gates)
    start = pc()
    true_objective = call_my_solver(M.to_sparse_csc())
    end = pc()
    full_time += end-start

    start = pc()
    est_objective = call_my_solver(M.to_sparse_csc(), max_iter = 5) 
    end = pc()
    est_time += end-start
    
    lams.append(true_objective)
    initlams.append(est_objective)

print(f'{full_time} {est_time}')
plt.scatter(lams, initlams)
plt.savefig('fig346-5iter.png')

