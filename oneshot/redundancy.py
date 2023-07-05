from new_constraints import constraints
from mysolver_interface import call_my_solver
from itertools import product, combinations
from random import choice, choices, randint
import click
import torch
import torch.nn.functional as F
from copy import deepcopy
from ast import literal_eval
import numpy as np
from tqdm import tqdm
from verify import parseaux
from descent2 import hash_constraints
import matplotlib.pyplot as plt

cache = {}
cache_hits = 0
cache_misses = 0

def make_wrapper(n1, n2, desired, included):

    def attempt(radius, functions):
        global cache_hits, cache_misses
        M, _, _ = constraints(n1, n2, radius = radius, desired=desired, included=included, bool_funcs = functions)
        key = (radius, hash_constraints(M))
        if key in cache:
            cache_hits += 1
            rhos = cache[key]
        else:
            cache_misses += 1
            objective, coeffs, rhos, _, _ = call_my_solver(M.to_sparse_csc(), fullreturn = True)
            cache[key] = rhos
        return rhos

    return attempt


n1, n2 = 4,4
desired = (0,1,2,3,4,5,6)
included = (7,)
num_inputs = n1 + n2 + (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1 + n2)
num_vars = num_inputs + num_outputs
radius = 1
attempt_fn = make_wrapper(n1, n2, desired, included)

thresh_power = 3

orbits = []
with open(f'dat/orbit{thresh_power}.dat', 'r') as FILE:
    for line in FILE.readlines():
        orbits.append(parseaux(literal_eval(line)))

neutrals = []
for orbit in orbits:
    neutrals.append(orbit[0])
    

reference_objective = sum(attempt_fn(radius, None))

#and_gate_list = list(product(range(num_inputs), range(num_inputs, num_vars)))
gate_list = []
for func in neutrals:
    for indices in combinations(range(num_vars), thresh_power):
        if min(indices) >= num_inputs or max(indices) < num_inputs:
            continue
        gate_list.append((indices, func))


X = []
Y = []
loop = tqdm(choices(list(combinations(gate_list, 2)), k = 10000), leave=True)
for gate1, gate2 in loop:
    rho1 = attempt_fn(radius, [gate1])
    if reference_objective - sum(rho1) < 1:
        continue
    rho2 = attempt_fn(radius, [gate2])
    if reference_objective - sum(rho2) < 1:
        continue
    combined_rho = attempt_fn(radius, [gate1, gate2])
    
    bb_heuristic = (reference_objective - sum(rho1)) + (reference_objective - sum(rho2)) - (reference_objective - sum(combined_rho))

    #rho_heuristic = sum(abs(rho1-rho2))
    rho1 = torch.tensor(rho1)
    rho2 = torch.tensor(rho2)
    rho_heuristic = torch.corrcoef(torch.cat([rho1.unsqueeze(0), rho2.unsqueeze(0)]))[0,1]

    X.append(bb_heuristic)
    Y.append(rho_heuristic)

plt.scatter(X, Y, alpha=0.1)
plt.savefig('redundancy.png')


    


