from new_constraints import constraints, make_correct_rows, make_wrongs, add_bool_func
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

from sklearn.ensemble import RandomForestRegressor

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


correct, num_fixed, num_free = make_correct_rows(n1, n2, included = included, desired = desired)
wrong, rows_per_input = make_wrongs(correct, num_fixed, num_free, radius)
allrows = torch.cat([correct, wrong])

thresh_power = 3

orbits = []
with open(f'dat/orbit{thresh_power}.dat', 'r') as FILE:
    for line in FILE.readlines():
        orbits.append(parseaux(literal_eval(line)))

neutrals = []
for orbit in orbits:
    neutrals.append(orbit[0])
    

reference_rho = attempt_fn(radius, None)
reference_objective = sum(reference_rho)
reference_rho = torch.tensor(reference_rho)

#and_gate_list = list(product(range(num_inputs), range(num_inputs, num_vars)))
gate_list = []
for func in neutrals:
    for indices in combinations(range(num_vars), thresh_power):
        if min(indices) >= num_inputs or max(indices) < num_inputs:
            continue
        gate_list.append((indices, func))


X = []
Y = []
X_test = []
Y_test = []
num_test = 5000
num_train = 100000
loop = tqdm(choices(list(combinations(gate_list, 2)), k = num_test + num_train), leave=True)
i = 0
for gate1, gate2 in loop:
    i += 1
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
    #rho_heuristic = torch.corrcoef(torch.cat([(reference_rho - rho1).unsqueeze(0), (reference_rho - rho2).unsqueeze(0)]))[0,1]

    f1 = add_bool_func(allrows, gate1)[..., -1]
    f2 = add_bool_func(allrows, gate2)[..., -1]
    #f_heuristic = torch.corrcoef(torch.cat([f1.unsqueeze(0), f2.unsqueeze(0)]))[0,1]

    x_sample = np.expand_dims(np.concatenate([f1.numpy(), rho1.numpy(), f2.numpy(), rho2.numpy()]), axis=0)
    y_sample = bb_heuristic

    if i < num_train:
        X.append(x_sample)
        Y.append(y_sample)
    else:
        X_test.append(x_sample)
        Y_test.append(y_sample)


regr = RandomForestRegressor(n_jobs = 16)
print("fitting model")
regr.fit(np.concatenate(X), np.array(Y))

print("plotting")
plt.scatter(regr.predict(np.concatenate(X)), np.array(Y), alpha=0.1)
plt.scatter(regr.predict(np.concatenate(X_test)), np.array(Y_test), alpha=0.2)
plt.savefig('redundancy.png')


