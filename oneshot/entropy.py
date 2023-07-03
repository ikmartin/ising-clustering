from new_constraints import constraints as get_constraints
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

from fast_constraints import batch_vspin, dec2bin, all_answers_basic

torch.set_printoptions(threshold = 50000)
np.set_printoptions(threshold = 50000)


n1, n2 = 3,3
num_input_levels = 1 << (n1+n2)
desired = (0,1,2,3,4,5)

g = n1 + n2 + len(desired)
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
gate_list = []
for func in neutrals:
    for indices in combinations(range(g), thresh_power):
        if min(indices) > n1+n2 or max(indices) < n1 + n2:
            continue
        gate_list.append((indices, func))


degree = 2
all_states = all_answers_basic(g)

inp2_mask = 2 ** (n2) - 1
correct_rows = torch.cat(
    [
        torch.tensor([[inp, (inp & inp2_mask) * (inp >> n2)]])
        for inp in range(2 ** (n1 + n2))
    ]
)
col_select = torch.cat([torch.arange(n1+n2), torch.tensor(desired) + n1+n2])
correct_rows = torch.flatten(dec2bin(correct_rows, n1 + n2), start_dim=-2)[..., col_select]

virtual_right = batch_vspin(correct_rows, degree)
rows_per_input = (2 ** len(desired))
exp_virtual_right = (
    virtual_right.unsqueeze(1)
    .expand(-1, rows_per_input, -1)
    .reshape(2 ** (n1 + n2) * rows_per_input, -1)
)
virtual_all = batch_vspin(all_states, degree)
constraints = virtual_all - exp_virtual_right

# filter out the rows with correct answers
row_mask = constraints[..., (n1 + n2) : g].any(dim=-1)

# filter out connections between inputs
col_mask = torch.tensor(constraints.any(dim=0))
constraints = constraints[row_mask][..., col_mask]

objective, coeffs, rhos, initcoeffs, initrhos = call_my_solver(constraints.to_sparse_csc(), fullreturn = True)



energies = F.linear(virtual_all[...,col_mask].double(), torch.tensor(coeffs))
energies_by_input_level = energies.reshape(num_input_levels, 1 << len(desired))
boltzmann_factors = torch.exp(-energies_by_input_level)
relative_probs = F.normalize(boltzmann_factors, p=1)/num_input_levels

flat_probs = torch.flatten(relative_probs).double()

def bool_func(matrix, func):
    indices, function = func
    args = matrix[...,indices]
    new_vec = torch.zeros(matrix.shape[0])
    two_powers = torch.tensor([(1 << (len(indices) - j - 1)) for j in range(len(indices))])
    vals = F.linear(args.float(), two_powers.float()).int()
    new_vec = torch.tensor(function[vals])
    return new_vec


loop = tqdm(range(1000), leave=True)
infos = []
errors = []
a = 1
true_answers = correct_rows[...,-1]
exp_answers = true_answers.unsqueeze(1).expand(-1, 1 << len(desired)).flatten()
print(exp_answers)

for i in loop:
    funcs = [choice(gate_list) for k in range(a)]
    patterns = torch.cat([
        bool_func(all_states, func).double().unsqueeze(1)
        for func in funcs
        ],
        dim = -1
    )
    #patterns = torch.cat([all_states[...,(n1+n2):], patterns], dim = -1)
    expected_info = 0
    binary_patterns = dec2bin(torch.arange(1 << a), a)
    for row in binary_patterns:
        matches = (~(((patterns+row) % 2).any(dim=-1))).int().double()
        for possible_answer in range(1 << len(desired)):
            answer_mask = (exp_answers == possible_answer).int().double()
            prob = torch.sum(matches * flat_probs * answer_mask)
            if prob > 1e-7:
                expected_info -= prob * np.log2(prob)

    m, _, _ = get_constraints(n1, n2, desired = desired, bool_funcs = funcs)
    objective = call_my_solver(m.to_sparse_csc())
    infos.append(expected_info)
    errors.append(objective)

print(infos)
print(errors)
plt.scatter(infos, errors)
plt.savefig(f'info{n1}x{n2}x{a}-{thresh_power}-{desired}.png')

""" test outputs
print(relative_probs)
print(objective)
true_answers = []

print("[predicted/true/rho]")

print(np.concatenate([
    torch.argmax(relative_probs, dim = -1).unsqueeze(0).numpy(),
    np.array([true_answers]),
    np.expand_dims((rhos*100).astype(int).reshape(num_input_levels,num_input_levels-1).sum(axis=-1), axis=0)
]))

"""
