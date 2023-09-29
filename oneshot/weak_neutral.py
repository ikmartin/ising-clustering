from new_constraints import make_correct_rows, make_wrongs, add_hyperplane
from fast_constraints import dec2bin, batch_vspin, keys_basic
from itertools import combinations
from random import choice
from mysolver_interface import call_my_solver
import torch
import numpy as np
from tqdm import tqdm

inputs_per_threshold = 5

n1, n2 = 3, 4
num_variables = 2 * (n1 + n2)
degree = 2

correct, num_fixed, num_free = make_correct_rows(n1, n2)
wrong, rows_per_input = make_wrongs(correct, num_fixed, num_free)
all_base_states = dec2bin(torch.arange(1 << num_variables), num_variables)

def test_F(F):
    right_aug_states = add_hyperplane(all_base_states, F)
    wrong_aug_states = torch.cat([all_base_states, (1-right_aug_states[...,-1]).unsqueeze(-1)], dim = -1)
    base_right = add_hyperplane(correct, F)
    base_wrong = add_hyperplane(wrong, F)

    virtual_right = batch_vspin(base_right, degree)
    exp_virtual_right = (
        virtual_right.unsqueeze(1)
        .expand(-1, rows_per_input, -1)
        .reshape(wrong.shape[0], -1)
    )
    virtual_wrong = batch_vspin(base_wrong, degree)

    virtual_aug_right = batch_vspin(right_aug_states, degree)
    virtual_aug_wrong = batch_vspin(wrong_aug_states, degree)

    full_virtual_right = torch.cat([exp_virtual_right, virtual_aug_right])
    full_virtual_wrong = torch.cat([virtual_wrong, virtual_aug_wrong])

    constraints = full_virtual_wrong - full_virtual_right

    RHS = torch.cat([torch.zeros(exp_virtual_right.shape[0]), torch.ones(virtual_aug_right.shape[0])])

    col_mask = constraints.any(dim = 0)
    row_mask = constraints.any(dim = 1)
    constraints = constraints[row_mask][...,col_mask]
    RHS = RHS[row_mask]

    objective = call_my_solver(constraints.to_sparse_csc(), RHS = RHS.numpy().astype(np.float64))

    return objective

with open(f"dat/WNTF{n1}{n2}.dat", "a") as FILE:
    finds = 0
    loop = tqdm(range(10000), leave=True)
    for i in loop:
        w = torch.zeros(num_variables)
        indices = choice(list(combinations(range(num_variables), inputs_per_threshold)))
        for i in indices:
            w[i] = torch.randn(1)
        b = torch.randn(1).item()
        #print(w)
        obj = test_F((w,b))
        #print(obj)

        if obj < 1:
            FILE.write(str((w.tolist(),b)))
            FILE.write("\n")
            FILE.flush()
            finds += 1

        loop.set_postfix(finds = finds)




