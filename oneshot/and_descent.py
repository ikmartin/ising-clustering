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


def run(num_aux):
    n1, n2 = 4,4
    desired = (0,1,2,3,4,5,6)
    included = (7,)
    num_inputs = n1 + n2 + (len(included) if included is not None else 0)
    num_outputs = (len(desired) if desired is not None else n1 + n2)
    num_vars = num_inputs + num_outputs
    radius = 1

    thresh_power = 3

    """
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
    """

    orbits = []
    with open(f'dat/orbit{thresh_power}.dat', 'r') as FILE:
        for line in FILE.readlines():
            orbits.append(parseaux(literal_eval(line)))

    neutrals = []
    for orbit in orbits:
        neutrals.append(orbit[0])
        

    #and_gate_list = list(product(range(num_inputs), range(num_inputs, num_vars)))
    and_gate_list = []
    for func in neutrals:
        for indices in combinations(range(num_vars), thresh_power):
            if min(indices) >= num_inputs or max(indices) < num_inputs:
                continue
            and_gate_list.append((indices, func))

    current_gates = choices(and_gate_list, k=num_aux)
    current_objective = 1000000
    total_tasks = len(and_gate_list) * num_aux
    num_since_last_success = 0

    cache = {}
    maxiter = 2

    while True:
        for i in range(num_aux):
            print(i)
            loop = tqdm(and_gate_list, leave=True)
            for gate in loop:#choices(and_gate_list, k=20):
                loop.set_postfix(objective = current_objective)
                new_gate_list = deepcopy(current_gates)
                new_gate_list[i] = gate
                M, _, _ = constraints(n1, n2, radius = radius, desired = desired, included = included, bool_funcs = new_gate_list)
                key = hash(M.numpy().data.tobytes())
                if key in cache:
                    objective = cache[key]
                else:
                    objective = call_my_solver(M.to_sparse_csc())
                    cache[key] = objective
                if objective < current_objective:
                    num_since_last_success = 0
                    current_objective = objective
                    current_gates = new_gate_list
                    #print(f'{i} {objective}')
                    while current_objective < 1:
                        if radius < num_outputs:
                            radius += 1
                            cache = {}
                            M, _, _ = constraints(n1, n2, radius = radius, desired = desired, included = included, bool_funcs = new_gate_list)
                            current_objective = call_my_solver(M.to_sparse_csc())
                            print(f'radius now {radius} objective {current_objective}')
                        else:
                            print(current_gates)
                            return

                else:
                    num_since_last_success += 1
                    if num_since_last_success > total_tasks:
                        print("stuck")
                        return

@click.command()
@click.option('--num_aux', default = 0, help = 'number of aux')
def main(num_aux):
    while True:
        run(num_aux)


if __name__ == '__main__':
    main()
