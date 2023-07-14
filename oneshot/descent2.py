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

cache = {}
cache_hits = 0
cache_misses = 0

def hash_constraints(M):
    colset = frozenset({
        M[...,i].numpy().data.tobytes()
        for i in range(M.shape[1])
    })
    return colset


def make_wrapper(n1, n2, desired, included):


    def attempt(radius, functions):
        global cache_hits, cache_misses
        M, _, _ = constraints(n1, n2, radius = radius, desired=desired, included=included, bool_funcs = functions)
        key = (radius, hash_constraints(M))
        if key in cache:
            cache_hits += 1
            objective = cache[key]
        else:
            cache_misses += 1
            objective = call_my_solver(M.to_sparse_csc(), tolerance=1e-4)
            cache[key] = objective
        return objective

    return attempt


def run(num_aux):
    global cache, cache_hits, cache_misses

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
        

    neutrals = [np.array([1,0,0,0,1,1,1,0])]
    neutrals = [np.array([0, 1, 1, 0, 1, 0, 0, 1])]


    #thresh_power = 2
    #neutrals = [np.array([1,1,1,0])]

    #and_gate_list = list(product(range(num_inputs), range(num_inputs, num_vars)))
    gate_list = []
    for func in neutrals:
        for indices in combinations(range(num_vars), thresh_power):
            if min(indices) >= num_inputs or max(indices) < num_inputs:
                continue
            gate_list.append((indices, func))


    # Try out all the gates in isolation to give them an approximate score
    print("Calculating initial scores...")
    base_objective = attempt_fn(radius, None)
    progress = tqdm(gate_list, desc = 'scores', leave=True)
    reference_scores = [base_objective for func in progress]
    gates_with_progress = sorted(zip(reference_scores, gate_list), key = lambda item: -item[0])
    print(f'Cache efficiency: {cache_hits/(cache_hits+cache_misses)}')
    
    print("Generating initial aux array...")
    aux_maps = choices(gates_with_progress, k = num_aux)
    current_objective = attempt_fn(radius, [pair[1] for pair in aux_maps])
    print(f'Initial objective is {current_objective}')
    
    total_tasks = len(gate_list) * num_aux
    num_since_last_success = 0


    block_set = set([])
    while True:
        aux_scores = []
        for k in range(num_aux):
            if k in block_set:
                aux_scores.append(100000)
                continue

            maps_without_k = [item for j, item in enumerate(aux_maps) if j != k]
            aux_scores.append(attempt_fn(radius, [pair[1] for pair in maps_without_k]) - current_objective)
        print(" ".join([f"{score:.2f}" for score in aux_scores]))
        i = np.argmin(aux_scores)
        cache = {}
        cache_hits = 0
        cache_misses = 0
        objective_before_loop = current_objective
        if current_objective < 50 or True:
            k = len(gates_with_progress)
        else:
            k = 30
        loop = tqdm(list(choices(gates_with_progress, k=k)), desc = f'bit {i}', leave=True)
        for score, gate in loop:
            if gate[0] in [pair[1][0] for pair in aux_maps]:
                continue
            # A guess about how to end the loop early: the idea is that a gate 
            # cannot possibly improve our situation more than it would improve the 
            # artificial variable score if used in isolation.


            # Try the new gate
            new_aux_maps = deepcopy(aux_maps)
            new_aux_maps[i] = (score, gate)
            new_objective = attempt_fn(radius, [pair[1] for pair in new_aux_maps])

            # report success
            if new_objective < current_objective:
                block_set = {i}
                num_since_last_success = 0
                current_objective = new_objective
                aux_maps = new_aux_maps
            else:
                num_since_last_success += 1
                block_set |= {i}
                if len(block_set) == len(aux_maps):
                    block_set = {i}
                if num_since_last_success > total_tasks:
                    print("stuck")
                    return

            current_improvement = objective_before_loop - current_objective
            loop.set_postfix(improve = current_improvement, curscore = new_objective, objective = current_objective)

            radius_updated = False
            while current_objective < 1:
                if radius == num_outputs:
                    print("done")
                    return

                radius_updated = True
                radius += 1
                current_objective = attempt_fn(radius, [pair[1] for pair in aux_maps])
                objective_before_loop = current_objective
                print(f'radius now {radius} objective {current_objective}')
                for m in aux_maps:
                    print(m)

            """
            if radius_updated:
                base_objective = attempt_fn(radius, None)
                progress = tqdm(gate_list, desc = 'scores', leave=True)
                reference_scores = [base_objective - attempt_fn(radius, [func]) for func in progress]
                gates_with_progress = sorted(zip(reference_scores, gate_list), key = lambda item: -item[0])
                print(f'Cache efficiency: {cache_hits/(cache_hits+cache_misses)}')
                break
            """


@click.command()
@click.option('--num_aux', default = 0, help = 'number of aux')
def main(num_aux):
    while True:
        run(num_aux)


if __name__ == '__main__':
    main()
