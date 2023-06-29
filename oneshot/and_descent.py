from new_constraints import constraints
from mysolver_interface import call_my_solver
from itertools import product
from random import choice, choices
from copy import deepcopy

n1, n2 = 4,4
desired = (0,1,2,3,4,5,6,)
included = (7,)
num_aux = 12
num_inputs = n1 + n2 + (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1 + n2)
num_vars = num_inputs + num_outputs
radius = 1

and_gate_list = list(product(range(num_inputs), range(num_inputs, num_vars)))
current_gates = choices(and_gate_list, k=num_aux)
current_objective = 1000000
total_tasks = len(and_gate_list) * num_aux
num_since_last_success = 0

cache = {}

while True:
    for i in range(num_aux):
        for gate in and_gate_list:#choices(and_gate_list, k=20):
            new_gate_list = deepcopy(current_gates)
            new_gate_list[i] = gate
            key = hash(tuple(new_gate_list))
            if key in cache:
                objective = cache[key]
            else:
                M, _, _ = constraints(n1, n2, radius = radius, desired = desired, included = included, function_ands = new_gate_list)
                objective = call_my_solver(M.to_sparse_csc())
                cache[key] = objective
            if objective < current_objective:
                num_since_last_success = 0
                current_objective = objective
                current_gates = new_gate_list
                print(f'{i} {objective}')
                while current_objective < 1e-2:
                    if radius < num_outputs:
                        radius += 1
                        cache = {}
                        M, _, _ = constraints(n1, n2, radius = radius, desired = desired, included = included, function_ands = new_gate_list)
                        current_objective = call_my_solver(M.to_sparse_csc())
                        print(f'radius now {radius} objective {current_objective}')
                    else:
                        print(current_gates)
                        exit()
            else:
                num_since_last_success += 1
                if num_since_last_success > total_tasks:
                    print("stuck")
                    exit()


