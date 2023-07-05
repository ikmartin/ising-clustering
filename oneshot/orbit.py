from new_constraints import constraints
from mysolver_interface import call_my_solver
from verify import parseaux
from ast import literal_eval
import torch
from itertools import combinations

torch.set_printoptions(threshold = 50000)
orbits = []
with open('dat/orbit4.dat', 'r') as FILE:
    for line in FILE.readlines():
        orbits.append(parseaux(literal_eval(line)))

n1, n2 = 3,3
desired = (0,1,2,3,4)
included = (5,)
num_aux = 4
num_inputs = n1 + n2 + (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1 + n2)
num_vars = num_inputs + num_outputs

for indices in combinations(range(num_vars), 4):
    for orbit in orbits:
        results = []
        for func in orbit:
            M, _, _ = constraints(n1, n2, desired = desired, included = included, bool_funcs = [(indices, func)])

            objective, coeffs, rhos, _, _ = call_my_solver(M.to_sparse_csc(), fullreturn = True)
            results.append(torch.tensor(rhos).unsqueeze(0))

        vars = torch.var(torch.cat(results), dim=0)
        print(torch.any(vars > 1e-4))


