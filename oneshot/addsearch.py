from add_constraints import constraints
from mysolver_interface import call_my_solver
from lmisr_interface import call_solver
import numpy as np
import torch.nn.functional as F
import torch
from verify import aux_array_as_hex, parseaux
from tqdm import tqdm
from itertools import combinations, product

np.set_printoptions(precision = 2, suppress = True)


# an interesting 3x3 aux consisting only of input and aux-based hyperplanes
#["ffff0003ffffffff", "10007000f550f770", "ecffccccccffccc"]

#aux_hex = ["1033ff675f3fff7f", "0f0b2f02cf8bff2f", "ffffffefcd8b5544"]
aux_hex = ["3333ffff2200bbbb3333ffff2220bbbb",
 "0000000000bb57770000000000035577",
 "3fffdfff0eff0fff0fff0fff00ff01ff"]
aux_hex = None

if aux_hex is not None:
    aux_array = parseaux(aux_hex)
    aux = torch.tensor(aux_array)
else:
    aux = None

n1, n2, n3 = 4,4,4
A = len(aux_hex) if aux_hex is not None else 0
desired = None
included = None
num_inputs = n1 + n2 + n3 + A +  (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1+2)
num_vars = num_inputs + num_outputs
original_vars = num_vars
hyperplanes = []
radius = None
search_iterations = 50
num_samples = 30
search_sigma = .2
cache = {}
M, terms, correct = constraints(n1, n2, n3, aux = aux, radius = radius, desired = desired, included = included, hyperplanes = None, auxfix = True)
M_hash = hash(M.numpy().data.tobytes())
if M_hash in cache:
    objective = cache[M_hash]
else:
    objective, coeffs, rhos, _, _ = call_my_solver(M.to_sparse_csc(), tolerance = 1e-8, fullreturn = True)
    cache[M_hash] = objective
print(f'initial obj {objective}')
for i in range(16):
    candidates = []
    #loop = tqdm(list(combinations(range(num_vars), 2)), leave = True)
    loop = tqdm(list(product(range(num_inputs), range(num_inputs, original_vars))), leave=True)
    current_weight = torch.randn(num_vars)
    current_weight[num_inputs:original_vars] = 0
    current_weight = F.normalize(current_weight, dim=0)
    current_bias = torch.randn(1).clamp(0,1).item()
    current_plane = (current_weight, current_bias)
    for x,y in loop:
        new_weight = torch.zeros(num_vars)
        new_weight[x] = 1
        new_weight[y] = 1

        new_plane = (new_weight, 0.5)
        M, terms, correct = constraints(n1, n2, n3, aux = aux, radius = radius, desired = desired, included = included, hyperplanes = hyperplanes + [new_plane], auxfix = True)
        M_hash = hash(M.numpy().data.tobytes())
        if M_hash in cache:
            objective = cache[M_hash]
        else:
            objective, coeffs, rhos, _, _ = call_my_solver(M.to_sparse_csc(), tolerance = 1e-8, fullreturn = True)
            cache[M_hash] = objective
        if objective < 1e-2:
            if radius is not None:
                break
            print("done")
            aux_array = torch.t(correct[...,original_vars:]).numpy().astype(np.int8)
            if aux is not None:
                aux_array = np.concatenate([aux_array, aux])
            print(aux_array_as_hex(aux_array))
            exit()
            objective = call_solver(n1, n2, aux_array)
            print(objective)
            exit()
            
        candidates.append({'plane': new_plane, 'val': objective, 'pair': (x,y)})
        best = min(candidates, key = lambda item: item['val'])
        current_plane = best['plane']
        current_weight, current_bias = current_plane
        loop.set_postfix(objective = best['val'])
    plane_dict = min(candidates, key = lambda item: item['val'])
    print(f'current min rho: {plane_dict["val"]}')
    print(f'selected {plane_dict["pair"]}')
    if plane_dict["val"] < 50:
        radius = None
    hyperplanes.append(plane_dict['plane'])
    num_vars += 1


