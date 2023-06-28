from new_constraints import constraints
from mysolver_interface import call_my_solver
from lmisr_interface import call_solver
import numpy as np
import torch.nn.functional as F
import torch
from verify import aux_array_as_hex, parseaux
from tqdm import tqdm

np.set_printoptions(precision = 2, suppress = True)


# an interesting 3x3 aux consisting only of input and aux-based hyperplanes
#["ffff0003ffffffff", "10007000f550f770", "ecffccccccffccc"]

#aux_hex = ["1033ff675f3fff7f", "0f0b2f02cf8bff2f", "ffffffefcd8b5544"]
aux_hex = None
aux_hex = ["3333ffff2200bbbb3333ffff2220bbbb",
 "0000000000bb57770000000000035577",
 "3fffdfff0eff0fff0fff0fff00ff01ff"]

if aux_hex is not None:
    aux_array = parseaux(aux_hex)
    aux = torch.tensor(aux_array)
else:
    aux = None

n1, n2 = 3,4
A = len(aux_hex) if aux_hex is not None else 0
desired = (0,1,2,3,4,5)
included = (6,)
num_inputs = n1 + n2 + A +  (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1 + n2)
num_vars = num_inputs + num_outputs
original_vars = num_vars
hyperplanes = []
radius = None
search_iterations = 50
num_samples = 30
search_sigma = .2
cache = {}
for i in range(12):
    candidates = []
    loop = tqdm(range(search_iterations), leave = True)
    current_weight = torch.randn(num_vars)
    current_weight[num_inputs:original_vars] = 0
    current_weight = F.normalize(current_weight, dim=0)
    current_bias = torch.randn(1).clamp(0,1).item()
    current_plane = (current_weight, current_bias)
    for j in loop:
        search_sigma = np.exp(-3 * j/search_iterations)
        samples = []
        for k in range(num_samples):
            new_weight = search_sigma * torch.randn(num_vars) + current_weight
            new_weight[num_inputs:original_vars] = 0
            new_weight = F.normalize(new_weight, dim=0)
            #print(new_weight)
            
            new_bias = (search_sigma * torch.randn(1) + current_bias).clamp(0,1).item()
            new_plane = (new_weight, new_bias)
            M, terms, correct = constraints(n1, n2, aux = aux, radius = radius, desired = desired, included = included, hyperplanes = hyperplanes + [new_plane], auxfix = True)
            M_hash = hash(M.numpy().data.tobytes())
            if M_hash in cache:
                objective = cache[M_hash]
            else:
                objective, coeffs, rhos = call_my_solver(M.to_sparse_csc(), tolerance = 1e-8, fullreturn = True)
                cache[M_hash] = objective
            if objective < 1e-2:
                print("done")
                aux_array = torch.t(correct[...,original_vars:]).numpy().astype(np.int8)
                if aux is not None:
                    aux_array = np.concatenate([aux_array, aux])
                print(aux_array_as_hex(aux_array))
                objective = call_solver(n1, n2, aux_array)
                print(objective)
                exit()
            
            candidates.append({'plane': new_plane, 'val': objective})
        best = min(candidates, key = lambda item: item['val'])
        current_plane = best['plane']
        current_weight, current_bias = current_plane
        loop.set_postfix(objective = best['val'])
    else:
        plane_dict = min(candidates, key = lambda item: item['val'])
        print(f'current min rho: {plane_dict["val"]}')
        if plane_dict["val"] < 100:
            radius = None
        hyperplanes.append(plane_dict['plane'])
        num_vars += 1
        continue
    break


