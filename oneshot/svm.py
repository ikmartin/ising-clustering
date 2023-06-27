from new_constraints import constraints
from mysolver_interface import call_my_solver
import numpy as np
import torch.nn.functional as F
import torch
from verify import aux_array_as_hex, parseaux
from tqdm import tqdm

np.set_printoptions(precision = 2, suppress = True)


#aux_hex = ["1033ff675f3fff7f", "0f0b2f02cf8bff2f", "ffffffefcd8b5544"]
aux_hex = ["55000000ffaa991155000000ffaa991100000000ea00110100000000ea281111",
"3373777751557777737777775555777710007551101154551013717710155475",
"8e0c8f000f000e008f08cf050f0d4e0cef0f8f0f0f0f8f0eef0f8f0f0f0f8f0e",
"e8ffa8faecffaaeafffffefffffffaeef0f0a0a0e0c7a08afeffeabffeffeafe",
"cccc0000cccc80808888008088880080cecc0080cccc80808888008088880080",
"2322222200003320230222031801333200002000000000000000212002000220",
"8800a000efc8aa028e08aa20fffdee86ea00a220efc5aa02eeaaaa22ffffef8e",
"0d040f07ffff2f205f005e017f4f781c40000001f747050f451d421bee3f4474",
"cccc5d5dffffddddddddddddffeefddd44404005f547450fdddd5c5feefffdfd"]
#aux_hex = None
if aux_hex is not None:
    aux_array = parseaux(aux_hex)
    aux = torch.tensor(aux_array)
else:
    aux = None

n1, n2 = 4,4
A = len(aux_hex) if aux_hex is not None else 0
desired = (3,4)
included = (0,1,2,5,6,7,)
num_vars = n1+n2 + A + (len(desired) if desired is not None else n1 + n2) + (len(included) if included is not None else 0)
original_vars = num_vars
hyperplanes = []
radius = None
search_iterations = 500
num_samples = 1
search_sigma = .5
cache = {}
for i in range(10):
    candidates = []
    loop = tqdm(range(search_iterations), leave = True)
    current_weight = F.normalize(torch.randn(num_vars), dim=0)
    current_bias = torch.randn(1).item()
    current_plane = (current_weight, current_bias)
    for j in loop:
        samples = []
        for k in range(num_samples):
            new_weight = F.normalize(search_sigma * torch.randn(num_vars) + current_weight, dim=0)
            new_bias = search_sigma * torch.randn(1).item() + current_bias
            new_plane = (new_weight, new_bias)
            M, terms, correct = constraints(n1, n2, aux = aux, radius = radius, desired = desired, included = included, hyperplanes = hyperplanes + [new_plane], auxfix = True)
            M_hash = hash(M.numpy().data.tobytes())
            if M_hash in cache:
                objective = cache[M_hash]
            else:
                objective, coeffs, rhos = call_my_solver(M.to_sparse_csc(), tolerance = 1e-6, fullreturn = True)
                cache[M_hash] = objective
            if objective < 1e-2:
                print("done")
                print(aux_array_as_hex(torch.t(correct[...,original_vars:]).numpy().astype(np.int8)))
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


