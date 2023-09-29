from new_constraints import constraints  
from mysolver_interface import call_my_solver
from torch import tensor
from tqdm import tqdm
from itertools import combinations
from random import choices
import torch


n1, n2 = 4,4 
hyperplanes = []
with open(f"dat/WNTF{n1}{n2}.dat", "r") as FILE:
    for line in FILE.readlines():
        w, b = eval(line)

        hyperplanes.append((tensor(w), b))
    """
    line_accumulator = ""
    for line in FILE.readlines():
        if line.startswith("tensor") and len(line_accumulator) > 0:
            hyperplanes.append((eval(line_accumulator),0))
            line_accumulator = ""

        line_accumulator += line
    """

num_base_vars = (n1+n2)*2
for indices in combinations(range(num_base_vars), 3):
    t = torch.zeros(num_base_vars)
    for j in indices:
        t[j] = 1
    hyperplanes.append((t, 0.5))

print(len(hyperplanes))
current_planes = []

loop = tqdm(range(1000), leave=True)
with open('dat/44rhodistr.dat', 'a') as FILE:
    for i in loop:
        cur_planes = choices(hyperplanes, k=12)
        ret = []
        for r in range(1, n1+n2+1):
            M, _, correct = constraints(n1, n2, radius=r, hyperplanes = cur_planes, basin_report = True)
            obj = call_my_solver(M.to_sparse_csc())
            print(f'r = {r} obj = {obj}')
            ret.append(obj)

        FILE.write(str(ret) + "\n")
        FILE.flush()

