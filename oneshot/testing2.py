from mysolver_interface import call_my_solver, call_imul_solver
from lmisr_interface import call_solver
from fast_constraints import fast_constraints, CSC_constraints
from ising import IMul
from solver import LPWrapper

import numpy as np
from time import perf_counter as pc

import torch

n1, n2 = 2, 2

my_time = 0
imul_time = 0
glop_time = 0
lmisr_time = 0

for i in range(1000):

    aux_array = np.random.choice([-1,1], 16, p = [0.5, 0.5])
    aux_array = np.expand_dims(aux_array, axis=0)

    aux_tensor = torch.tensor(aux_array).clamp(min=0)
    
    start = pc()
    constraints, keys = CSC_constraints(n1, n2, aux_tensor, 2)
    objective = call_my_solver(constraints)
    end = pc()
    my_time += end-start
    myanswer = objective > 1e-2

    print(constraints)
    
    
    start = pc()
    objective = call_imul_solver(n1, n2, aux_tensor.numpy().astype(np.int8))
    end = pc()
    imul_time += end-start
    imulanswer = objective > 1e-2

    exit()

    start = pc()
    circuit = IMul(n1, n2)
    circuit.set_all_aux(aux_array)
    constraints, keys = fast_constraints(circuit, 2)
    glop = LPWrapper(keys)
    glop.add_constraints(constraints)
    result = glop.solve()
    end = pc()
    glop_time += end-start

    aux_array[aux_array == -1] = 0
    start = pc()
    lmisranswer = call_solver(2, 2, aux_array)
    end = pc()
    lmisr_time += end-start

    glopanswer = result is None
    if myanswer != glopanswer or imulanswer != glopanswer:
        print(f'{objective} {imulanswer} {myanswer} {glopanswer} {not lmisranswer}')

print(f'my_time = {my_time} imul = {imul_time} glop_time = {glop_time} lmisr_time = {lmisr_time}')


