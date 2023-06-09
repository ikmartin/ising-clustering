from mysolver_interface import call_my_solver
from lmisr_interface import call_solver
from fast_constraints import fast_constraints
from ising import IMul
from solver import LPWrapper

import numpy as np
from time import perf_counter as pc


circuit = IMul(2,2)

my_time = 0
glop_time = 0
lmisr_time = 0

for i in range(1000):

    aux_array = np.random.choice([-1,1], 16, p = [0.5, 0.5])
    aux_array = np.expand_dims(aux_array, axis=0)
    circuit.set_all_aux(aux_array)
    constraints, keys = fast_constraints(circuit, 2)
    d_constraints = constraints.to_dense().numpy().astype(float)

    start = pc()
    objective = call_my_solver(d_constraints)
    end = pc()
    my_time += end-start

    start = pc()
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

    myanswer = objective > 1e-2
    glopanswer = result is None
    if myanswer != glopanswer:
        print(f'{objective} {myanswer} {glopanswer} {not lmisranswer}')

print(f'my_time = {my_time} glop_time = {glop_time} lmisr_time = {lmisr_time}')


