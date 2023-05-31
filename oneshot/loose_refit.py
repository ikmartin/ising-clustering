from polyfit import search_polynomial, gen_var_keys, tryall
from spinspace import Spin, Spinspace
from ising import IMul, PICircuit
from oneshot import reduce_poly, MLPoly, single_positive_FGBZ, single_negative_FGBZ, single_rosenberg, full_Rosenberg
from more_itertools import powerset
from fast_constraints import fast_constraints
from solver import build_solver
import numpy as np

from fittertry import IMulBit

def coeff_heuristic(dat):
    C = dat[0]
    H = dat[1]
    
    return sum([
        abs(value)
        for key, value in H
    ])


n1, n2 = 3,4
circuit = IMul(n1,n2)

print(f'IMul{n1}x{n2}')
poly = search_polynomial(circuit)
regular_reduction = full_Rosenberg(poly, coeff_heuristic)
print('Original Polynomial')
print(poly)
print('')
print('Full Rosenberg')
print(regular_reduction)
print('')
print('Trying iterative reduction')
aux_array = []
while True:
    poly, aux_map = single_rosenberg(poly)
    if aux_map is None:
        break
    print(poly)
    print("")
    aux_array.append([
        aux_map(tuple(circuit.inout(inspin).binary())) for inspin in circuit.inspace
    ])
    circuit.set_all_aux(aux_array)
    print(np.array(circuit.get_aux_array()))
    
    M, keys = fast_constraints(circuit, 2)
    solver, variables, bans = build_solver(M, keys)
    status = solver.Solve()
    if status == 0:
        break
    
