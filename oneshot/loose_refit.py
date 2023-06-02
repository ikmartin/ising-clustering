from polyfit import search_polynomial, gen_var_keys, tryall, fit
from spinspace import Spin, Spinspace
from ising import IMul, PICircuit
from oneshot import reduce_poly, MLPoly, single_positive_FGBZ, single_negative_FGBZ, single_rosenberg, full_Rosenberg
from more_itertools import powerset
from fast_constraints import fast_constraints
from solver import build_solver
import numpy as np

from fittertry import IMulBit, coeff_heuristic

def coeff_heuristic(dat):
    C = dat[0]
    H = dat[1]
    
    return sum([
        abs(value)
        for key, value in H
    ])

def add_aux_by_map(aux_array, aux_map, circuit):
    aux_array.append([
        aux_map(tuple(circuit.inout(inspin).binary())) for inspin in circuit.inspace
    ])

n1, n2 = 3,3
circuit = IMul(n1,n2)

print(f'IMul{n1}x{n2}')
poly = fit(circuit,6)
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
    poly, aux_map = single_rosenberg(poly, heuristic = coeff_heuristic)
    if aux_map is None:
        break
    print(poly)
    print("")
    add_aux_by_map(aux_array, aux_map, circuit)
    circuit.set_all_aux(aux_array)
    print(np.array(circuit.get_aux_array()))
    
    poly2 = fit(circuit, 2)
    if poly2 is not None:
        print(poly2)
        break


    
