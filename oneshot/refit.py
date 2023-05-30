from polyfit import search_polynomial, gen_var_keys, tryall
from spinspace import Spin, Spinspace
from ising import IMul, PICircuit
from oneshot import reduce_poly, MLPoly, single_positive_FGBZ, single_negative_FGBZ, single_rosenberg, full_Rosenberg
from more_itertools import powerset
import numpy as np

from fittertry import IMulBit

def coeff_heuristic(dat):
    C = dat[0]
    H = dat[1]
    
    return 0 if not len(H) else max([
               abs(value)
               for key, value in H
               ])
    """
    return sum([
        abs(value)
        for key, value in H
    ])
    """

def fast_poly_search(circuit):
    for degree in range(2, circuit.G+1):
        poly = sparse_solve(circuit, degree)
        if poly is not None:
            return poly

    return None

n1 = 3
n2 = 3
for i in range(0,1):    
    #circuit = IMulBit(n1,n2,i)
    circuit = IMulBit(4,4,2)
    # tryall(circuit)
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
        poly = search_polynomial(circuit)
        print(poly)
        print("")
        print(np.array(circuit.get_aux_array()))
    

# 2x2x1
# 2x3x2
# 2x4x4

# 3x3x6
