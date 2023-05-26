from polyfit import search_polynomial, gen_var_keys, tryall
from spinspace import Spin, Spinspace
from ising import IMul, PICircuit
from oneshot import reduce_poly, MLPoly, single_positive_FGBZ, single_negative_FGBZ, single_rosenberg, full_Rosenberg
from more_itertools import powerset
import numpy as np


class IMulBit(PICircuit):
    def __init__(self, N1, N2, bit):
        super().__init__(N=N1+N2, M=1, A=0)
        self.inspace = Spinspace(shape=(N1,N2))
        self.N1 = N1
        self.N2 = N2
        self.bit = bit

    def fout(self, inspin: Spin):
        num1, num2 = inspin.splitint()
        result = Spin(spin = num1 * num2, shape = (self.N,))
        result_spin = result.spin()[self.bit]
        result_int = 0 if result_spin == -1 else 1
        return Spin(spin = result_int, shape = (self.M,))

class IMulSame(PICircuit):
    def __init__(self, N):
        super().__init__(N=2*N, M=N+1, A=0)
        self.inspace = Spinspace(shape=(N,N))
        self.N1 = N
        self.N2 = N

    def fout(self, inspin: Spin):
        num1, num2 = inspin.splitint()
        result = Spin(spin = num1 * num2, shape = (self.N,))
        result_spin = result.spin()[:self.M]
        return Spin(spin = result_spin, shape = (self.M,))

def find_patterns(circuit):
    states = []
    for spin in circuit.inspace:
        states.append(circuit.inout(spin).binary().tolist())
    states = states.T
    print(states)
    N = states.shape[0] - 1
    for combo in powerset(range(N)):
        if len(combo) <= 1:
            continue
        combo = combo + (N,)
        products = np.prod(states[np.array(combo)], axis=0)
        if np.all(products == 0):
            print(f'deduc {combo} = 0')
        if np.all(products == 1):
            print(f'deduc {combo} = 1')
        



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
    circuit = IMul(3,3)
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
