from polyfit import search_polynomial, gen_var_keys
from spinspace import Spin, Spinspace
from ising import IMul, PICircuit
from oneshot import reduce_poly, MLPoly

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

"""
n1 = 3
n2 = 3
for i in range(0,n1+n2):    
    #circuit = IMulBit(n1,n2,i)
    circuit = IMul(3,3)
    poly = search_polynomial(circuit)
    num_aux_l1 = poly.num_variables() - n1 - n2 - 1
    for degree in range(2, circuit.G): 
        coeffs_l0 = l0_fit(circuit, degree)
        if coeffs_l0 is None:
            continue
        keys = gen_var_keys(degree, circuit)
        coeff_dict = {
            key: val
            for key, val in zip(keys, coeffs_l0)
        }
        poly2 = MLPoly(coeff_dict)
        poly2.clean(threshold = 0.01)
        poly2 = reduce_poly(poly2, ['rosenberg'])
        num_aux_l0 = poly2.num_variables() - n1 - n2 - 1
        print(poly)
        print(poly2)
        print(f'l1: {num_aux_l1} l0: {num_aux_l0}')
        break

circuit = IMul(n1, n2)
poly = search_polynomial(circuit)
print(f'full num aux {poly.num_variables() - n1 - n2 - n1 - n2}')
"""
