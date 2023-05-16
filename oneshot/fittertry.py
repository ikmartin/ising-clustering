from polyfit import search_polynomial
from spinspace import Spin, Spinspace
from ising import IMul, PICircuit
from oneshot import reduce_poly

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

n1 = 4
n2 = 4
for i in range(0,n1+n2):    
    circuit = IMulBit(n1,n2,i)
    aux_nums = []
    for j in range(100):
        poly = search_polynomial(circuit)
        # print(f'output {i}: {poly}')
        if poly is not None:
            aux_nums.append(poly.num_variables() - n1 - n2 - 1)

    print(f'bit {i} min {min(aux_nums)}')

circuit = IMul(n1, n2)
poly = search_polynomial(circuit)
print(f'full num aux {poly.num_variables() - n1 - n2 - n1 - n2}')
