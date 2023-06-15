from mysolver_interface import call_my_solver
from ising import IMul
from fittertry import IMulBit
from fast_constraints import fast_constraints
from guided_lmisr import CircuitFactory

import numpy as np

def crc_remainder(input_bitstring, polynomial_bitstring, initial_filler):
    """Calculate the CRC remainder of a string of bits using a chosen polynomial.
    initial_filler should be '1' or '0'.
    """
    polynomial_bitstring = polynomial_bitstring.lstrip('0')
    len_input = len(input_bitstring)
    initial_padding = (len(polynomial_bitstring) - 1) * initial_filler
    input_padded_array = list(input_bitstring + initial_padding)
    while '1' in input_padded_array[:len_input]:
        cur_shift = input_padded_array.index('1')
        for i in range(len(polynomial_bitstring)):
            input_padded_array[cur_shift + i] \
            = str(int(polynomial_bitstring[i] != input_padded_array[cur_shift + i]))
    return ''.join(input_padded_array)[len_input:]


def check(factory, generator):
    poly = format(generator, 'b')
    circuit = factory.get()
    aux_lines = []
    print(f'generator {poly}')
    for inspin in circuit.inspace:
        store_pattern = circuit.inout(inspin).binary()
        store_string = "".join([str(b) for b in store_pattern])
        aux_string = crc_remainder(store_string, poly, "0")
        aux_lines.append([int(b) for b in aux_string])

    aux_array = np.array(aux_lines).T
    print(f'Checking {aux_array}')

    circuit = factory.get(aux_array)
    constraints, keys = fast_constraints(circuit, 2)
    constraints = constraints.to_sparse_csc()
    objective = call_my_solver(constraints, tolerance = 1e-8)
    if objective > 0.1:
        print("Infeasible")
        return 0

    print("Good!")
    return 1

factory = CircuitFactory(IMulBit, args = (3,3,1))
num_good = 0
for i in range(8, 1 << 8):
    num_good += check(factory, i)

print(num_good)



