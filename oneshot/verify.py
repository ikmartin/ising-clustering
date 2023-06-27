from lmisr_interface import call_solver
from fast_constraints import constraints_building
import numpy as np
from solver import LPWrapper
import torch


def aux_array_as_hex(aux_array):
    return "\n".join([
        "".join([
            format(int("".join([str(x) for x in row[i:(i+4)]]),2), 'x')
            for i in range(0,len(row),4)
        ])
        for row in aux_array
    ])#.replace("0", '_')

def parseaux(lines):
    array = None
    for line in lines:
        binary_string = ''.join([f'{int(c,16):04b}' for c in line])
        print(binary_string)
        numpy_line = np.array([[int(x) for x in binary_string]])
        array = numpy_line if array is None else np.concatenate([array,numpy_line])

    return array


def verify():
    np.set_printoptions(threshold = 50000000)
    n1 = 4
    n2 = 4
    # working 3x4
    #aux_keys = [(0,9), (5,12), (1,13), (1,11), (0,11), (4,10), (3,10)]

    # working 4x4x13
    #aux_keys = [(2, 14), (6, 14), (5, 8), (1, 13), (5, 13), (3, 6), (0, 9), (0, 10), (4, 11), (1, 12), (0, 12), (2, 12), (5, 11)]

    #aux_keys = [(1, 13), (2, 14), (1, 9), (5, 13), (0, 1), (4, 11), (2, 10), (0, 11), (5, 12), (4, 12)]
    #aux_keys = [(1,2), (4,10), (0,5), (0,8), (3,9), (0,9)]
    aux_keys = [(0, 1), (4, 11), (0, 10), (2, 13), (2, 14), (5, 13), (5, 12), (4, 12)]
    _, _, correct = constraints_building(n1,n2, None, 1, radius=None, mask=None, include=None)
    correct = correct.numpy()

    aux_vecs = np.concatenate([np.expand_dims(np.prod(correct[...,key], axis=-1), axis=0) for key in aux_keys]).astype(np.int8)
    print(aux_array_as_hex(aux_vecs))

    with open("saved-aux", "w") as FILE:
        FILE.write(str(aux_vecs.tolist()))

    objective = call_solver(n1,n2, aux_vecs)
    print(objective)

#verify()

"""
print("making constraints")
constraints, keys, _ = constraints_building(n1,n2, torch.tensor(aux_vecs), 2, radius = None, mask = None, include = None)
print("making solver")
solver = LPWrapper(keys)
constraints = constraints.to_sparse()
print("loading constraints")
solver.add_constraints(constraints)
print("solving")
solution = solver.solve()
print(solution)
"""

