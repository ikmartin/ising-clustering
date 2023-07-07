from lmisr_interface import call_solver
from mysolver_interface import call_my_solver
from fast_constraints import constraints_building
from new_constraints import constraints
import numpy as np
from numpy import array
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
        numpy_line = np.array([[int(x) for x in binary_string]])
        array = numpy_line if array is None else np.concatenate([array,numpy_line])

    return array


def verify():
    np.set_printoptions(threshold = 50000000)
    n1 = 3
    n2 = 4
    """
    aux_hex = ["000000f300000020003032fa002000c0",
    "f511fd51f751ff3df575fffefff7fffb",
    "f000200011110000f3f32120bb932203",
    "00000000000001000000000002001100",
    "00000040000000002020808000000000",
    "3333ffff2200bbbb3333ffff2220bbbb",
    "0000000000bb57770000000000035577",
    "3fffdfff0eff0fff0fff0fff00ff01ff",]
    """
    aux_hex = ["000000000000000000000000fb00ff00",
    "0000800000008000c000f800e000fc00",
    "11115555000011115555555500001111",
    "30000000333031303000000032303030",
    "8bffffff0000ffff0000ffff0000dfff",
    "af8f0f0fbf2f2f0f0e080000040e0008",
    "737fffff7373fffff3f7ffff7ff7ffff",
    "a0800000ffff2222aaa80000ffffa2a0"]
    aux_hex = ["2000ffffffffffffffffffffffffffff",
    "002000000000002000f300fb00ff00ff",
    "000000000000000000000000f010f000",
    "00000000dddd44440000000044444444",
    "00000000cccc000088f388aaccfec4ee",
    "0c0cdf5f0000040400005f5f00000000",
    "3fffff7f33333f7f7fff7f7f3bbf3b7f",
    "00007000fcccffff8000f8a0fffeffff"]
    aux_hex = ["00ff00ff00ff03ff0fff0eff1cff19ff",
    "33333333777777773333333377777777",
    "00000f0f33332d2dffffffffffffffff",
    "000000ff0f0f1ce3ffffffffffffffff",
    "000000ffffffffff333336c9ffffffff",
    "0f0f5f5f0f0f5f5f0f0f5f5f0f0f5f5f",
    "00000f0fffffffff55555a5affffffff"]
    aux_array = parseaux(aux_hex)
    objective = call_solver(n1, n2, aux_array)
    print(objective)

def parse_ands():
    # working 3x4
    #aux_keys = [(0,9), (5,12), (1,13), (1,11), (0,11), (4,10), (3,10)]

    # working 4x4x13
    #aux_keys = [(2, 14), (6, 14), (5, 8), (1, 13), (5, 13), (3, 6), (0, 9), (0, 10), (4, 11), (1, 12), (0, 12), (2, 12), (5, 11)]

    #aux_keys = [(1, 13), (2, 14), (1, 9), (5, 13), (0, 1), (4, 11), (2, 10), (0, 11), (5, 12), (4, 12)]
    #aux_keys = [(1,2), (4,10), (0,5), (0,8), (3,9), (0,9)]
    #aux_keys = [(0, 1), (4, 11), (0, 10), (2, 13), (2, 14), (5, 13), (5, 12), (4, 12)]
    #aux_keys = [(5, 13), (1, 13), (6, 12), (6, 11), (4, 12), (6, 14), (4, 10), (3, 13), (5, 11), (4, 9), (3, 14), (5, 12)]
    n1 = 4
    n2 = 4


    functions = [
        ((0, 4, 11), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((4, 5, 10), array([1, 1, 1, 1, 1, 0, 1, 0])), 
        ((2, 6, 11), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((2, 7, 13), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((1, 4, 12), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((0, 1, 12), array([1, 1, 1, 1, 1, 0, 1, 0])), 
        ((0, 5, 11), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((6, 8, 11), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((3, 15, 13), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((4, 6, 9), array([1, 0, 0, 0, 1, 1, 1, 0]))
    ]
    """    
    functions = [
        ((0, 4, 11), 	array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((1, 9, 10), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((3, 6, 13), 	array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((4, 5, 11), 	array([1, 1, 1, 1, 1, 0, 1, 0])), 
        ((6, 9, 12), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((0, 9, 12), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((0, 5, 12), 	array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((1, 12, 13), array([1, 0, 0, 0, 1, 1, 1, 0])), 
        ((0, 2, 13), 	array([1, 0, 1, 1, 1, 0, 1, 1])), 
        ((2, 4, 10), 	array([1, 0, 0, 0, 1, 1, 1, 0]))
    ]
    """

    functions = [
            ((0, 4, 12), array([1, 0, 0, 0, 1, 1, 1, 0])), 
            ((4, 5, 11), array([1, 1, 1, 1, 1, 0, 1, 0])), 
            ((2, 6, 12), array([1, 0, 0, 0, 1, 1, 1, 0])), 
            ((2, 7, 14), array([1, 0, 0, 0, 1, 1, 1, 0])), 
            ((1, 4, 13), array([1, 0, 0, 0, 1, 1, 1, 0])), 
            ((0, 1, 13), array([1, 1, 1, 1, 1, 0, 1, 0])), 
            ((0, 5, 12), array([1, 0, 0, 0, 1, 1, 1, 0])), 
            ((6, 9, 12), array([1, 0, 0, 0, 1, 1, 1, 0])), 
            ((3, 8, 14), array([1, 0, 0, 0, 1, 1, 1, 0])), 
            ((4, 6, 10), array([1, 0, 0, 0, 1, 1, 1, 0]))
    ]
    M, _, correct = constraints(n1,n2, degree=2, radius=None, included=(7,), desired=(0,1,2,3,4,5,6), bool_funcs = functions)
    correct = correct[...,(2*(n1+n2)):].numpy()
    print(aux_array_as_hex(correct.T))
    aux_vecs = correct.T

    with open("saved-aux2", "w") as FILE:
        #FILE.write(str(aux_vecs.tolist()))
        aux_vecs = aux_vecs.T
        for line in aux_vecs:
            FILE.write(' '.join([str(int((2*x)-1)) for x in line]) + '\n')

    objective = call_my_solver(M.to_sparse_csc())
    print(objective)

if __name__ == '__main__':
    parse_ands()
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

