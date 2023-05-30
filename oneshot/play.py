from ising import IMul
from fast_constraints import fast_constraints, fast_constraints2
from time import perf_counter as pc

x = [0,0,0, 0, 0, 0, 1, 1]
y = [0,0,0, 0, 0, 0, 1, 1]

k = 3 
Nk = 2 ** k

def xor(x, y):
    return [a ^ b for a, b in zip(x,y)]

def dyadic_convolution(x, y):
    return [
        (1 / Nk) * sum([
            x[d] * y[n - d]
            for d in range(Nk)
        ])
        for n in range(Nk)
    ]


