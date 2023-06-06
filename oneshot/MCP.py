import torch

"""
Nocedal and Wright consider a standard LP problem to be of the form

min <c, x> ; Ax = b, x >= 0

The problem that we want to solve is of the following form:

min <c, r> ; Mt + r >= v, r >= 0

This implies that there is no equality constraints. 

max     [0  -1] * [t    r] 
s.t. 
[-B     -I] [t] +   [s1]    = [-1]
[0      -I] [r]     [s2]      [ 0]
s1, s2 >= 0

i.e. -Bt - r + s1 = -1 and -r + s2 = 0
=> s2 = r, s1 = Bt + r - 1

=> r >= 0 and Bt + r - 1 >= 0 
=> r >= 0 and Bt + r >= 1

Therefore, let 
b := [0     -1]
c := [-1     0]
A := [M^t    0]
     [-I    -I]




"""

def r_c(A, x, b):
    return A @ x - b

def r_b(A, l, s, c):
    return torch.t(A) @ l + s - c




