import numpy as np

def solve_system():
    print(f'x = {x}')
    print(f'rc = {rc}')
    print(f'L = {L}')
    print(f's = {s}')
    val1 = -(x + (x*rc + L)/s)
    print(f'val1 = {val1}')
    val2 = A @ val1
    print(f'val2 = {val2}')
    rhs = b + val2
    rhs2 = b - A @ (x + (x*rc + L)/s)
    print(f'rhs = {rhs}')
    print(f'rhs2 = {rhs2}')
    dl = np.linalg.solve(A @ np.diag(x/s) @ At, rhs)
    print(f'dl = {dl}')
    exit()


    ds = -rc - At @ dl
    dx = (L - x*ds)/s
    return dl, ds, dx

M = np.array([
    [1, 1],
    [-1, 0],
    [1, -1]
], dtype = np.float64)

from mysolver_interface import call_my_solver

print(M.dtype)
result = call_my_solver(M)
print(result)
exit()

m = M.shape[0]
n = M.shape[1]

At = np.block([
    [-M, -np.eye(3)],
    [np.zeros((3, 2)), -np.eye(3)]
])
A = At.T

b = np.concatenate([np.zeros(n), -np.ones(m)])
c = np.concatenate([-np.ones(m), np.zeros(m)])

# calculate initial guesses

x = At @ np.linalg.solve(A @ At, b)
l = np.linalg.solve(A @ At, A @ c)
s = c - At @ l

delta_x = max(-(3/2)*np.min(x), 0)
delta_s = max(-(3/2)*np.min(s), 0)
x = x + delta_x
s = s + delta_s
delta_x = (1/2) * (np.dot(x,s)) / sum(s)
delta_s = (1/2) * (np.dot(x,s)) / sum(x)
x = x + delta_x
s = s + delta_s

print(f'x = {x}')
print(f's = {s}')
print(f'l = {l}')

eta = 0.9
for i in range(200):
    rc = At @ l + s - c
    L = -x*s
    dl, ds, dx = solve_system()

    alpha_primal_list = [1]
    alpha_dual_list = [1]
    print(f'{dx}')
    for j in range(2*m):
        if dx[j] < 0:
            alpha_primal_list.append(-x[j]/dx[j])
        if ds[j] < 0:
            alpha_dual_list.append(-s[j]/ds[j])
    alpha_primal = min(alpha_primal_list)
    alpha_dual = min(alpha_dual_list)

    mu_aff = (1/(2*m)) * np.dot(x + alpha_primal * dx, s + alpha_dual * ds)
    mu = (1/(2*m)) * np.dot(x, s)
    sigma = np.power((mu_aff/mu), 3.0)

    L -= dx * ds - sigma*mu

    dl, ds, dx = solve_system()

    alpha_primal_list = [1]
    alpha_dual_list = [1]
    for j in range(2*m):
        if dx[j] < 0:
            alpha_primal_list.append(-eta * x[j]/dx[j])
        if ds[j] < 0:
            alpha_dual_list.append(-eta * s[j]/ds[j])

    alpha_primal = min(alpha_primal_list)
    alpha_dual = min(alpha_dual_list)

    x = x + alpha_primal * dx
    l = l + alpha_dual * dl
    s = s + alpha_dual * ds
    
    print(f'lambda = {l}')









