from constraints import get_constraint_matrix
from ising import IMul

import click
import numpy as np

from qpsolvers import solve_qp

def projector(circuit, degree):
    M = get_constraint_matrix(circuit, degree).numpy()
    P = np.eye(M.shape[1])
    G = -M
    h = -np.ones(G.shape[0])

    def solve(x):
        return solve_qp(P, -x, G, h, solver='osqp')

    return solve, M.shape[1]

def grad_f(x, sigma):
    dir = 2 * sigma * x / np.power(x * x + sigma, 2)
    return dir / np.linalg.norm(dir)

def l0_fit(circuit, degree):
    P_C, dimension = projector(circuit, degree)

    x = P_C(np.random.normal(size = (dimension,)))
    if x is None:
        return None

    eta = 1e-2
    gamma = 1-1e-3
    sigma = 10

    for k in range(1000):
        z = x - eta * grad_f(x, sigma)
        x = P_C(z)
        sigma *= gamma
        print(k)

    return x

@click.command()
@click.option('--degree', default = 2, help = "Degree of the Hamiltonian polynomial")
def main(degree: int):

    circuit = IMul(2, 3)

    
    #print(x)
    print(x.astype(int))
    print(sum(abs(x) < 0.01))
    

if __name__ == '__main__':
    main()
