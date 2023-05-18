from constraints import get_constraint_matrix
from ising import IMul

import click
import torch
import torch.nn as nn
import numpy as np

from qpsolvers import solve_qp


import time

class DualProjection(nn.Module):
    def __init__(self, M):
        super(DualProjection, self).__init__()
        self.M = M
        self.n, self.m = M.shape
        self.alpha = nn.Parameter(torch.zeros(self.n))
        self.reset_alpha()

    def set_x(self, x):
        self.linear_factor = self.M @ x - torch.ones(self.n).to(self.M.device)

    def reset_alpha(self):
        self.alpha.data = torch.randn(self.n)

    def nn_orthant(self):
        self.alpha.data.clamp_(min=0)

    def forward(self):
        MTalpha = torch.t(self.M) @ self.alpha
        loss = torch.sum(MTalpha * MTalpha) + 4 * torch.dot(self.linear_factor, self.alpha)
        return MTalpha, loss


@click.command()
@click.option('--degree', default = 2, help = "Degree of the Hamiltonian polynomial")
def main(degree: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    circuit = IMul(2, 2)
    constraint_matrix = get_constraint_matrix(circuit, degree).float().to(device)

    
    M = get_constraint_matrix(circuit, degree).numpy()
    x = np.zeros(M.shape[1])
    q = -1 * x # -4 * x
    P = np.eye(len(q))
    G = -M
    h = -np.ones(G.shape[0])
    
    start = time.perf_counter()

    solution = solve_qp(P, q, G, h, solver='osqp')
    
    end = time.perf_counter()
    print(end-start)
    
    P = M @ M.transpose()
    q = 2 * (M @ x + h)
    lb = np.zeros(M.shape[0])

    
    start = time.perf_counter()

    solution = solve_qp(P, q, lb = lb, solver = 'osqp')
    y = x + 0.5 * (M.T @ solution)

    end = time.perf_counter()
    print(end-start)

    check_vec = M @ (x + 0.5 * (M.T @ solution)) + h

    exit()



    projector = DualProjection(constraint_matrix).to(device)

    # Optimizer for the outer gradient descent, i.e. descending along the gradients of the approximate L0 norm.
    optimizer = torch.optim.SGD(
        projector.parameters(),
        lr = 1e-4,
        weight_decay = 0
    )

    # test x
    x = torch.zeros(constraint_matrix.shape[1]).float().to(device)
    projector.set_x(x)

    for i in range(1000000):
        MTalpha, dual_loss = projector()

        optimizer.zero_grad()
        dual_loss.backward()
        optimizer.step()

        projector.nn_orthant()

        y = 0.5 * MTalpha + x
        res = constraint_matrix @ y - torch.ones(constraint_matrix.shape[0]).to(device)
        print(res)
        print(sum(res < 0))


if __name__ == '__main__':
    main()
