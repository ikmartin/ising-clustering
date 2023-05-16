from constraints import get_constraint_matrix
from ising import IMul

import click
import torch
import torch.nn as nn

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

    for i in range(1000):
        MTalpha, dual_loss = projector()

        optimizer.zero_grad()
        dual_loss.backward()
        optimizer.step()

        projector.alpha.data.clamp_(min = 0)

        y = 0.5 * MTalpha + x
        res = constraint_matrix @ y - torch.ones(constraint_matrix.shape[0]).to(device)
        print(torch.sum(res < 0))


if __name__ == '__main__':
    main()
