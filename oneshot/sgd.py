import torch, click
import torch.nn as nn
from torch.optim import SGD
from ising import IMul
from spinspace import Spinspace, Spin
from tqdm import tqdm
from itertools import combinations
from typing import Iterable, Callable
from math import comb as choose
    
def tensor_power(tensor, power):
    """
    Computes all products of (power) distinct variables in the input tensor. Returns answer in tensor format.
    """
    
    return torch.tensor([
        torch.prod(torch.tensor(term))
        for term in combinations(tensor, power)   
    ])

def lse(domain: Iterable, function: Callable):
    """
    Torch implementation of the log-sum-exp function
    """
    return torch.log(sum([torch.exp(function(element)) for element in domain]))

class Hamiltonian(nn.Module):
    def __init__(self, degree: int, dimension: int) -> None:
        super(Hamiltonian, self).__init__()

        assert degree <= dimension

        self.degree = degree
        self.dimension = dimension

        # Create initial sets of coefficients using an iid Gaussian distribution. We keep the coefficients for terms of each degree seperate mostly in order to keep the code organized.
        self.coeff_vectors = [
            (
                deg, 
                nn.Parameter(torch.randn(choose(dimension, deg))
            )
            for deg in range(1, degree+1)
        ]

    def forward(self, input):
        """
        Evaluates the Hamiltonian polynomial on the given choice of the variables. The outer products are, for now, computed at runtime, but if we need more speed later we could trade this off for memory by precomputing the outer products for every spin state.
        """

        assert len(input) == self.dimension

        return sum([
            tensor_power(input, deg) @ coeffs
            for deg, coeffs in self.coeff_vectors
        ])
        
class BoltzmannModel(nn.Module):
    def __init__(self, H: Hamiltonian, circuit: PICircuit, spinspace: Spinspace, skew: float = 10, beta: float = 1):
        assert skew > 0 and beta > 0

        self.H = H
        self.circuit = circuit
        self.spinspace = spinspace
        self.skew = skew # named lambda in the math writeup
        self.beta = beta

    def _inputlevel(self, input):
        for 


    def _wrongspace(self, input):
        """
        Returns an iterator for W_sigma, i.e. the set of all wrong answers. 
        """
        

    def forward(self, input):
        energy = lambda state: -self.beta * self.H(state)
        return torch.exp(self.skew * (
            lse(self._wrongspace(input), energy) - lse(self._inputlevel(input), energy)
        ))
        

@click.command()
@click.option('--n1', default=2, help = 'Length in bits of first input')
@click.option('--n2', default=2, help = 'Length in bits of second input')
@click.option('--aux', default=0, help = 'Number of auxilliary spins')
@click.option('--degree', default=2, help = 'Degree of the Hamiltonian polynomial to fit.')
def main(n1: int, n2: int, aux: int, degree: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print(f'Warning: Running without CUDA support! Current device is {device}')

    circuit = IMul(n1, n2)


if __name__ == '__main__':
    main()
