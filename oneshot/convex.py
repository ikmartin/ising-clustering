import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from ising import IMul
from spinspace import Spin
from oneshot import MLPoly, reduce_poly
from itertools import chain, combinations

DEGREE = 3
torch.set_printoptions(threshold=50000)

def tensor_power(tensor: torch.Tensor, power: int) -> torch.Tensor:
    """
    Computes all products of (power) distinct variables in the input tensor. Returns answer in tensor format.
    """
   
    return torch.prod(torch.combinations(tensor, r=power), dim=1)
    
def vspin(state):
    # Return concatenated tensor powers of said explicit binary
    return torch.cat([tensor_power(state, deg) for deg in range(1, DEGREE+1)])


def hypercube_single_constraint(inspin, outspin, i):
    other_out = outspin.clone()
    other_out[i] = 1-other_out[i]
    other = torch.cat([inspin, other_out])
    current = torch.cat([inspin, outspin])
    return vspin(other) - vspin(current)

def gen_hypercube_constraints(circuit, spin):
    inspin, outspin, _ = spin.split()
    inspin = Spin(inspin.asint(), shape = (circuit.N1, circuit.N2))
    correct_bin = torch.tensor(circuit.fout(inspin).binary())
    current_bin = torch.tensor(outspin.binary())
    in_bin = torch.tensor(inspin.binary())
    constraint_list = [
        hypercube_single_constraint(in_bin, current_bin, i).unsqueeze(0)
        for i, b in enumerate(current_bin)
        if b == correct_bin[i]
    ]
    if len(constraint_list):
        return torch.cat(constraint_list)
    
    return None

def make_poly(dim: int, coeffs: torch.Tensor) -> MLPoly:
    keys = list(chain.from_iterable([
        combinations(range(dim), deg)
        for deg in range(1, DEGREE+1)
    ]))
    coeff_dict = {
        key: val.item()
        for key, val in zip(keys, coeffs)
        if val
    }
    return MLPoly(coeff_dict)

circuit = IMul(2,2)
constraint_sets = [
    gen_hypercube_constraints(circuit, spin)
    for spin in circuit.spinspace
]
constraint_sets = list(filter(lambda x: x is not None, constraint_sets))
constraint_matrix = torch.cat(constraint_sets)
print(f'sparsity = {torch.sum(constraint_matrix == 0)/torch.numel(constraint_matrix)}')

h = cp.Variable(constraint_matrix.shape[1])
M = cp.Parameter(constraint_matrix.shape)
constraints = [ M @ h >= 1 ]
objective = cp.Minimize(cp.pnorm(h, p = 1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

layer = CvxpyLayer(problem, parameters = [M], variables = [h]).cuda()
constraint_matrix = constraint_matrix.cuda()
solution, = layer(constraint_matrix)
print(solution)

poly = make_poly(circuit.G, solution.cpu())
print(poly)

print(reduce_poly(poly, ['rosenberg']))
