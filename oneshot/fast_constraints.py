from spinspace import Spin
from itertools import chain, combinations
import torch

#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits-1, -1, -1).to(x.device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def all_answers(circuit):
    return dec2bin(torch.arange(2 ** circuit.G).to(DEVICE), circuit.G)

def batch_tensor_power(input, power):
    num_rows, num_cols = input.shape
    index = torch.combinations(torch.arange(num_cols).to(DEVICE), r=power).unsqueeze(0).expand(num_rows, -1, -1)
    input = input.unsqueeze(-1).expand(-1, -1, power)
    return torch.gather(input, 1, index).prod(dim=-1)

def batch_vspin(input, degree):
    return torch.cat(
        [input] + [
            batch_tensor_power(input, i)
            for i in range(2, degree+1)
        ],
        dim = 1
    )

def keys(circuit, degree):
    return list(chain.from_iterable([
        combinations(range(circuit.G), i)
        for i in range(1, degree + 1)
    ]))

def fast_constraints(circuit, degree):
    all_states = all_answers(circuit)

    correct_rows = torch.cat([
        torch.tensor(
            circuit.inout(inspin).binary()
        ).unsqueeze(0).to(DEVICE)
        for inspin in circuit.inspace
    ])

    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = (2 ** circuit.A) * (2 ** circuit.M)
    exp_virtual_right = virtual_right.unsqueeze(1).expand(-1, rows_per_input, -1).reshape(2 ** circuit.N * rows_per_input, -1)
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # Filter out the rows with correct answers
    row_mask = constraints[..., circuit.N:(circuit.N + circuit.M)].any(dim=-1)
    terms = keys(circuit, degree)

    # Filter out connections between inputs
    col_mask = torch.tensor([
        max(key) >= circuit.N
        for key in terms
    ])
    terms = [
        term for term in terms if max(term) >= circuit.N
    ]
    constraints = constraints[row_mask][..., col_mask]

    return constraints.cpu(), terms
