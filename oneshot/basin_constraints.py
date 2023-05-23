import torch
from ising import PICircuit
from spinspace import Spin
from oneshot import MLPoly, reduce_poly
from itertools import chain, combinations
from polyfit import gen_var_keys

def tensor_power(tensor: torch.Tensor, power: int) -> torch.Tensor:
    """
    Computes all products of (power) distinct variables in the input tensor. Returns answer in tensor format.
    """
   
    return torch.prod(torch.combinations(tensor, r=power), dim=1)
    
def vspin(state: torch.Tensor, degree: int) -> torch.Tensor:
    # Return concatenated tensor powers of said explicit binary
    return torch.cat([tensor_power(state, deg) for deg in range(1, degree+1)])


def hypercube_single_constraint(inspin, outspin, degree, i):
    other_out = outspin.clone()
    other_out[i] = 1-other_out[i]
    other = torch.cat([inspin, other_out])
    current = torch.cat([inspin, outspin])
    complete_constraint_row = vspin(other, degree) - vspin(current, degree)
    return complete_constraint_row

def gen_hypercube_constraints(circuit, degree, spin):
    inspin, outspin, _ = spin.split()
    inspin = Spin(inspin.asint(), shape = (circuit.N1, circuit.N2))
    correct_bin = torch.tensor(circuit.fout(inspin).binary())
    current_bin = torch.tensor(outspin.binary())
    in_bin = torch.tensor(inspin.binary())
    constraint_list = [
        hypercube_single_constraint(in_bin, current_bin, degree, i).unsqueeze(0)
        for i, b in enumerate(current_bin)
        if b == correct_bin[i]
    ]
    if len(constraint_list):
        return torch.cat(constraint_list)
    
    return None

def make_poly(dim: int, degree: int, coeffs: torch.Tensor) -> MLPoly:
    keys = list(chain.from_iterable([
        combinations(range(dim), deg)
        for deg in range(1, degree+1)
    ]))
    coeff_dict = {
        key: val.item()
        for key, val in zip(keys, coeffs)
        if val
    }
    return MLPoly(coeff_dict)

def get_constraints(circuit, degree):
    return get_constraint_matrix(circuit, degree), gen_var_keys(circuit, degree)

def get_constraint_matrix(circuit: PICircuit, degree: int) -> torch.Tensor:
    constraint_sets = [
        gen_hypercube_constraints(circuit, degree, spin)
        for spin in circuit.spinspace
    ]
    constraint_sets = list(filter(lambda x: x is not None, constraint_sets))
    constraint_matrix = torch.cat(constraint_sets)

    ## At this point, the constraint matrix is the `entire' constraint matrix, and still includes columns corresponding to the terms which are products only of the input spins. These are constant on every input level, and totally irrelevant to the behavior of the Hamiltonian. Therefore, we do not want to waste effort on fitting variables to these. We will determine which columns are useless and remove them from the constraint matrix.
    mask_tensor = torch.cat([torch.ones(circuit.N), torch.zeros(circuit.M)]).byte()
    vspin_mask = (vspin(mask_tensor, degree) - 1).nonzero(as_tuple = True)[0]
    constraint_matrix = torch.index_select(constraint_matrix, dim=1, index=vspin_mask)
    return constraint_matrix
