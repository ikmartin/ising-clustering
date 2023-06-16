from csv import Error
from spinspace import Spin
from itertools import chain, combinations
import torch
import numba

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = "cpu"


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def all_answers(circuit):
    return dec2bin(torch.arange(2**circuit.G).to(DEVICE), circuit.G)


def all_answers_basic(size):
    return dec2bin(torch.arange(2**size).to(DEVICE), size)


def batch_tensor_power(input, power):
    num_rows, num_cols = input.shape
    index = (
        torch.combinations(torch.arange(num_cols).to(DEVICE), r=power)
        .unsqueeze(0)
        .expand(num_rows, -1, -1)
    )
    input = input.unsqueeze(-1).expand(-1, -1, power)
    return torch.gather(input, 1, index).prod(dim=-1)


def batch_vspin(input, degree):
    return torch.cat(
        [input] + [batch_tensor_power(input, i) for i in range(2, degree + 1)], dim=1
    )


def keys(circuit, degree):
    return list(
        chain.from_iterable(
            [combinations(range(circuit.G), i) for i in range(1, degree + 1)]
        )
    )


basin_dict = {}


def get_basin(N, num, d):
    """Returns all binary strings length N which are hamming distance <dist> from the given number num.

    Parameters
    ----------
    N : int
        the length of the binary string
    num : int
        the number to be used as the center of the hamming distance circle
    dist : int
        the hamming distance radius
    """

    try:
        return basin_dict[(N, num, d)]
    except KeyError:
        b = dec2bin(num, N)
        basin_dict[(N, num, d)] = []
        for ind in combinations(range(N), r=d):
            flipped_guy = flip_bits(num, ind)
            basin_dict[(N, num, d)].append(flipped_guy)

        return basin_dict[(N, num, d)]


def flip_bits(num, ind):
    for k in ind:
        # XOR with (00001) shifted into kth position
        num = num ^ (1 << k)
    return num


def all_filtered_states(circuit, filtered_levels):
    shift = circuit.M + circuit.A
    states = [
        (i << shift) | out for i, row in enumerate(filtered_levels) for out in row
    ]
    return dec2bin(torch.tensor(states), bits=circuit.N + shift)


def IMul_correct_rows(circuit):
    correct_rows = torch.cat(
        [
            torch.tensor(circuit.inout(inspin).binary()).unsqueeze(0).to(DEVICE)
            for inspin in circuit.inspace
        ]
    )


def basin_d_constraints(circuit, d, degree=2):
    """A generator for constraint sets. Starts with basin 2 constraints (that is basin one and two) and then"""
    MA = circuit.M + circuit.A
    filtered_levels = [
        sum([get_basin(MA, circuit.f(i).asint(), r) for r in range(1, d + 1)], [])
        for i in range(1 << circuit.N)
    ]
    return filtered_constraints(circuit, filtered_levels, degree=degree)


def filtered_constraints(circuit, filtered_levels, degree=2):
    """A method for building all levels of a filtered constraint set simultaneously.

    Parameters
    ----------
    circuit : (PICircuit)
        the circuit for which we build the constraints
    filtered_levels : list[list[int]]
        a list of list where row i is the list of wrong outauxes, represented by integers, to be used as constraints for input level i.
    """
    if len(filtered_levels) != 1 << circuit.N:
        raise KeyError("The list <filtered_levels> is missing input levels!")

    # make the correct block
    correct_rows = IMul_correct_rows(circuit)
    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = [len(row) for row in filtered_levels]
    exp_virtual_right = torch.repeat_interleave(
        virtual_right, torch.tensor(rows_per_input), dim=0
    )

    # make the wrong block
    all_states = all_filtered_states(circuit, filtered_levels)
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # filter out the rows with correct answers
    row_mask = constraints[..., circuit.N : (circuit.N + circuit.M)].any(dim=-1)
    terms = keys(circuit, degree)

    # filter out connections between inputs
    col_mask = torch.tensor([max(key) >= circuit.N for key in terms])
    terms = [term for term in terms if max(term) >= circuit.N]
    constraints = constraints[row_mask][..., col_mask]

    return constraints.cpu().to_sparse_csc()


def basin_d_constraints(circuit, d, degree=2):
    """A generator for constraint sets. Starts with basin 2 constraints (that is basin one and two) and then"""
    MA = circuit.M + circuit.A
    filtered_levels = [
        sum([get_basin(MA, circuit.f(i).asint(), r) for r in range(1, d + 1)], [])
        for i in range(1 << circuit.N)
    ]
    return filtered_constraints(circuit, filtered_levels, degree=degree)


def filtered_sequential_constraints(circuit, degree, sieve):
    """A currying function. Returns a method used for filtering out constraints built sequentially (i.e. one input level at a time.)

    Parameters
    ----------
    circuit : PICircuit
        the circuit for whom constraints are to be generated
    degree : int
        an upper bound on the degree of interactions
    sieve : callable
        a method which returns a list of the wrong outaux spins, written as integers, which will be used to build constraints at the given input level. OUTPUT NEEDS TO BE A LIST OF outaux PAIRS IN INTEGER FORMAT.
    """

    terms = keys(circuit, degree)

    # Filter out connections between inputs
    col_mask = torch.tensor([max(key) >= circuit.N for key in terms])
    terms = [term for term in terms if max(term) >= circuit.N]

    num_out_aux = circuit.M + circuit.A
    rows_per_input = 2**num_out_aux

    def make(inspin):
        # generate the wrong answers for this input level
        wrong_outaux, num_constraints = sieve(inspin)
        all_wrong_block = dec2bin(torch.tensor(wrong_outaux).to(DEVICE), num_out_aux)

        # generate a row tensor corresponding to the correct output and convert to virtual spins
        correct_row = (
            torch.tensor(circuit.inout(inspin).binary()).unsqueeze(0).to(DEVICE)
        )
        virtual_right = batch_vspin(correct_row, degree)
        exp_virtual_right = virtual_right.expand(num_constraints, -1)

        # mash the input together with the wrong outputs for virtual spin calculation
        inspin_block = (
            torch.tensor(inspin.binary()).unsqueeze(0).expand(num_constraints, -1)
        )
        wrong_block = torch.cat([inspin_block, all_wrong_block], dim=-1)
        virtual_wrong = batch_vspin(wrong_block, degree)

        # get the constraints
        constraints = virtual_wrong - exp_virtual_right

        # mask rows whose output portions are all 0
        row_mask = constraints[..., circuit.N : (circuit.N + circuit.M)].any(dim=-1)
        red_constraints = constraints[row_mask][..., col_mask]
        numconst = red_constraints.shape[0]
        return red_constraints.to_sparse(), numconst

    return make, terms
