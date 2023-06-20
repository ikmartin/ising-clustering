from csv import Error
from spinspace import Spin
from functools import cache
from itertools import chain, combinations
import numba
import torch
import torch.nn.functional as F

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = "cpu"


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def all_answers(G):
    return dec2bin(torch.arange(2**G).to(DEVICE), G)


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


def keys(G, degree):
    return list(
        chain.from_iterable([combinations(range(G), i) for i in range(1, degree + 1)])
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


####################################
### Full Constraint Methods
####################################


# Andrew's current tensor-fied way to retreive the correct rows of IMulN1xN2xA
def make_correct_rows(n1, n2, aux_array=None):
    if isinstance(aux_array, torch.Tensor) is not True and aux_array is not None:
        aux_array = torch.tensor(aux_array)
    N = n1 + n2
    num_inputs_1 = 1 << n1
    num_inputs_2 = 1 << n2
    num_inputs = num_inputs_1 * num_inputs_2
    num_fixed_columns = N

    inp1, inp2 = torch.meshgrid(
        torch.arange(num_inputs_1), torch.arange(num_inputs_2), indexing="ij"
    )
    outputs = inp1 * inp2
    inp1_bits = dec2bin(inp1, n1).to(torch.int8).reshape(num_inputs, n1)
    inp2_bits = dec2bin(inp2, n2).to(torch.int8).reshape(num_inputs, n2)
    output_bits = dec2bin(outputs, N).to(torch.int8).reshape(num_inputs, N)

    desired = tuple(range(N))
    num_free_columns = len(desired)
    aux_array = torch.tensor([], dtype=torch.int8) if aux_array is None else aux_array
    num_free_columns += aux_array.shape[0]

    original_correct = torch.cat([inp1_bits, inp2_bits, output_bits], dim=-1)

    return torch.cat(
        [
            inp1_bits,
            inp2_bits,
            output_bits,
            torch.t(aux_array),
        ],
        dim=-1,
    )


def full_constraints(N1, N2, aux, degree):
    N = N1 + N2
    M = N1 + N2
    try:
        A = len(aux)
    except TypeError:
        A = 0
    G = N + M + A
    all_states = all_answers(G)

    correct_rows = make_correct_rows(N1, N2, aux)

    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = (2**A) * (2**M)
    exp_virtual_right = (
        virtual_right.unsqueeze(1)
        .expand(-1, rows_per_input, -1)
        .reshape(2**N * rows_per_input, -1)
    )
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # Filter out the rows with correct answers
    row_mask = constraints[..., N : (N + M)].any(dim=-1)
    terms = keys(G, degree)

    # Filter out connections between inputs
    col_mask = torch.tensor([max(key) >= N for key in terms])
    terms = [term for term in terms if max(term) >= N]
    constraints = constraints[row_mask][..., col_mask]

    return constraints.cpu().to_sparse(), terms


def all_filtered_states(N1, N2, A, filtered_levels):
    N = N1 + N2
    shift = N + A
    states = [
        (i << shift) | out for i, row in enumerate(filtered_levels) for out in row
    ]
    return dec2bin(torch.tensor(states), bits=N + shift)


def filtered_constraints(N1, N2, aux, filtered_levels, degree=2):
    """A method for building all levels of a filtered constraint set simultaneously.

    Parameters
    ----------
    circuit : (PICircuit)
        the circuit for which we build the constraints
    filtered_levels : list[list[int]]
        a list of list where row i is the list of wrong outauxes, represented by integers, to be used as constraints for input level i.
    """
    N = N1 + N2
    M = N
    try:
        A = len(aux)
    except TypeError:
        A = 0

    if len(filtered_levels) != 1 << N:
        raise KeyError("The list <filtered_levels> is missing input levels!")

    # make the correct block
    correct_rows = make_correct_rows(N1, N2, aux)
    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = [len(row) for row in filtered_levels]
    exp_virtual_right = torch.repeat_interleave(
        virtual_right, torch.tensor(rows_per_input), dim=0
    )

    # make the wrong block
    all_states = all_filtered_states(N1, N2, A, filtered_levels)
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # filter out the rows with correct answers
    row_mask = constraints[..., N : (N + M)].any(dim=-1)
    terms = keys(N + M + A, degree)

    # filter out connections between inputs
    col_mask = torch.tensor([max(key) >= N for key in terms])
    terms = [term for term in terms if max(term) >= N]
    constraints = constraints[row_mask][..., col_mask]

    return constraints.cpu().to_sparse_csc()


def basin_d_constraints(circuit, d, degree=2):
    """A generator for constraint sets. Starts with basin 2 constraints (that is basin one and two) and then"""
    MA = circuit.M + circuit.A
    filtered_levels = [
        sum([get_basin(MA, circuit.f(i).asint(), r) for r in range(1, d + 1)], [])
        for i in range(1 << circuit.N)
    ]
    aux = circuit.get_aux_array()
    return filtered_constraints(
        circuit.N1, circuit.N2, aux, filtered_levels, degree=degree
    )
