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
def full_constraints(N1, N2, aux):
    return constraints(N1, N2, aux, 2, None, None, None, None)


def make_correct_rows(
    n1, n2, aux_array=None, included=None, desired=None, and_pairs=None
):
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

    included = () if included is None else included
    num_fixed_columns += len(included)
    desired = tuple(range(N)) if desired is None else desired
    num_free_columns = len(desired)
    aux_array = torch.tensor([], dtype=torch.int8) if aux_array is None else aux_array
    num_free_columns += aux_array.shape[0]

    original_correct = torch.cat([inp1_bits, inp2_bits, output_bits], dim=-1)
    ands = torch.tensor([], dtype=torch.int8)
    if and_pairs is not None:
        ands = torch.cat(
            [
                torch.prod(original_correct[..., pair], dim=-1, keepdim=True)
                for pair in and_pairs
            ],
            dim=-1,
        )
        num_fixed_columns += len(and_pairs)

    return (
        torch.cat(
            [
                inp1_bits,
                inp2_bits,
                output_bits[..., included],
                ands,
                output_bits[..., desired],
                torch.t(aux_array),
            ],
            dim=-1,
        ),
        num_fixed_columns,
        num_free_columns,
    )


@cache
def make_wrong_mask(num_fixed, num_free, radius):
    radius = num_free if radius is None else radius
    flip = torch.cat(
        [
            torch.cat(
                [
                    torch.sum(
                        F.one_hot(indices, num_classes=num_free), dim=0, keepdim=True
                    )
                    for indices in torch.combinations(torch.arange(num_free), r)
                ]
            )
            for r in range(1, radius + 1)
        ]
    ).to(torch.int8)
    return torch.cat(
        [torch.zeros(flip.shape[0], num_fixed, dtype=torch.int8), flip], dim=-1
    )


def make_wrongs(correct, num_fixed, num_free, radius):
    wrong_mask = make_wrong_mask(num_fixed, num_free, radius)
    exp_correct = correct.unsqueeze(1).expand(-1, wrong_mask.shape[0], -1)
    exp_wrong_mask = wrong_mask.unsqueeze(0).expand(correct.shape[0], -1, -1)
    return ((exp_correct + exp_wrong_mask) % 2).reshape(
        wrong_mask.shape[0] * correct.shape[0], -1
    ), wrong_mask.shape[0]


def constraints(n1, n2, aux, degree, radius, included, desired, and_pairs):
    correct, num_fixed, num_free = make_correct_rows(
        n1, n2, aux, included, desired, and_pairs
    )
    wrong, rows_per_input = make_wrongs(correct, num_fixed, num_free, radius)

    virtual_right = batch_vspin(correct, degree)
    exp_virtual_right = (
        virtual_right.unsqueeze(1)
        .expand(-1, rows_per_input, -1)
        .reshape(wrong.shape[0], -1)
    )
    virtual_wrong = batch_vspin(wrong, degree)
    constraints = virtual_wrong - exp_virtual_right

    col_mask = constraints.any(dim=0)
    constraints = constraints[..., col_mask]

    terms = keys(num_fixed + num_free, degree)
    terms = [term for term, flag in zip(terms, col_mask) if flag]

    return constraints.cpu(), terms, correct


def all_filtered_states(N1, N2, A, filtered_levels):
    N = N1 + N2
    shift = N + A
    states = [
        (i << shift) | out for i, row in enumerate(filtered_levels) for out in row
    ]
    return dec2bin(torch.tensor(states), bits=N + shift)


def correct_rows(circuit):
    correct_rows = torch.cat(
        [
            torch.tensor(circuit.inout(inspin).binary()).unsqueeze(0).to(DEVICE)
            for inspin in circuit.inspace
        ]
    )
    return correct_rows


# Andrew's current tensor-fied way to retreive the correct rows of IMulN1xN2xA
def IMul_correct_rows(
    n1, n2, aux_array=None, included=None, desired=None, and_pairs=None
):
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

    included = () if included is None else included
    num_fixed_columns += len(included)
    desired = tuple(range(N)) if desired is None else desired
    num_free_columns = len(desired)
    aux_array = torch.tensor([], dtype=torch.int8) if aux_array is None else aux_array
    num_free_columns += aux_array.shape[0]

    original_correct = torch.cat([inp1_bits, inp2_bits, output_bits], dim=-1)
    ands = torch.tensor([], dtype=torch.int8)
    if and_pairs is not None:
        ands = torch.cat(
            [
                torch.prod(original_correct[..., pair], dim=-1, keepdim=True)
                for pair in and_pairs
            ],
            dim=-1,
        )
        num_fixed_columns += len(and_pairs)

    return (
        torch.cat(
            [
                inp1_bits,
                inp2_bits,
                output_bits[..., included],
                ands,
                output_bits[..., desired],
                torch.t(aux_array),
            ],
            dim=-1,
        ),
        num_fixed_columns,
        num_free_columns,
    )


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
    A = len(aux)
    if len(filtered_levels) != 1 << N:
        raise KeyError("The list <filtered_levels> is missing input levels!")

    # make the correct block
    correct_rows = IMul_correct_rows(N1, N2, aux)
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
