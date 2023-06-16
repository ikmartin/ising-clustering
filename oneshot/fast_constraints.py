#from constraint-reduction.basin_one_constraint_stats import num_constraints
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


def keys_basic(size, degree):
    return list(
        chain.from_iterable(
            [combinations(range(size), i) for i in range(1, degree + 1)]
        )
    )


def fast_constraints(circuit, degree):
    all_states = all_answers(circuit)

    correct_rows = torch.cat(
        [
            torch.tensor(circuit.inout(inspin).binary()).unsqueeze(0).to(DEVICE)
            for inspin in circuit.inspace
        ]
    )

    #print(correct_rows)

    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = (2**circuit.A) * (2**circuit.M)
    exp_virtual_right = (
        virtual_right.unsqueeze(1)
        .expand(-1, rows_per_input, -1)
        .reshape(2**circuit.N * rows_per_input, -1)
    )
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # Filter out the rows with correct answers
    row_mask = constraints[..., circuit.N : (circuit.N + circuit.M)].any(dim=-1)
    terms = keys(circuit, degree)

    # Filter out connections between inputs
    col_mask = torch.tensor([max(key) >= circuit.N for key in terms])
    terms = [term for term in terms if max(term) >= circuit.N]
    constraints = constraints[row_mask][..., col_mask]

    return constraints.cpu().to_sparse(), terms


# @numba.njit(nopython = True)
def CSC_constraints(n1, n2, aux, degree):
    aux = torch.t(aux)
    num_aux = aux.shape[1]
    G = 2 * (n1 + n2) + num_aux
    all_states = all_answers_basic(G)

    inp2_mask = 2 ** (n2) - 1
    correct_rows = torch.cat(
        [
            torch.tensor([[inp, (inp & inp2_mask) * (inp >> n2)]]).to(DEVICE)
            for inp in range(2 ** (n1 + n2))
        ]
    )
    correct_rows = torch.flatten(dec2bin(correct_rows, n1 + n2), start_dim=-2)
    correct_rows = torch.cat([correct_rows, aux], dim=-1)

    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = (2**num_aux) * (2 ** (n1 + n2))
    exp_virtual_right = (
        virtual_right.unsqueeze(1)
        .expand(-1, rows_per_input, -1)
        .reshape(2 ** (n1 + n2) * rows_per_input, -1)
    )
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # Filter out the rows with correct answers
    row_mask = constraints[..., (n1 + n2) : (2 * (n1 + n2))].any(dim=-1)
    terms = keys_basic(G, degree)

    # Filter out connections between inputs
    col_mask = torch.tensor([max(key) >= n1 + n2 for key in terms])
    terms = [term for term in terms if max(term) >= n1 + n2]
    constraints = constraints[row_mask][..., col_mask]

    return constraints.cpu().to_sparse_csc(), terms

def constraints_building(n1, n2, aux, degree, radius=None, random = 0, mask = None, include = None, exclude_correct_out = False):
    
    if aux is not None:
        aux = torch.t(aux)
        num_aux = aux.shape[1]
    else: 
        num_aux = 0

    if include is not None:
        N = n1 + n2 + len(include)
    else:
        N = n1 + n2

    if mask is None:
        M = n1 + n2
    else:
        M = len(mask)

    G = N + M + num_aux
    if radius is None:
        radius = M + num_aux

    bit_filter = torch.tensor([False] * (2*(n1+n2)))
    #print(f'DEBUG {bit_filter} {bit_filter.shape}')
    if mask is not None:
        bit_filter[torch.tensor(mask) + n1 + n2] = True
    else:
        bit_filter[(n1+n2):(2*(n1+n2))] = True

    include_filter = torch.tensor([False] * (2*(n1+n2)))
    if include is not None:
        include_filter[torch.tensor(include) + n1 + n2] = True

    inp2_mask = 2 ** (n2) - 1
    correct_rows = torch.cat(
        [
            torch.tensor([[inp, (inp & inp2_mask) * (inp >> n2)]]).to(DEVICE)
            for inp in range(2 ** (n1+n2))
        ]
    )
    correct_rows = torch.flatten(dec2bin(correct_rows, n1 + n2), start_dim=-2)
    output_bits = correct_rows[..., bit_filter]
    input_bits = correct_rows[..., :(n1+n2)]
    extra_inputs = correct_rows[..., include_filter]
    """
    torch.set_printoptions(threshold=50000, linewidth=200)

    print(output_bits)
    print(input_bits)
    print(extra_inputs)
    print(correct_rows)
    """
    args = [input_bits, extra_inputs, output_bits, aux]
    args = [arg for arg in args if arg is not None]
    correct_rows = torch.cat(args, dim=-1)
    #print(correct_rows)


    all_states = []
    num_per_row = 0
    for row in correct_rows:
        num_per_row = 0
        for i in range(1, radius+1):
            for diff in combinations(range(N, G), i):
                new_row = row.clone().detach()
                for k in diff:
                    new_row[k] = 1 - new_row[k]
                all_states.append(new_row.unsqueeze(0))
                num_per_row += 1

        for i in range(random):
            new_row = torch.cat([row[:N], torch.bernoulli(0.5*torch.ones(M+num_aux))]).unsqueeze(0)
            all_states.append(new_row)
            num_per_row += 1


    all_states = torch.cat(all_states)

    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = num_per_row
    exp_virtual_right = (
        virtual_right.unsqueeze(1)
        .expand(-1, rows_per_input, -1)
        .reshape((1 << (n1+n2)) * rows_per_input, -1)
    )
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # Filter out the rows with correct answers
    if exclude_correct_out:
        row_mask = constraints[..., N : (N+M)].any(dim=-1)
        constraints = constraints[row_mask]

    col_mask = constraints.any(dim=0)
    constraints = constraints[..., col_mask]

    terms = keys_basic(G, degree)
    terms = [term for term, flag in zip(terms, col_mask) if flag]

    #torch.set_printoptions(threshold=50000, linewidth=200)
    return constraints.cpu(), terms, correct_rows

def constraints_basin(n1, n2, aux, degree, radius, random = 0, mask = None):
    aux = torch.t(aux)
    num_aux = aux.shape[1]
    N = n1 + n2
    M = n1 + n2 if mask is None else len(mask)
    G = N + M + num_aux

    if mask is not None:
        bit_filter = torch.tensor([False] * (2*N + num_aux))
        bit_filter[:N] = True
        bit_filter[(2*N):] = True
        bit_filter[torch.tensor(mask) + N] = True

    inp2_mask = 2 ** (n2) - 1
    correct_rows = torch.cat(
        [
            torch.tensor([[inp, (inp & inp2_mask) * (inp >> n2)]]).to(DEVICE)
            for inp in range(2 ** N)
        ]
    )
    correct_rows = torch.flatten(dec2bin(correct_rows, n1 + n2), start_dim=-2)
    correct_rows = torch.cat([correct_rows, aux], dim=-1)

    if mask is not None:
        correct_rows = correct_rows[..., bit_filter]


    all_states = []
    num_per_row = 0
    for row in correct_rows:
        num_per_row = 0
        for i in range(1, radius+1):
            for diff in combinations(range(N, G), i):
                new_row = row.clone().detach()
                for k in diff:
                    new_row[k] = 1 - new_row[k]
                all_states.append(new_row.unsqueeze(0))
                num_per_row += 1

        for i in range(random):
            new_row = torch.cat([row[:N], torch.bernoulli(0.5*torch.ones(M+num_aux))]).unsqueeze(0)
            all_states.append(new_row)
            num_per_row += 1


    all_states = torch.cat(all_states)

    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = num_per_row
    exp_virtual_right = (
        virtual_right.unsqueeze(1)
        .expand(-1, rows_per_input, -1)
        .reshape((1 << N) * rows_per_input, -1)
    )
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # Filter out the rows with correct answers
    row_mask = constraints[..., N : (N+M)].any(dim=-1)
    col_mask = constraints.any(dim=0)
    constraints = constraints[row_mask][..., col_mask]

    terms = keys_basic(G, degree)
    terms = [term for term, flag in zip(terms, col_mask) if flag]

    #torch.set_printoptions(threshold=50000, linewidth=200)
    return constraints.cpu(), terms


def sequential_constraints(circuit, degree):
    terms = keys(circuit, degree)

    # Filter out connections between inputs
    col_mask = torch.tensor([max(key) >= circuit.N for key in terms])
    terms = [term for term in terms if max(term) >= circuit.N]

    num_out_aux = circuit.M + circuit.A
    rows_per_input = 2**num_out_aux
    all_wrong_block = dec2bin(torch.arange(2**num_out_aux).to(DEVICE), num_out_aux)

    def make(inspin):
        correct_row = (
            torch.tensor(circuit.inout(inspin).binary()).unsqueeze(0).to(DEVICE)
        )
        virtual_right = batch_vspin(correct_row, degree)
        exp_virtual_right = virtual_right.expand(rows_per_input, -1)

        inspin_block = (
            torch.tensor(inspin.binary()).unsqueeze(0).expand(rows_per_input, -1)
        )
        wrong_block = torch.cat([inspin_block, all_wrong_block], dim=-1)
        virtual_wrong = batch_vspin(wrong_block, degree)

        constraints = virtual_wrong - exp_virtual_right

        row_mask = constraints[..., circuit.N : (circuit.N + circuit.M)].any(dim=-1)
        return constraints[row_mask][..., col_mask].to_sparse()

    return make, terms


def filtered_constraints(circuit, degree, sieve):
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


def constraints_basin2(circuit, degree):
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

    MA = circuit.M + circuit.A

    def make(inspin):
        # generate the wrong answers for this input level
        wrong_outaux = get_basin(MA, circuit.f(inspin).asint(), 1) + get_basin(
            MA, circuit.f(inspin).asint(), 2
        )
        num_constraints = len(wrong_outaux)
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
