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


def constraints_basin1(n1, n2, aux, degree):
    aux = torch.t(aux)
    num_aux = aux.shape[1]
    G = 2 * (n1 + n2) + num_aux

    inp2_mask = 2 ** (n2) - 1
    correct_rows = torch.cat(
        [
            torch.tensor([[inp, (inp & inp2_mask) * (inp >> n2)]]).to(DEVICE)
            for inp in range(2 ** (n1 + n2))
        ]
    )
    correct_rows = torch.flatten(dec2bin(correct_rows, n1 + n2), start_dim=-2)
    correct_rows = torch.cat([correct_rows, aux], dim=-1)

    all_states = []
    for row in correct_rows:
        for i in range(n1 + n2, 2 * (n1 + n2) + num_aux):
            new_row = row.clone().detach()
            new_row[i] = 1 - new_row[i]
            all_states.append(new_row.unsqueeze(0))
            print(new_row)

    all_states = torch.cat(all_states)

    virtual_right = batch_vspin(correct_rows, degree)
    rows_per_input = n1 + n2 + num_aux
    exp_virtual_right = (
        virtual_right.unsqueeze(1)
        .expand(-1, rows_per_input, -1)
        .reshape(2 ** (n1 + n2) * rows_per_input, -1)
    )
    virtual_all = batch_vspin(all_states, degree)
    constraints = virtual_all - exp_virtual_right

    # Filter out the rows with correct answers
    row_mask = constraints[..., (n1 + n2) : (2 * (n1 + n2))].any(dim=-1)
    col_mask = constraints.any(dim=0)
    constraints = constraints[row_mask][..., col_mask]

    terms = keys_basic(G, degree)
    terms = [term for term, flag in zip(terms, col_mask) if flag]

    torch.set_printoptions(threshold=50000, linewidth=200)
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
