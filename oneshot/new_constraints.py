import torch
import torch.nn.functional as F
from fast_constraints import dec2bin, batch_vspin, keys_basic
from functools import cache
from itertools import combinations

def make_correct_rows(n1, n2, aux_array = None, included = None, desired = None, and_pairs = None, auxfix = False):
    N = n1 + n2
    num_inputs_1 = 1 << n1
    num_inputs_2 = 1 << n2
    num_inputs = num_inputs_1 * num_inputs_2
    num_fixed_columns = N

    inp1, inp2 = torch.meshgrid(torch.arange(num_inputs_1), torch.arange(num_inputs_2), indexing = "ij")
    outputs = inp1*inp2
    inp1_bits = dec2bin(inp1, n1).to(torch.int8).reshape(num_inputs, n1)
    inp2_bits = dec2bin(inp2, n2).to(torch.int8).reshape(num_inputs, n2)
    output_bits = dec2bin(outputs, N).to(torch.int8).reshape(num_inputs, N)

    included = () if included is None else included
    num_fixed_columns += len(included)
    desired = tuple(range(N)) if desired is None else desired
    num_free_columns = len(desired)

    original_correct = torch.cat([inp1_bits, inp2_bits, output_bits], dim = -1)
    ands = torch.tensor([], dtype=torch.int8)
    if and_pairs is not None:
        ands = torch.cat([torch.prod(original_correct[..., pair], dim = -1, keepdim = True) for pair in and_pairs], dim = -1)
        num_fixed_columns += len(and_pairs)

    # JUST FOR TESTING---HAVE AUX ARRAY BE FIXED INPUTS
    if auxfix:
        num_fixed_columns += aux_array.shape[0] if aux_array is not None else 0
        aux_array = torch.tensor([], dtype=torch.int8) if aux_array is None else aux_array
        return torch.cat([inp1_bits, inp2_bits, output_bits[..., included], ands, torch.t(aux_array), output_bits[..., desired]], dim = -1), num_fixed_columns, num_free_columns

    # ORIGINAL---AUX ARRAY IS FREE
    num_free_columns += aux_array.shape[0] if aux_array is not None else 0
    aux_array = torch.tensor([], dtype=torch.int8) if aux_array is None else aux_array
    return torch.cat([inp1_bits, inp2_bits, output_bits[..., included], ands, output_bits[..., desired], torch.t(aux_array)], dim = -1), num_fixed_columns, num_free_columns

@cache
def make_wrong_mask(num_fixed, num_free, radius):
    radius = num_free if radius is None else radius
    possible_masks = dec2bin(torch.arange(1, 1 << num_free), num_free).to(torch.int8)
    masks = possible_masks[torch.sum(possible_masks, dim=-1) <= radius]
    return torch.cat([torch.zeros(masks.shape[0], num_fixed, dtype = torch.int8), masks], dim = -1)
        
def make_wrongs(correct, num_fixed, num_free, radius):
    wrong_mask = make_wrong_mask(num_fixed, num_free, radius)
    exp_correct = correct.unsqueeze(1).expand(-1, wrong_mask.shape[0], -1)
    exp_wrong_mask = wrong_mask.unsqueeze(0).expand(correct.shape[0], -1, -1)
    return ((exp_correct + exp_wrong_mask) % 2).reshape(wrong_mask.shape[0] * correct.shape[0], -1), wrong_mask.shape[0]



def add_function_ands(matrix, functions):
    for func in functions:
        col = torch.prod(matrix[...,func], dim=-1, keepdim = True)
        matrix = torch.cat([matrix,col], dim=-1)
    return matrix

def add_hyperplane(matrix, hyperplane):
    w, b = hyperplane
    new_vec = torch.sign(-b + F.linear(matrix[...,:len(w)].float(), w)).unsqueeze(1)
    new_vec[new_vec == 0] = 1
    new_vec = (new_vec + 1)/2
    #print(torch.t(new_vec).to(torch.int8))
    return torch.cat([matrix, new_vec], dim = -1)

def add_bool_func(matrix, func):
    indices, function = func
    args = matrix[...,indices]
    new_vec = torch.zeros(matrix.shape[0])
    two_powers = torch.tensor([(1 << (len(indices) - j - 1)) for j in range(len(indices))])
    vals = F.linear(args.float(), two_powers.float()).int()
    new_vec = torch.tensor(function[vals])

    return torch.cat([matrix, new_vec.unsqueeze(1)], dim = -1)

def constraints(n1, n2, aux = None, degree = 2, radius = None, included = None, desired = None, and_pairs = None, ising=False, function_ands = None, auxfix = False, hyperplanes = None, bool_funcs = None):
    correct, num_fixed, num_free = make_correct_rows(n1, n2, aux, included, desired, and_pairs, auxfix)
    wrong, rows_per_input = make_wrongs(correct, num_fixed, num_free, radius)

    if function_ands is not None:
        correct = add_function_ands(correct, function_ands)
        wrong = add_function_ands(wrong, function_ands)
    
    if hyperplanes is not None:
        for hyperplane in hyperplanes:
            correct = add_hyperplane(correct, hyperplane)
            wrong = add_hyperplane(wrong, hyperplane)

    if bool_funcs is not None:
        for func in bool_funcs:
            correct = add_bool_func(correct, func)
            wrong = add_bool_func(wrong, func)
            

    if ising:
        correct = (2*correct)-1
        wrong = (2*wrong)-1
    
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

    terms = keys_basic(num_fixed+num_free, degree)
    terms = [term for term, flag in zip(terms, col_mask) if flag]

    return constraints.cpu(), terms, correct
