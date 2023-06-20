import torch
from fast_constraints import dec2bin

def make_correct_rows(n1, n2, aux_array = None, included = None, desired = None, and_pairs = None):
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
    aux_array = torch.tensor([], dtype=torch.int8) if aux_array is None else aux_array

    original_correct = torch.cat([inp1_bits, inp2_bits, output_bits], dim = -1)
    ands = torch.tensor([], dtype=torch.int8)
    if and_pairs is not None:
        ands = torch.cat([torch.prod(original_correct[..., pair], dim = -1, keepdim = True) for pair in and_pairs], dim = -1)
        num_fixed_columns += len(and_pairs)

    return torch.cat([inp1_bits, inp2_bits, output_bits[..., included], ands, output_bits[..., desired], torch.t(aux_array)], dim = -1), num_fixed_columns


print(make_correct_rows(2, 3, and_pairs = [(0,1), (1,2)]))
