import torch, click
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from ising import IMul, PICircuit
from spinspace import Spinspace, Spin
from tqdm import tqdm
from itertools import combinations, product
from typing import Iterable, Callable
from math import comb as choose
from statistics import mean
from functools import cache
import os
from joblib import Parallel, delayed
    
def tensor_power(tensor: torch.Tensor, power: int) -> torch.Tensor:
    """
    Computes all products of (power) distinct variables in the input tensor. Returns answer in tensor format.
    """
   
    return torch.prod(torch.combinations(tensor, r=power), dim=1)

def binary(x: int, b: int) -> torch.Tensor:
    """
    Returns a binary representation of x with b bits
    """

    mask = 2 ** torch.arange(b).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

        
class BoltzmannModel(nn.Module):
    def __init__(self, circuit: PICircuit, num_aux: int = 0, skew: float = 10, beta: float = 1):
        super(BoltzmannModel, self).__init__()

        assert skew > 0 and beta > 0

        self.H = nn.LazyLinear(1, bias = False)
        self.skew = skew # named lambda in the math writeup
        self.beta = beta

    def _boltzmann_factors(self, virtual_states: torch.Tensor):
        return torch.exp(-self.beta * self.H(virtual_states))

    def forward(self, input_levels: torch.Tensor, answers: torch.Tensor):
        # input_levels is of shape (batch_size, 2**M, 2**A, vspin_length)
        factors = self._boltzmann_factors(input_levels).squeeze(-1) # shape (batch_size, 2**M, 2**A)
        total_sums = torch.sum(factors, dim = (-1, -2)) # shape (batch_size)

        right_aux_factors = torch.diagonal(factors[..., answers, :]) # shape (2 ** A, batch_size)
        right_sums = torch.sum(right_aux_factors, dim = -2) # shape (batch_size)
        wrong_sums = total_sums - right_sums # shape (batch_size)

        return torch.exp(self.skew * (torch.log(wrong_sums) - torch.log(total_sums)))
        
class VirtualSpinDataset(Dataset):
    def __init__(self, circuit, A, degree=2):
        self.circuit = circuit
        N = self.circuit.inspace.dim
        M = self.circuit.outspace.dim
        self.shape = (N, M, A)
        self.degree = degree
        self.vspin_length = len(self._vspin((0,0,0)))
        self.answer_table = self._gen_answer_table()

    def _gen_answer_table(self):
        return torch.tensor([
            self.circuit.fout(self.circuit.inspace.getspin(i)).asint()
            for i in range(self.circuit.inspace.size)
        ])

    def _format_binary_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Expects a state encoded as a tensor of ints. Returns the concatenated binary tensor.
        """

        return torch.cat([
            binary(torch.tensor([x]), b)[0] for x, b in zip(state, self.shape) if b != 0
        ])

    def _vspin(self, state):
        """
        Expects int-encoded spin state. Returns a binary virtual spin.
        """

        # Convert state to explicit binary
        state = self._format_binary_state(state)

        # Return concatenated tensor powers of said explicit binary
        return torch.cat([tensor_power(state, deg) for deg in range(1, self.degree+1)])

    def __len__(self):
        return 2 ** self.shape[0]

    def cache_signature(self, index):
        return f'mul{self.shape}d{self.degree}_{index}'

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Given an input spin index, returns a tensor of shape [2^M, 2^A, (vspin length)] representing the input level, as well as the index of the correct answer.
        """

        cache_name = f'cache/{self.cache_signature(index)}.dat'
        if os.path.exists(cache_name):
            return torch.load(cache_name)

        N, M, A = self.shape
        input_level = torch.cat([
            self._vspin(torch.tensor([index, m, a]))
            for m, a in product(range(2 ** M), range(2 ** A))
        ]).reshape(2 ** M, 2 ** A, self.vspin_length)

        return input_level.float(), self.answer_table[index]


class ParallelProgress(Parallel):
    def __init__(self, num_total_tasks, *args, **kwargs):
        super(ParallelProgress, self).__init__(*args, **kwargs)
        self.num_total_tasks = num_total_tasks

    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            super().__call__(*args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.num_total_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def cache_write(dataset: VirtualSpinDataset, index):
    filename = f'cache/{dataset.cache_signature(index)}.dat'
    if not os.path.exists(f'cache/{filename}'):
        data_tuple = dataset.__getitem__(index)
        torch.save(data_tuple, filename)

def write_data_to_disk(dataset: VirtualSpinDataset):
    if not os.path.exists('cache'):
        os.makedirs('cache')
    
    print('Writing dataset to disk...')
    report = ParallelProgress(len(dataset), n_jobs=-1)(delayed(cache_write)(dataset, i) for i in range(len(dataset)))

def train(device, model, optimizer, dataloader, scheduler, epoch):
    losses = []
    loop = tqdm(dataloader, leave = True)
    for input_levels, answers in loop: 
        input_levels, answers = input_levels.to(device), answers.to(device)
        out = model(input_levels, answers)
        loss = sum(out) / len(out)
        losses.append(sum(out).cpu())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = mean([val.item() for val in losses]), epoch = epoch)

    estimated_max_failure_prob = (1 / model.skew) * torch.log(sum(losses))
    scheduler.step(estimated_max_failure_prob)
    print(f'approx max failure probability = {estimated_max_failure_prob}')

@click.command()
@click.option('--n1', default=2, help = 'Length in bits of first input')
@click.option('--n2', default=2, help = 'Length in bits of second input')
@click.option('--aux', default=0, help = 'Number of auxilliary spins')
@click.option('--degree', default=2, help = 'Degree of the Hamiltonian polynomial to fit.')
def main(n1: int, n2: int, aux: int, degree: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print(f'Warning: Running without CUDA support! Current device is {device}')

    # Hyperparameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-1
    WEIGHT_DECAY = 1e-5
    MOMENTUM = 0.9
    NESTEROV = True
    LAMBDA = 10
    BETA = 1

    # Set up the model
    circuit = IMul(n1, n2)
    dimension = circuit.G + aux
    model = BoltzmannModel(circuit, aux, LAMBDA, BETA).to(device)

    # Set up the dataset
    dataset = VirtualSpinDataset(circuit, aux, degree=degree)
    write_data_to_disk(dataset)
    drop_last = bool(len(dataset) % BATCH_SIZE)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        pin_memory = True,
        num_workers = 2,
        drop_last = drop_last
    )

    # Optimizer
    optimizer = SGD(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        momentum = MOMENTUM,
        nesterov = NESTEROV
    )

    #optimizer = torch.optim.Adam(H.parameters())

    # LR Scheduler (annealing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        patience = 5,
        threshold = 1e-3,
        verbose = True
    )

    # Run the training loop
    for epoch in range(NUM_EPOCHS):
        train(device, model, optimizer, dataloader, scheduler, epoch)

    print(list(model.H.parameters()))
    

if __name__ == '__main__':
    main()
