import torch, click
import torch.nn as nn
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
    
def tensor_power(tensor: torch.Tensor, power: int) -> torch.Tensor:
    """
    Computes all products of (power) distinct variables in the input tensor. Returns answer in tensor format.
    """
   
    return torch.prod(torch.combinations(tensor, r=power), dim=1)

def lse(domain: Iterable, function: Callable):
    """
    Torch implementation of the log-sum-exp function
    """
    return torch.log(sum([torch.exp(function(element)) for element in domain]))

def binary(x: int, b: int) -> torch.Tensor:
    """
    Returns a binary representation of x with b bits
    """
    
    if not b:
        return torch.tensor([]).cuda()

    mask = 2 ** torch.arange(b).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

class Hamiltonian(nn.Module):
    def __init__(self, degree: int, dimension: int) -> None:
        super(Hamiltonian, self).__init__()

        assert degree <= dimension

        self.degree = degree
        self.dimension = dimension

        # Create initial sets of coefficients using an iid Gaussian distribution. We keep the coefficients for terms of each degree seperate mostly in order to keep the code organized.
        num_coeffs = sum([choose(dimension, deg) for deg in range(1, degree+1)])
        self.coeffs = nn.Parameter(torch.randn(num_coeffs))

    def forward(self, input: torch.Tensor):
        """
        Evaluates the Hamiltonian polynomial on the given choice of the variables. The outer products are, for now, computed at runtime, but if we need more speed later we could trade this off for memory by precomputing the outer products for every spin state.
        """

        assert len(input) == self.dimension
        
        terms_tensor = torch.cat([tensor_power(input, deg) for deg in range(1, self.degree+1)])
        output = terms_tensor.float() @ self.coeffs
        return output 
        
class BoltzmannModel(nn.Module):
    def __init__(self, device, H: Hamiltonian, circuit: PICircuit, num_aux: int = 0, skew: float = 10, beta: float = 1):
        super(BoltzmannModel, self).__init__()
        assert skew > 0 and beta > 0

        self.device = device
        self.H = H.to(device)
        self.circuit = circuit
        self.num_aux = num_aux
        self.skew = skew # named lambda in the math writeup
        self.beta = beta

        self.answer_table = self._gen_answer_table()
        self.space_sizes = [circuit.inspace.dim, circuit.outspace.dim, num_aux]

    def _gen_answer_table(self):
        return torch.tensor([
            self.circuit.fout(self.circuit.inspace.getspin(i)).asint()
            for i in range(self.circuit.inspace.size)
        ]).to(self.device)

    @cache
    def _format_binary_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Expects a state encoded as a tensor of ints. Returns the concatenated binary tensor.
        """

        binary_tensor = torch.cat([
            binary(x, b) for x, b in zip(state, self.space_sizes)
        ]).to(self.device)
        return binary_tensor

    @cache
    def _inputlevel(self, input: int) -> torch.Tensor:
        return torch.tensor([
            [input, out, aux]
            for out, aux in product(range(2 ** self.space_sizes[1]), range(2 ** self.num_aux))
        ]).to(self.device)

    @cache
    def _wrongspace(self, input: int) -> Iterable:
        return torch.tensor([
            [input, out, aux]
            for out, aux in product(range(2 ** self.space_sizes[1]), range(2 ** self.num_aux))
            if out != self.answer_table[input]
        ]).to(self.device)

    def _energy(self, state: torch.Tensor):
        """
        State should be a tensor of the integer values of each spin part, i.e. [in, out, aux] or [in, out]. We will convert it to binary, then feed it to the Hamiltonian.
        """
        state = self._format_binary_state(state)
        return torch.exp(-self.beta * self.H(state))

    def forward(self, input: int):
        """
        Expects the input as an integer.
        """

        wrong_factors = [self._energy(state) for state in self._wrongspace(input)]
        all_factors = [self._energy(state) for state in self._inputlevel(input)]
        output = torch.exp(self.skew * (
                         torch.log(sum(wrong_factors))
                         - torch.log(sum(all_factors))
        ))
        return output
        
class SpinDataset(Dataset):
    def __init__(self, spinspace: Spinspace):
        self.spinspace = spinspace
        self.size = spinspace.size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> int:
        return index

def train(device, model, optimizer, dataloader, scheduler, epoch):
    losses = []
    loop = tqdm(dataloader, leave = True)
    for input in loop: 
        loss = model(input)
        losses.append(loss.cpu())
        
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
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-1
    WEIGHT_DECAY = 1e-5
    MOMENTUM = 0.9
    NESTEROV = True
    LAMBDA = 10
    BETA = 1

    # Set up the model
    circuit = IMul(n1, n2)
    dimension = circuit.G + aux
    H = Hamiltonian(degree, dimension)
    model = BoltzmannModel(device, H, circuit, aux, LAMBDA, BETA).to(device)

    # Set up the dataset
    dataset = SpinDataset(circuit.inspace)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        pin_memory = True,
        num_workers = 2
    )

    # Optimizer
    optimizer = SGD(
        H.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        momentum = MOMENTUM,
        nesterov = NESTEROV
    )

    #optimizer = torch.optim.Adam(H.parameters())

    # LR Scheduler (annealing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        patience = 2,
        threshold = 1e-3,
        verbose = True
    )

    # Run the training loop
    for epoch in range(NUM_EPOCHS):
        train(device, model, optimizer, dataloader, scheduler, epoch)
    

if __name__ == '__main__':
    main()
