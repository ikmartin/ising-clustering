import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import matplotlib.pyplot as plt
from new_constraints import constraints
from fast_constraints import batch_tensor_power

torch.set_printoptions(threshold=50000)

class Model(nn.Module):
    def __init__(self, na):
        super().__init__()

        self.main = nn.LazyLinear(1)
        self.thresh = nn.LazyLinear(na)
        self.h = nn.Parameter(torch.randn(na))
        self.u = nn.Parameter(torch.randn(na))
        self.q = nn.LazyLinear(1)
        self.s = nn.Tanh()
        self.r = nn.ReLU()
        self.gamma=1

    def setgamma(self, g):
        self.gamma = g

    def forward(self, inps):
        inps=inps.float()
        x = inps[..., :-1]
        y = inps[..., -1]
        
        t1 = self.s(self.gamma * self.thresh(torch.cat([x,y.unsqueeze(1)], dim=-1)))
        t2 = self.s(self.gamma * self.thresh(torch.cat([x,-y.unsqueeze(1)], dim=-1)))

        b1 = batch_tensor_power(t1, 2)
        b2 = batch_tensor_power(t2, 2)
        print(t1)

        diff = (2 * y * self.main(x).squeeze()
            + F.linear(t1-t2, self.h).squeeze()
            + y * F.linear(t1+t2, self.u).squeeze()
            + self.q(b1-b2).squeeze()
            + 1
        )
        return self.r(diff) + torch.sum((self.gamma/100) * (1-t1**2)**2)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(16)

    _, _, correct = constraints(
        4,
        4,
        radius=1,
        degree=1,
        desired=(4,),
        included=(7,),
        ising=True
    )

    optimizer = torch.optim.SGD(
        params = model.parameters(),
        lr = 1e-4,
        weight_decay = 1e-5,
        momentum = 0.9,
        nesterov = True
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, verbose = True)

    for epoch in range(10000):

        model.setgamma(epoch/100)
        out = model(correct)
        loss = sum(out)

        print(out)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
