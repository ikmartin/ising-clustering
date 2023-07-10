import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import matplotlib.pyplot as plt
from new_constraints import constraints
from fast_constraints import batch_tensor_power, batch_vspin
from mysolver_interface import call_my_solver

torch.set_printoptions(threshold=50000)

class Model(nn.Module):
    def __init__(self, na):
        super().__init__()

        self.main = nn.LazyLinear(1, bias = False)
        self.thresh = nn.LazyLinear(na)
        self.s = nn.Tanh()
        self.r = nn.ReLU()
        self.gamma=1

    def setgamma(self, g):
        self.gamma = g

    def forward(self, inps):
        inps=inps.float()
        x = inps[..., :-1]
        y = inps[..., -1]

        i_right = torch.cat([x,y.unsqueeze(1)], dim=-1)
        i_wrong = torch.cat([x,-y.unsqueeze(1)], dim=-1)

        w1 = self.thresh(i_right)
        w2 = self.thresh(i_wrong)
        
        t1 = self.s(self.gamma * w1)
        t2 = self.s(self.gamma * w2)

        b1 = torch.cat([i_right, t1], dim=-1)
        b2 = torch.cat([i_wrong, t2], dim=-1)

        v1 = batch_vspin(b1, 2)
        v2 = batch_vspin(b2, 2)

        diff = self.main(v1-v2) + 1
        int_penalty = torch.sum((self.gamma/300) * ((1-t1**2)**2 + (1-t2**2)**2))
        neut_penalty = torch.sum((t2 * w2 - t1 * w1) ** 2)

        print(f'int_penalty = {int_penalty} neut_penalty = {neut_penalty}')
        """
        nearest_t1 = torch.sign(self.thresh(i_right))
        nearest_t2 = torch.sign(self.thresh(i_wrong))
        nearest_b1 = torch.cat([i_right, nearest_t1], dim=-1)
        nearest_b2 = torch.cat([i_wrong, nearest_t2], dim=-1)
        obj = call_my_solver((batch_vspin(nearest_b2, 2)-batch_vspin(nearest_b1, 2)).clone().detach().to_sparse_csc())
        print(f'obj = {obj}')
        """
        return diff, int_penalty, neut_penalty

def get_constraints(planes, ising):
    return constraints(
        4,
        4,
        radius=1,
        degree=1,
        desired=(4,),
        included=(7,),
        ising=ising,
        hyperplanes = planes
    )


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(12).to(device)

    _, _, correct = get_constraints(None, True)
    correct = correct.to(device)

    optimizer = torch.optim.SGD(
        params = model.parameters(),
        lr = 1e-4,
        weight_decay = 1e-5,
        momentum = 0.9,
        nesterov = True
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, verbose = True)

    constraint_loss = nn.Softplus()
    rho_loss = nn.ReLU().to(device)


    for epoch in range(100000):

        rho, int_penalty, neut_penalty = model(correct)
        closs = constraint_loss(rho)
        rhosum = sum(rho_loss(rho))
        loss = rhosum + int_penalty + neut_penalty

        model.setgamma(epoch/100)

        print(f'loss={loss} ai={rhosum}')
        
        if loss < 1e-6:
            print(model.thresh.weight)
            print(model.thresh.bias)
            planes = [(w,b.item()) for w,b in zip(model.thresh.weight.clone().detach(), model.thresh.bias)]
            M, _, _ = get_constraints(planes, True)
            objective = call_my_solver(M.to_sparse_csc())
            print(objective)
            exit()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()


if __name__ == '__main__':
    main()
