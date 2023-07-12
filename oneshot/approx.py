import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import matplotlib.pyplot as plt
from new_constraints import constraints, make_correct_rows, make_wrongs
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

    def forward(self, right, wrong, wrong_per_input):
        w1 = self.thresh(right)
        w2 = self.thresh(wrong)
        
        t1 = self.s(self.gamma * w1)
        t2 = self.s(self.gamma * w2)

        b1 = torch.cat([right, t1], dim=-1)
        b2 = torch.cat([wrong, t2], dim=-1)

        v1 = batch_vspin(b1, 2)
        v2 = batch_vspin(b2, 2)

        v1_exp = (
            v1.unsqueeze(1)
            .expand(-1, wrong_per_input, -1)
            .reshape(wrong.shape[0], -1)
        )

        right_thresh_energies = (
            (t1*w1).unsqueeze(1)
            .expand(-1, wrong_per_input, -1)
            .reshape(wrong.shape[0], -1)
        )

        rhosum = torch.sum(self.r(self.main(v1_exp - v2) + 1))
        int_penalty = (self.gamma / 300) * (torch.sum((1-t1**2)**2) + torch.sum((1-t2**2)**2))
        neutral_penalty = torch.sum(self.r(t2*w2 - right_thresh_energies) ** 2)
        return rhosum, int_penalty, neutral_penalty

    def forward_bit(self, inps):
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
        # Quadratic penalty: strong neutralizability
        #neut_penalty = torch.sum((t2 * w2 - t1 * w1) ** 2)
        neut_penalty = torch.sum(self.r(t2 * w2 - t1 * w1) ** 2)

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

    correct, num_fixed, num_free = make_correct_rows(4, 4, included = (7,), desired = (0,1,2,3,4,5,6))
    wrong, rows_per_input = make_wrongs(correct, num_fixed, num_free, None)
    correct, wrong = correct.float().to(device), wrong.float().to(device)
    correct = (correct * 2) - 1
    wrong = (wrong * 2) - 1

    optimizer = torch.optim.SGD(
        params = model.parameters(),
        lr = 1e-4,
        weight_decay = 1e-5,
        momentum = 0.9,
        nesterov = True
    )

    #optimizer = torch.optim.Adam(model.parameters())

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, verbose = True)


    for epoch in range(100000):

        rho, int_penalty, neut_penalty = model(correct, wrong, rows_per_input)
        loss = rho + int_penalty# + neut_penalty

        model.setgamma(epoch/100)

        print(f'loss={loss.item():.6f} ai={rho.item():.2f} int_pen={int_penalty.item():.2f} neut_pen={neut_penalty.item():.2f}')
        
        if loss < 1e-6:
            planes = [(w,-b.item()) for w,b in zip(model.thresh.weight.clone().detach().cpu(), model.thresh.bias)]
            print(planes)
            exit()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()


if __name__ == '__main__':
    main()
