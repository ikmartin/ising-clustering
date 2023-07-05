from new_constraints import constraints
from mysolver_interface import call_my_solver
from itertools import product, combinations
from random import choice, choices, randint
import torch
import torch.nn.functional as F
from copy import deepcopy
from ast import literal_eval
import numpy as np
from tqdm import tqdm
from fast_constraints import all_answers_basic
import pandas as pd
import matplotlib.pyplot as plt

n1, n2 = 3,3
desired = (0,1,2,3,4)
included = (5,)
num_aux = 4
num_inputs = n1 + n2 + (len(included) if included is not None else 0)
num_outputs = (len(desired) if desired is not None else n1 + n2)
num_vars = num_inputs + num_outputs

X = []
Y = []

for sigma in torch.linspace(0.0001,0.005,10):
    loop = tqdm(range(20), leave=True)
    for i in loop:
        weights = torch.randn(num_vars)
        weights = F.normalize(weights, dim=0)
        bias = torch.randn(1).item()
        M, _, correct = constraints(n1, n2, desired = desired, included = included, hyperplanes = [(weights, bias)])
        
        objective = call_my_solver(M.to_sparse_csc())
        for j in range(10):
            new_weights = sigma * torch.randn(num_vars) + weights
            new_weights = F.normalize(new_weights, dim = 0)
            new_bias = (sigma * torch.randn(1).item() + bias)
            M, _, correct = constraints(n1, n2, desired = desired, included = included, hyperplanes = [(new_weights, new_bias)])
            new_objective = call_my_solver(M.to_sparse_csc())
            X.append(sigma)
            Y.append(objective - new_objective)

plt.scatter(X, Y, alpha = 0.2)
plt.savefig('landscape.png')


