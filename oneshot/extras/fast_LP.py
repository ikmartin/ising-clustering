from fast_constraints import fast_constraints
import os, subprocess
import torch
import pandas as pd
import numpy as np

from ising import IMul
from oneshot import MLPoly

SOLVER_PATH = '../LPsparse/LPsparse'

def sparse_solve(circuit, degree):
    problem_name = "UNIQUE_NAME"

    if not os.path.exists('problems'):
        os.mkdir('problems')

    problem_path = os.path.abspath(f'problems/{problem_name}')
    if not os.path.exists(problem_path):
        os.mkdir(problem_path)

    constraints, keys = fast_constraints(circuit, degree)

    num_variables = len(keys)
    num_y = num_variables

    sparse_data = torch.cat([constraints.indices() + 1, -constraints.values().unsqueeze(0)])
    sparse_data[0] += 2 * num_y # Move the real constraints down to make room for y constraints
    sparse_data[1] += num_y
    sparse_data = torch.t(sparse_data)

    y_constraints = torch.cat([
        torch.tensor([
            [i + 1, i + 1, -1],
            [i + 1, num_y + i + 1, -1],
        ])
        for i in range(num_y)
    ] + [
        torch.tensor([
            [num_y + i + 1, i + 1, -1],
            [num_y + i + 1, num_y + i + 1, 1]
        ])
        for i in range(num_y)
    ])

    sparse_data = torch.cat([y_constraints, sparse_data])

    print(sparse_data)
    pd.DataFrame(
        sparse_data.numpy(),
        columns = [str(constraints.shape[0] + 2*num_y), str(num_variables + num_y ), "0.0"]
    ).to_csv(
        os.path.join(problem_path, 'A'),
        index = False,
        sep = '\t'
    )

    
    with open(os.path.join(problem_path, 'Aeq'), 'w') as FILE:
        FILE.write(f'0\t{num_variables + num_y}\t0.0')

    with open(os.path.join(problem_path, 'b'), 'w') as FILE:
        for i in range(2 * num_y):
            FILE.write('0\n')
        
        for i in range(constraints.shape[0]):
            FILE.write('-1\n')

    with open(os.path.join(problem_path, 'beq'), 'w') as FILE:
        pass
    
    with open(os.path.join(problem_path, 'c'), 'w') as FILE:
        for i, key in zip(range(num_y), keys):
            if len(key) > 2:
                FILE.write('1\n')
            else:
                FILE.write('0\n')

        for i in range(num_variables):
            FILE.write('0\n')
    
    with open(os.path.join(problem_path, 'meta'), 'w') as FILE:
        FILE.write(f'nb\t{num_y}\n')
        FILE.write(f'nf\t{num_variables}\n')
        FILE.write(f'mI\t{constraints.shape[0] + 2 * num_y}\n')
        FILE.write(f'mE\t0\n')

    readout = subprocess.run([SOLVER_PATH, '../oneshot/problems/UNIQUE_NAME'])
    print(f'return code {readout}')
    print(readout.returncode)
    if readout.returncode == 1:
        print('infeasible')
        return None

    x, V = np.loadtxt(os.path.join(problem_path, 'sol'), unpack = True)
    V = V[x > num_y]
    x = x.astype(int)[x > num_y] - num_y - 1
    return MLPoly({
        keys[i]: val
        for i, val in zip(x, V)
    })


