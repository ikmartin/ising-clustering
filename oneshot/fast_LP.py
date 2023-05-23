from fast_constraints import fast_constraints
import os, subprocess
import torch
import pandas as pd

from ising import IMul

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

    sparse_data = torch.t(torch.cat([constraints.indices() + 1, -constraints.values().unsqueeze(0)]))
    print(sparse_data)
    pd.DataFrame(
        sparse_data.numpy(),
        columns = [str(constraints.shape[0]), str(num_variables), "0.0"]
    ).to_csv(
        os.path.join(problem_path, 'A'),
        index = False,
        sep = '\t'
    )

    with open(os.path.join(problem_path, 'Aeq'), 'w') as FILE:
        FILE.write(f'0\t{num_variables}\t0.0')

    with open(os.path.join(problem_path, 'b'), 'w') as FILE:
        for i in range(num_variables):
            FILE.write('-1\n')

    with open(os.path.join(problem_path, 'beq'), 'w') as FILE:
        pass
    
    with open(os.path.join(problem_path, 'c'), 'w') as FILE:
        for i in range(num_variables):
            FILE.write('0\n')
    
    with open(os.path.join(problem_path, 'meta'), 'w') as FILE:
        FILE.write('nb\t0\n')
        FILE.write(f'nf\t{num_variables}\n')
        FILE.write(f'mI\t{constraints.shape[0]}\n')
        FILE.write(f'mE\t0\n')

    readout = subprocess.run([SOLVER_PATH, '../oneshot/problems/UNIQUE_NAME'])
    #print(readout.stdout)


circuit = IMul(2,2)
degree = 3

sparse_solve(circuit, degree)

