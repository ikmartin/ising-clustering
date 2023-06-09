from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
import torch, os
import numpy as np
from ortools.linear_solver.pywraplp import Solver

from constraints import get_constraints
from oneshot import MLPoly
from ising import IMul
from spinspace import Spinspace

def solve(n1, n2, degree, aux_array, radius, threshold = 1e-2):
    circuit = IMul(n1, n2)
    circuit.set_all_aux(aux_array)
    M, keys = get_constraints(circuit, degree, radius = radius)
    num_terms = M.shape[1]
    solver = Solver.CreateSolver("GLOP")
    inf = solver.infinity()
    variables = [solver.NumVar(-inf, inf, f'x_{i}')
                      for i in range(num_terms)]
    
    for row in M:
        constraint = solver.Constraint(1.0, inf)
        for i, coeff in enumerate(row):
            if not coeff:
                continue
           
            constraint.SetCoefficient(variables[i], int(coeff))

    y_vars = [solver.NumVar(-inf, inf, f'y_{var}') for var in variables]
    for x, y in zip(variables, y_vars):
        constraint1 = solver.Constraint(-inf, 0)
        constraint2 = solver.Constraint(0, inf)
        constraint1.SetCoefficient(x, 1)
        constraint1.SetCoefficient(y, -1)
        constraint2.SetCoefficient(x, 1)
        constraint2.SetCoefficient(y, 1)

    objective = solver.Objective()
    for key, var in zip(keys, y_vars):
        if len(key) > 2:
            objective.SetCoefficient(var, 1)

    objective.SetMinimization()

    status = solver.Solve()

    print(f'{status} {aux_array}')
    return status

def main():
    n1 = 2
    n2 = 2
    aux_space = Spinspace(shape = (2 ** (n1 + n2) - 1,))

    n_jobs = len(os.sched_getaffinity(0))
    print(f'Running with {n_jobs} jobs...')
    with ProcessPoolExecutor(max_workers = n_jobs) as executor:
        futures = []
        for aux_array in aux_space:
            A = aux_array.binary()
            A = [[1] + A.tolist()]
            print(f'Submitting aux array {A}')
            futures.append(executor.submit(
                solve,
                n1, n2, 2, A, 1
            ))

        num_viable = 0

        for future in futures:
            coeffs = future.result()
            if not coeffs:
                num_viable += 1

        print(num_viable)

if __name__ == '__main__':
    main()




