from ortools.linear_solver.pywraplp import Solver 
import torch
from time import perf_counter as pc

def build_solver(M, keys):
    """
    Builds a GLOP solver from a sparse constraint matrix.
    """

    start = pc()

    num_vars = M.shape[1]

    solver = Solver.CreateSolver("GLOP")
    inf = solver.infinity()
    variables = [
        solver.NumVar(-inf, inf, f"x_{i}") for i in range(num_vars)
    ]
    ban_constraints = [solver.Constraint(0, 0) for var in variables]
    

    constraint_vals = torch.t(torch.cat([M.indices(), M.values().unsqueeze(0)]))
    constraints = [solver.Constraint(1.0, inf) for _ in range(M.shape[0])]

    for x, y, val in constraint_vals:
        constraints[x].SetCoefficient(variables[y], int(val))

    y_vars = [solver.NumVar(-inf, inf, f"y_{var}") for var in variables]
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
    
    end = pc()
    print(f'Building solver took {end-start}')

    return solver, variables

