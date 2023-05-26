from ortools.linear_solver.pywraplp import Solver 
import torch
from time import perf_counter as pc
from numba import jit
import numpy as np

class LPWrapper:
    def __init__(self, M, keys):
        self.keys = keys
        self.threshold = 1e-2
        self.solver, self.variables, self.bans = build_solver(M, keys)
        
    def _clear(self):
        for ban in self.bans:
            ban.Clear()

    def _ban(self, index):
        self.bans[index].SetCoefficient(self.variables[index], 1)

    def solve(self, bans = None):
        self._clear()
        if bans is not None:
            for ban in bans:
                self._ban(ban)

        status = self.solver.Solve()
        if status == 2:     # Infeasible
            return None
            
        answer = np.array([var.solution_value() for var in self.variables])
        answer[abs(answer) < self.threshold] = 0
        return answer

def build_solver(M, keys):
    """
    Builds a GLOP solver from a sparse constraint matrix.
    """

    M = M.to_sparse()
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
    
    return solver, variables, ban_constraints

