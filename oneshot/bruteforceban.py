from multiprocessing.managers import BaseManager
from multiprocessing import Process, Lock, freeze_support, Manager
import multiprocessing
from collections import deque
import time, os, torch, click
import numpy as np
from math import comb

from ortools.linear_solver.pywraplp import Solver

from constraints import get_constraints
from oneshot import reduce_poly, MLPoly
from ising import IMul
from fittertry import IMulBit


def request_task(admin):
    with admin['lock']:
        if len(admin['queue']):
            task = admin['queue'].pop(0)
            admin['working'].append(task)
            return task
        
        return None
    
def submit(admin, task, coeffs, score, new_tasks):
    with admin['lock']:
        admin['working'].remove(task)
        admin['cache'][task] = (coeffs, score)

        for new_task in new_tasks:
            if new_task in admin['cache'] or new_task in admin['working'] or new_task in admin['queue']:
                continue

            admin['queue'].append(new_task)



def search(circuit, degree, num_workers):
    print('[Master] Getting constraints...')
    M, keys = get_constraints(circuit, degree)
    
    print('[Master] Creating global manager.')
    manager = Manager()
    admin = {
        'lock': manager.Lock(),
        'queue': manager.list(),
        'working': manager.list(),
        'cache': manager.dict()
    }

    print('[Master] Creating solvers...')
    workers = [
        SolverProcess(
            admin,
            circuit, 
            degree, 
            constraints = (M, keys),
            name = f'solve_process_{i}',
        ) 
        for i in range(num_workers)
    ]
    print('[Master] Initialized.')

    print('[Master] Starting processes.')
    for worker in workers:
        worker.start()

    print('[Master] Setting initial task.')
    with admin['lock']:
        admin['queue'].append(frozenset())

    print('[Master] Letting solvers work.')
    while True:
        with admin['lock']:
            queue_length = len(admin['queue'])
            in_progress_length = len(admin['working'])
            values = [val[1] for key, val in admin['cache'].items() if val[1] is not None]
            if len(values):
                best = min(values)
            else:
                best = -1

    
        print(f'[Master] {queue_length} tasks in queue and {in_progress_length} tasks in progress.')
        print(f'[Master] Best so far is {best}')
        time.sleep(2)


# 3x3 bitwise best
# 0 + 4 + 5 + 6 + 2 + 0
# +fgbz/fd
# 0 + 10 + 19 + 24 + 5 + 0

# 2x3 bitwise best
# 0 + 1 + 2 + 2 + 0
# fgbz/fd
# 0 + 3 + 6 + 5 + 0

# 4x4 bitwise best
# 1 + 11 + [16] + [23] + [23] + 6 + 2 + 0


class SolverProcess(Process):
    def __init__(self, admin, circuit, degree, constraints = None, name = None):
        self.admin = admin
        self.circuit = circuit
        self.degree = degree
        self.M, self.keys = constraints if constraints is not None else get_constraints(circuit, degree)
        self.min_ban_number = circuit.G + comb(circuit.G, 2)

        self.name = name if name is not None else 'solver'

        self.num_terms = self.M.shape[1]
        self.threshold = 0.01
        super().__init__()

    def run(self):
        print(f'[{self.name}] Building...')
        self.solver = self.build_solver()

        print(f'[{self.name}] Ready.')
        # Run an infinite loop waiting for tasks to be added to the parent queue
        while True:
            task = request_task(self.admin)
            if task is None:
                time.sleep(0.1)
                continue

            coeffs, score = self.solve(task)
            print(f'[{self.name}] Task {task}: {score}')
            new_tasks = self._make_new_tasks(task, coeffs)
            submit(self.admin, task, coeffs, score, new_tasks)

    def _make_new_tasks(self, task, coeffs):
        return [
            task | frozenset([i.item()])
            for i in torch.nonzero(coeffs, as_tuple = True)[0]
            if i.item() >= self.min_ban_number
        ] if coeffs is not None else []

    def _clear(self):
        for ban_constraint in self.ban_constraints:
            ban_constraint.Clear()

    def _ban(self, index):
        self.ban_constraints[index].SetCoefficient(self.variables[index], 1)

    def build_solver(self):
        solver = Solver.CreateSolver("GLOP")
        inf = solver.infinity()
        self.variables = [solver.NumVar(-inf, inf, f'x_{i}')
                          for i in range(self.num_terms)]
        self.ban_constraints = [solver.Constraint(0,0)
                                for var in self.variables]
        
        for row in self.M:
            constraint = solver.Constraint(1.0, inf)
            for i, coeff in enumerate(row):
                if not coeff:
                    continue
               
                constraint.SetCoefficient(self.variables[i], int(coeff))
    
        y_vars = [solver.NumVar(-inf, inf, f'y_{var}') for var in self.variables]
        for x, y in zip(self.variables, y_vars):
            constraint1 = solver.Constraint(-inf, 0)
            constraint2 = solver.Constraint(0, inf)
            constraint1.SetCoefficient(x, 1)
            constraint1.SetCoefficient(y, -1)
            constraint2.SetCoefficient(x, 1)
            constraint2.SetCoefficient(y, 1)

        objective = solver.Objective()
        for key, var in zip(self.keys, y_vars):
            if len(key) > 2:
                objective.SetCoefficient(var, 1)

        objective.SetMinimization()

        return solver

    def get_poly(self, coeffs) -> MLPoly:
        coeff_dict = {
            key: val
            for key, val in zip(self.keys, coeffs)
        }
        return MLPoly(coeff_dict)


    def solve(self, bans: tuple):
        self._clear()
        for i in bans:
            self._ban(i)


        status = self.solver.Solve()
        if status:
            return None, None
        
        coeffs = torch.tensor([var.solution_value() for var in self.variables])
        coeffs[abs(coeffs) < self.threshold] = 0
        poly = self.get_poly(coeffs.tolist())
        
        reduced_poly = reduce_poly(poly, ['rosenberg'])
        num_aux = reduced_poly.num_variables() - self.circuit.G

        return coeffs, int(num_aux)



@click.command()
@click.option('--n1', default = 2, help = 'First input')
@click.option('--n2', default = 2, help = 'Second input')
@click.option('--degree', default = 3, help = 'Degree')
@click.option('--bit', default = None, help = 'Select an output bit.')
def main(n1, n2, degree, bit):
    if bit is None:
        circuit = IMul(n1,n2)
    else:
        circuit = IMulBit(n1,n2,int(bit))

    num_workers = os.cpu_count()

    search(circuit, degree, num_workers)

if __name__ == '__main__':
    main()
