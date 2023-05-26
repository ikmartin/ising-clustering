
import torch
import torch.nn as nn
import torch.nn.functional as F

from ortools.linear_solver.pywraplp import Solver

from multiprocessing.managers import BaseManager
from multiprocessing import Process, Lock, freeze_support, Manager
import multiprocessing
from collections import deque
import time, os, torch, click
import numpy as np
from math import comb

from constraints import get_constraints
from oneshot import reduce_poly, MLPoly
from ising import IMul
from fittertry import IMulBit

from torch.utils.data import default_collate
from copy import deepcopy


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(512),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(1)
        )

    def forward(self, actions, current_bans: torch.Tensor, current_coeffs: torch.Tensor):
        num_actions = current_coeffs.shape[1]
        action_hot = F.one_hot(actions, num_actions).to(actions.device)
        input_tensor = torch.cat([action_hot, current_bans, current_coeffs], dim=-1).float()
        self.network = self.network.to(actions.device)
        return self.network(input_tensor)

class TrainerProcess(Process):
    def __init__(self, admin, device, model):
        self.admin = admin
        self.device = device
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters())
        self.loss_fn = nn.MSELoss().to(device)
        self.BATCH_SIZE = 64

        super().__init__()

    def run(self):
        #os.setpriority(os.PRIO_PROCESS, 0, -1)
        while True:
            self.train()
    
    def train(self):
        with self.admin['memory_lock']:
            memory = self.admin['memory']
            len_memory = len(memory)
            if not len_memory:
                return

            if len_memory <= self.BATCH_SIZE:
                batch_indices = np.arange(len_memory)
            else:
                batch_indices = np.random.choice(np.arange(len_memory), self.BATCH_SIZE)

            samples = [memory[i] for i in batch_indices]

        
        data = default_collate(samples)
        for key in data:
            data[key] = data[key].to(self.device)

        Y = torch.tensor([
            get_target(self.admin, self.model, bans, coeffs, reward, game_over)
            for bans, coeffs, reward, game_over in zip(data['new_bans'], data['new_coeffs'], data['reward'], data['game_over'])
        ]).float().to(self.device)

        with torch.autograd.detect_anomaly():
            with self.admin['model_lock']:
                self.model.train()

                X = self.model(data['action'], data['bans'], data['coeffs']).squeeze().to(self.device)
                print(X)
                print(Y)
                loss = self.loss_fn(X, Y)
                print(f'[{self.name}] loss: {loss}')
                if loss.item() > 1e6:
                    print(f"[{self.name}] warning: loss too high!")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.model.eval()


def best_action(admin, model, bans, coeffs):
    num_actions = coeffs.shape[0]
    ban_batch = bans.unsqueeze(0).expand(num_actions, -1)
    coeff_batch = coeffs.unsqueeze(0).expand(num_actions, -1)
    action_batch = torch.arange(num_actions).to(bans.device)

    with admin['model_lock']:
        qualities = model(action_batch, ban_batch, coeff_batch).squeeze(-1)

    qualities[coeffs == 0] = -100000    # No point banning something which is already 0! Actually we may want to revise this idea. But for now we should keep it because it keeps things faster. 
    
    return torch.argmax(qualities), torch.max(qualities)

def get_action(admin, model, bans, coeffs, epsilon):
    if np.random.uniform() < epsilon:
        nonzero_coeffs_indices = torch.nonzero(coeffs, as_tuple = True)[0]
        return np.random.choice(nonzero_coeffs_indices.cpu())

    global best_action
    
    action, quality = best_action(admin, model, bans, coeffs)
    return action.item()

def get_target(admin, model, bans, coeffs, reward, game_over):
    if game_over:
        return reward

    global best_action

    action, quality = best_action(admin, model, bans, coeffs)
    return reward + 0.9 * quality

def submit_memory(admin, entry):
    with admin['memory_lock']:
        admin['memory'].append(entry)
        while len(admin['memory']) > 2048:
            admin['memory'].pop(0)

class SolverProcess(Process):
    def __init__(self, admin, device, model, circuit, degree, constraints = None):
        self.admin = admin
        self.device = device
        self.model = model
        self.circuit = circuit
        self.degree = degree
        self.M, self.keys = constraints if constraints is not None else get_constraints(circuit, degree)

        self.num_terms = self.M.shape[1]
        self.threshold = 0.01
        self.epsilon = 0.3
        super().__init__()

    def run(self):
        print(f'[{self.name}] Building...')
        self.solver = self.build_solver()

        print(f'[{self.name}] Ready.')
        initial_bans = torch.zeros(self.num_terms)
        initial_coeffs, initial_score = self.solve(torch.tensor([]))
        
        if initial_coeffs is None:
            print(f'[{self.name}] Error! Infeasible initial problem!')
            return
        
        bans, coeffs, score = initial_bans, initial_coeffs, initial_score
        num_rounds = 0

        while True:
            game_over = False

            action = get_action(self.admin, self.model, bans.to(self.device), coeffs.to(self.device), self.epsilon)
            
            new_bans = bans.clone()
            new_bans[action] = 1
            which_bans = torch.nonzero(new_bans, as_tuple = True)[0]
            new_coeffs, new_score = self.solve(which_bans)
            
            if new_coeffs is None:
                new_coeffs = coeffs
                new_score = score
                game_over = True
                print(f'[{self.name}] eps {self.epsilon} round {num_rounds} score {score} bans {which_bans.tolist()}')

            memory_entry = {
                'bans': bans.numpy(),
                'coeffs': coeffs.numpy(), 
                'action': action, 
                'reward': score - new_score, 
                'new_bans': new_bans.numpy(),
                'new_coeffs': new_coeffs.numpy(),
                'game_over': game_over
            }

            submit_memory(self.admin, memory_entry)
            
            if game_over:
                coeffs = initial_coeffs
                bans = initial_bans
                score = initial_score
                num_rounds = 0
            else:
                coeffs = new_coeffs
                bans = new_bans
                score = new_score
                num_rounds += 1

            if self.epsilon > 0.01:
                self.epsilon -= 1e-4

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
        bans = bans.int().tolist()
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    num_threads = len(os.sched_getaffinity(0))
    print(f'[Master] Running on {device} with {num_threads} threads.')

    model = QNetwork().to(device)

    print('[Master] Getting constraints...')
    M, keys = get_constraints(circuit, degree)

    min_ban_length = 3
    min_ban_idx = len([
        key for key in keys
        if len(key) < min_ban_length
    ])

    print('[Master] Creating global manager.')
    manager = Manager()
    admin = {
        'memory_lock': manager.Lock(),
        'model_lock': manager.Lock(),
        'cache': manager.dict(),
        'memory': manager.list(),
    }

    print('[Master] Creating solvers...')
    workers = [
        SolverProcess(
            admin,
            device,
            model,
            circuit, 
            degree, 
            constraints = (M, keys),
        ) 
        for i in range(num_threads-1)
    ]

    print('[Master] Creating trainer...')
    trainer = TrainerProcess(admin, device, model)

    print('[Master] Initialized.')

    print('[Master] Starting processes.')
    trainer.start()
    for worker in workers:
        worker.start()

    trainer.join()
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
