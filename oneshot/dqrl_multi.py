import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import default_collate

from ortools.linear_solver.pywraplp import Solver

from collections import deque
import click
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing.managers import BaseManager
import threading


from constraints import get_constraint_matrix, make_poly
from oneshot import reduce_poly, MLPoly
from polyfit import gen_var_keys
from ising import IMul

FAILURE_SCORE = -1

class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()

        self.num_actions = num_actions
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
        action_hot = F.one_hot(actions, self.num_actions)
        input_tensor = torch.cat([action_hot, current_bans, current_coeffs], dim=-1).float()
        return self.network(input_tensor)

class DQAgent():
    def __init__(self, device, circuit, degree, epsilon, MAX_MEMORY_SIZE, BATCH_SIZE, NUM_PLAYERS = 1):
        self.device = device
        self.circuit = circuit
        self.degree = degree
        self.global_emulator = BanGameEmulator(circuit, degree)
        self.global_emulator.setup()
        self.num_actions = self.global_emulator.num_terms
        self.Q = QNetwork(self.num_actions).to(device)
        self.Q.eval()
        self.memory = deque([], MAX_MEMORY_SIZE)
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_PLAYERS = NUM_PLAYERS


        self.optimizer = torch.optim.Adam(
            self.Q.parameters(),
            weight_decay = 1e-5
        )

        self.discount = 1
        self.epsilon = epsilon
        
        self.loss_fn = nn.MSELoss()

        self.cache = {}

        self.highscores = []

        

    def best_actions(self, bans_batch, coeffs_batch):
        bans_batch = bans_batch.to(self.device)
        coeffs_batch = coeffs_batch.to(self.device)

        qualities = torch.cat([
            self.Q(
                torch.arange(self.num_actions).to(self.device),
                bans_batch[i].unsqueeze(0).expand(self.num_actions, -1),
                coeffs_batch[i].unsqueeze(0).expand(self.num_actions, -1)
            ).squeeze().unsqueeze(0)
            for i in range(bans_batch.shape[0])
        ])

        qualities[coeffs_batch == 0] = -100000    # No point banning something which is already 0! Actually we may want to revise this idea. But for now we should keep it because it keeps things faster.
        
        maxima = torch.max(qualities, dim = -1)
        return maxima.indices, maxima.values

    def get_actions(self, bans, coeffs):
        bans = bans.to(self.device)
        coeffs = coeffs.to(self.device)
        nonzero_coeffs_indices = [
            torch.nonzero(vec, as_tuple = True)[0]
            for vec in coeffs
        ]

        random_nonzero_indices = torch.tensor([
            np.random.choice(choices.cpu())
            for choices in nonzero_coeffs_indices
        ]).to(self.device)
        
        best_actions, best_qualities = self.best_actions(bans, coeffs)
        best_actions = best_actions.squeeze(-1)

        random_action_mask = torch.bernoulli(torch.ones(coeffs.shape[0]) * self.epsilon).bool().to(self.device)
        best_actions[random_action_mask] = random_nonzero_indices[random_action_mask]
        return best_actions

    def update_beliefs(self, samples):
        data = default_collate(samples)
        for key in data:
            data[key] = data[key].to(self.device)

        Y = data['reward'].float()

        running = ~data['game_over']
        if torch.any(running):
            running_bans = data['new_bans'][running]
            running_coeffs = data['new_coeffs'][running]
            best_actions, best_qualities = self.best_actions(running_bans, running_coeffs)
            Y[running] += self.discount * best_qualities.clone().detach()

        
        with torch.autograd.detect_anomaly():
            self.Q.train()

            X = self.Q(data['action'], data['bans'], data['coeffs']).squeeze()
            loss = self.loss_fn(X, Y)
            if loss.item() > 1e6:
                print("warning: loss too high!")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.Q.eval()


    def train(self):
        if len(self.memory) <= self.BATCH_SIZE:
            batch_indices = np.arange(len(self.memory))
        else:
            batch_indices = np.random.choice(np.arange(len(self.memory)), self.BATCH_SIZE)

        samples = [self.memory[i] for i in batch_indices]

        self.update_beliefs(samples)


    def emulator_cache(self, emulator, bans):
        ban_key = tuple(bans.tolist())
        if ban_key in self.cache:
            return self.cache[ban_key]
        
        emulator.clear()
        for i, val in enumerate(bans):
            if val:
                emulator.ban(i)

        coeffs, score = emulator()
        self.cache[ban_key] = (coeffs, score)

        return coeffs, score

    def run(self):
        self.create_emulators()
        initial_bans = torch.zeros(self.NUM_PLAYERS, self.num_actions)
        initial_coeffs, initial_scores = self.run_emulators(initial_bans)
        bans, coeffs, scores = initial_bans, initial_coeffs, initial_scores

        while True:
            bans, coeffs, scores, game_over = self.play_round(bans, coeffs, scores)

            # Handle game-over states
            final_scores = scores[game_over].tolist()
            print(f'eps {self.epsilon} scores: {final_scores}')
            self.highscores += final_scores
            
            coeffs[game_over] = initial_coeffs[game_over]
            scores[game_over] = initial_scores[game_over]
            bans[game_over] = initial_bans[game_over]
            
            if self.epsilon > 0.01:
                self.epsilon *= 0.95

    def run_emulators(self, bans):
        futures = []
        with ProcessPoolExecutor(max_workers = self.NUM_PLAYERS) as executor:
            for i, emulator in enumerate(self.emulators):
                futures.append((executor.submit(emulator.run, bans[i]), i))

        new_coeffs = []
        new_scores = []
        for future, i in futures:
            c, s = future.result()
            new_coeffs.append(c.unsqueeze(0))
            new_scores.append(s)

        new_coeffs = torch.cat(new_coeffs)
        new_scores = torch.tensor(new_scores)
        return new_coeffs, new_scores

    def play_round(self, bans, coeffs, scores):
        actions = self.get_actions(bans, coeffs).cpu()

        new_bans = torch.clamp(
            bans.clone().detach()
            + F.one_hot(actions, bans.shape[1]),
            max = 1
        )

        # RUN EMULATORS TO GET NEW COEFFS, NEW SCORE
        new_coeffs, new_scores = self.run_emulators(new_bans)

        game_over = new_scores == FAILURE_SCORE
        new_scores[game_over] = scores[game_over] # Reward of 0 for crashing

        #term_name = ''.join([f'x_{i}' for i in emulator.keys[action_index]])
        #print(f'[{thread_name}] ban {coeffs[action_index]:0.2f}{term_name}')

        for i in range(self.NUM_PLAYERS):
            memory_entry = {
                'bans': bans[i],
                'coeffs': coeffs[i], 
                'action': actions[i], 
                'reward': scores[i] - new_scores[i], 
                'new_bans': new_bans[i],
                'new_coeffs': new_coeffs[i],
                'game_over': game_over[i]
            }

            self.memory.append(memory_entry)
            self.train()
        
        return new_bans, new_coeffs, new_scores, game_over

    def create_emulators(self):
        print('Initializing emulators...')
        self.managers = []
        self.emulators = []
        for i in range(self.NUM_PLAYERS):
            manager = GameManager()
            manager.start()
            self.managers.append(manager)
            self.emulators.append(manager.Emulator(self.circuit, self.degree, M=self.global_emulator.M, keys = self.global_emulator.keys))

        print('Running setup...')
        futures = []
        with ProcessPoolExecutor(max_workers = self.NUM_PLAYERS) as executor:
            for emulator in self.emulators:
                futures.append(executor.submit(emulator.setup))

        for future in futures:
            future.result()

        print('Emulator setup complete.')

        
class BanGameEmulator():
    def __init__(self, circuit, degree, M = None, keys = None):
        self.M = M
        self.keys = keys
        self.circuit = circuit
        self.degree = degree

    def setup(self):
        self.M = get_constraint_matrix(self.circuit, self.degree).int() if self.M is None else self.M
        self.num_terms = self.M.shape[1]
        self.threshold = 0.01
        self.keys = gen_var_keys(self.degree, self.circuit) if self.keys is None else self.keys
        self.solver = self.build_solver()

    def clear(self):
        for ban_constraint in self.ban_constraints:
            ban_constraint.Clear()

    def ban(self, index):
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

    def get_poly(self, coeffs: torch.Tensor) -> MLPoly:
        coeff_dict = {
            key: val
            for key, val in zip(self.keys, coeffs)
        }
        return MLPoly(coeff_dict)

    def run(self, bans):
        self.clear()
        for i, val in enumerate(bans):
            if val:
                self.ban(i)

        status = self.solver.Solve()
        if status:
            return torch.zeros(self.num_terms), FAILURE_SCORE
        
        coeffs = torch.tensor([var.solution_value() for var in self.variables])
        coeffs[abs(coeffs) < self.threshold] = 0
        poly = self.get_poly(coeffs)
        
        reduced_poly = reduce_poly(poly, ['rosenberg'])
        num_aux = reduced_poly.num_variables() - self.circuit.G

        return coeffs, num_aux


"""
Set up the multiprocessing manager class
"""
class GameManager(BaseManager): pass
GameManager.register('Emulator', BanGameEmulator)

@click.command()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_threads = len(os.sched_getaffinity(0))
    print(f'Running on {device} with {num_threads} threads.')
    circuit = IMul(3,3)
    degree = 4

    agent = DQAgent(
        device = device,
        circuit = circuit,
        degree = degree,
        epsilon = 0.9,
        MAX_MEMORY_SIZE = 2048,
        BATCH_SIZE = 256,
        NUM_PLAYERS = 8
    )

    agent.run()


if __name__ == '__main__':
    main()
