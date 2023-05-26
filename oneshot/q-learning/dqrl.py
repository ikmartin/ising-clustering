import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ortools.linear_solver.pywraplp import Solver

from collections import deque
import click

from constraints import get_constraint_matrix, make_poly
from oneshot import reduce_poly, MLPoly
from polyfit import gen_var_keys
from ising import IMul

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

    def forward(self, action_indices: torch.Tensor, current_bans: torch.Tensor, current_coeffs: torch.Tensor):
        action_tensor = F.one_hot(action_indices, current_bans.shape[-1])
        input_tensor = torch.cat([action_tensor, current_bans, current_coeffs], dim=1).float()
        return self.network(input_tensor)

class DQAgent():
    def __init__(self, device, emulator, epsilon, MAX_MEMORY_SIZE, BATCH_SIZE):
        self.device = device
        self.Q = QNetwork().to(device)
        self.Q.eval()
        self.memory = deque([], MAX_MEMORY_SIZE)
        self.BATCH_SIZE = BATCH_SIZE

        self.optimizer = torch.optim.SGD(
            self.Q.parameters(),
            lr = 1e-3,
            weight_decay = 1e-5,
            momentum = 0.9,
            nesterov = True
        )

        self.optimizer = torch.optim.Adam(
            self.Q.parameters()
        )


        self.discount = 1
        self.epsilon = epsilon
        
        self.emulator = emulator
        self.num_actions = self.emulator.num_terms + 1

        self.loss_fn = nn.MSELoss()

        self.cache = {}

        self.show_initial_answer()

        self.highscoretable = []

    def show_initial_answer(self):
        bans = torch.zeros(self.num_actions)
        coeffs, score = self.emulator_cache(bans)
        print(f'Initial score {score}')

    def best_action(self, bans, coeffs):
        ban_batch = bans.unsqueeze(0).expand(self.num_actions, -1).to(self.device)
        coeff_batch = coeffs.unsqueeze(0).expand(self.num_actions, -1).to(self.device)
        action_batch = torch.arange(self.num_actions).to(self.device).to(self.device)
        qualities = self.Q(action_batch, ban_batch, coeff_batch).squeeze(-1)
        qualities[coeffs == 0] = -100000    # No point banning something which is already 0! Actually we may want to revise this idea. But for now we should keep it because it keeps things faster. 
        
        return torch.argmax(qualities), torch.max(qualities)

    def get_action(self, bans, coeffs):
        if np.random.uniform() < self.epsilon:
            nonzero_coeffs_indices = torch.nonzero(coeffs, as_tuple = True)[0]
            return np.random.choice(nonzero_coeffs_indices)
        
        best_action, best_quality = self.best_action(bans, coeffs)
        return best_action

    def get_target(self, sample):
        if sample['new_coeffs'] is None:
            return sample['reward']

        best_action, best_quality = self.best_action(sample['new_bans'], sample['new_coeffs'])
        return sample['reward'] + self.discount * best_quality.detach().item()

    def train(self):
        if len(self.memory) <= self.BATCH_SIZE:
            batch_indices = np.arange(len(self.memory))
        else:
            batch_indices = np.random.choice(np.arange(len(self.memory)), self.BATCH_SIZE)

        samples = [self.memory[i] for i in batch_indices]
        
        actions = torch.tensor([
            sample['action']
            for sample in samples
        ]).to(self.device)

        bans = torch.cat([
            sample['bans'].unsqueeze(0)
            for sample in samples
        ]).to(self.device)

        coeffs = torch.cat([
            sample['coeffs'].unsqueeze(0)
            for sample in samples
        ]).to(self.device)

        
        Y = torch.tensor([
            self.get_target(sample)
            for sample in samples
        ]).float().to(self.device)
        
        with torch.autograd.detect_anomaly():
            self.Q.train()
            X = self.Q(actions, bans, coeffs).squeeze(-1)
            loss = self.loss_fn(X, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.Q.eval()

    def emulator_cache(self, bans):
        ban_key = tuple(bans.tolist())
        if ban_key in self.cache:
            return self.cache[ban_key]
        
        self.emulator.clear()
        for i, val in enumerate(bans):
            if val:
                self.emulator.ban(i)

        coeffs, score = self.emulator()
        self.cache[ban_key] = (coeffs, score)
        return coeffs, score

    def run_episode(self):
        bans = torch.zeros(self.num_actions)
        coeffs, score = self.emulator_cache(bans)
        game_running = True
        num_rounds = 0
        while game_running:
            action = self.get_action(bans, coeffs)
            action_index = action - 1
            new_bans = bans.clone().detach()

            term_name = ''.join([f'x_{i}' for i in self.emulator.keys[action]])
            print(f'ban {coeffs[action]:0.2f}{term_name}')
            # action -1 means choose to quit
            if action_index >= 0:
                new_bans[action_index] = 1
                new_coeffs, new_score = self.emulator_cache(new_bans)
                if new_coeffs is None or bans[action_index]:
                    new_coeffs = None
                    new_score = score + 5          # penalty of 1 for crashing
                    game_running = False
            else:
                # intentionally quit
                print('chose to quit')
                new_coeffs = None
                new_score = score
                game_running = False
            
            memory_entry = {
                'bans': bans,
                'coeffs': coeffs, 
                'action': action, 
                'reward': score - new_score, 
                'new_bans': new_bans,
                'new_coeffs': new_coeffs
            }
            self.memory.append(memory_entry)
            #print(memory_entry)

            self.train()

            if game_running:
                coeffs = new_coeffs
                score = new_score
                bans = new_bans

            num_rounds += 1

        self.highscoretable.append(score)
        highscores = sorted(self.highscoretable) if len(self.highscoretable) < 10 else sorted(self.highscoretable)[:10]
        print(f'Eps {self.epsilon} Game over round {num_rounds}: score = {score} highscores: {highscores}')
        if self.epsilon > 0.01:
            self.epsilon -= 1e-4


class BanGameEmulator():
    def __init__(self, circuit, degree):
        self.M = get_constraint_matrix(circuit, degree).int()
        self.num_terms = self.M.shape[1]
        self.threshold = 0.01
        self.circuit = circuit
        self.degree = degree
        self.keys = gen_var_keys(degree, circuit)
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


    def __call__(self):
        status = self.solver.Solve()
        if status:
            return None, None
        
        coeffs = torch.tensor([var.solution_value() for var in self.variables])
        coeffs[abs(coeffs) < self.threshold] = 0
        poly = self.get_poly(coeffs)
        
        reduced_poly = reduce_poly(poly, ['rosenberg'])
        num_aux = reduced_poly.num_variables() - self.circuit.G

        return coeffs, num_aux


@click.command()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    circuit = IMul(3,3)
    degree = 4

    emulator = BanGameEmulator(circuit, degree)
    agent = DQAgent(
        device = device,
        emulator = emulator,
        epsilon = 0.2,
        MAX_MEMORY_SIZE = 2048,
        BATCH_SIZE = 256
    )

    while True:
        agent.run_episode()



if __name__ == '__main__':
    main()
