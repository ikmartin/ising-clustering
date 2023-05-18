from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment1 import MLPolyEnv, ReductAlgo


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(512),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(1),
        )

    def forward(
        self,
    ):
        pass


class ReductAgent:
    def __init__(
        self,
        gamma=1.0,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_log_decay=0.995,
        alpha=0.01,
        alpha_decay=0.01,
        BATCH_SIZE=64,
        MAX_MEMORY_SIZE=1024,
        NUM_EPISODES=1024,
    ):
        # initialize the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = DQN().to(self.device)
        self.Q.eval()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.01)

        # hyperparameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_log_decay = 0.995

        self.memory = deque([], maxlen=MAX_MEMORY_SIZE)
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPISODES = NUM_EPISODES

        # initialize the environment
        self.env = MLPolyEnv(maxvar=100, maxaux=50, maxdeg=10)
        self.env.set_params(numvar=10, mindeg=0, sparsity=0.3, intcoeff=True)
        self.env.reset()


# processes action (reduct_method, C) into a tensor
def process_action(self):
    pass


# get the best action predicted by Q for the current state
def best_action(self, state):
    pass


# get an action in (reduct_method, C) format
def get_action(self, state):
    if np.random.uniform() < self.epsilon:
        pass


def train(self):
    # select training batch from memories
    if len(self.memory) <= self.BATCH_SIZE:
        batch_indices = np.arange(len(self.memory))
    else:
        batch_indices = np.random.choice(np.arange(len(self.memory)), self.BATCH_SIZE)

    # store the sampled memories
    samples = [self.memory[i] for i in batch_indices]


def run_episodes(self):
    for ep in range(self.NUM_EPISODES):
        # reset the environment to generate a new polynomial
        self.env.reset()
        reduced = False
        while self.env.done == False:
            # select action
            action = self.get_action(state=self.env.state())
            # execute action
            self.env.reduce(*action)
            pass
