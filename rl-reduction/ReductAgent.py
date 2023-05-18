from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.LazyLinear(1024)
        self.fc3 = nn.LazyLinear(512)
        self.fc4 = nn.LazyLinear(1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class ReductAgent:
    def __init__(
        self,
        n_episode=1000,
        gamma=1.0,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_log_decay=0.995,
        alpha=0.01,
        alpha_decay=0.01,
        batchsize=64,
    ):
        self.memory = deque(maxlen=1024)

        # initialize the model
        self.Q = DQN()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.01)


def step(self):
    
