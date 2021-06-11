from Network.Network import Network
from torch.optim import Adam
from MCTS.UCT import MCTS
import torch
import numpy as np
import pickle
from Utils import logger, MemoryBuffer


class Agent:
    def __init__(self, state_dim, action_dim, lr, batch_size, n_sim):
        self.net = Network(state_dim, action_dim)
        self.optimizer = Adam(
            self.net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.8, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5,
        )
        self.n_sim = n_sim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.mcts = MCTS(self.net)
                

    def getAction(self, grid):
        visits = self.mcts.getAction(grid, self.n_sim)
        probs = [p / sum(visits) for p in visits]
        if self.step <= 50:
            action = np.random.choice(range(4), p=probs)
        else:
            action = np.argmax(visits)        
        return action
    
    def learn(self, memory):
        if len(memory.Z) < self.batch_size:
            return 0

        S, Z = memory.getSample(self.batch_size)
        S = S.float().cuda().reshape(-1, *self.state_dim)        
        Z = Z.float().cuda().reshape(-1, 1)
        
        _, value = self.net(S)
        value_loss = torch.square(value - Z).mean()        
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return value_loss.item()

    def save(self, env_name):
        torch.save(self.net.state_dict(), f"./data/model/{env_name}_2048zero.pt")

    def load(self, env_name):
        state_dict = torch.load(f"./data/model/{env_name}_2048zero.pt")
        self.net.load_state_dict(state_dict)
