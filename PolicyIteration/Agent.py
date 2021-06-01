import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PolicyIteration.Network import Network
from torch.optim import Adam
from Environment.Utils import *
from MCTS.MCTS_ValueNetwork import MCTS
import torch
import numpy as np
import random
from Logger import logger
from collections import deque


class Agent:
    def __init__(self, state_dim, action_dim, lr, batch_size, n_sim, maxlen):
        self.net = Network(state_dim, action_dim)
        self.optimizer = Adam(
            self.net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.8, 0.999)
        )
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
        self.n_sim = n_sim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.mcts = MCTS(self.net)
        self.state_memory = deque(maxlen=maxlen)
        self.reward_memory = deque(maxlen=maxlen)
        self.tmp_state_memory = deque()

    def getAction(self, grid):
        action = self.mcts.getAction(grid, self.n_sim)        
        return action

    def storeTranstion(self, state):
        #state = transition[0]        
        self.tmp_state_memory.append(state)        

    def pushMemory(self):    
        length = len(self.tmp_state_memory)
        if length <= 50:
            self.tmp_state_memory.clear()
            return
        
        step_rewards = deque([1] * (length-50))
        step_rewards += deque([-1] * 50)        
        self.state_memory += self.tmp_state_memory
        self.reward_memory += step_rewards
        self.tmp_state_memory.clear()

    def learn(self):
        idx_max = len(self.state_memory)
        if idx_max <= self.batch_size:
            return
        
        indice = random.sample(range(idx_max), idx_max)
        states = np.array(self.state_memory, dtype=np.float32)[indice]
        rewards = np.array(self.reward_memory, dtype=np.float32)[indice]

        S = torch.tensor(states, dtype=torch.float).cuda().reshape(-1, *self.state_dim)
        value = self.net(S)
        outcome = torch.tensor(rewards, dtype=torch.float).cuda().view(*value.shape)
        value_loss = torch.square(value - outcome).mean()

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        return value_loss.item()

    def save(self, env_name):
        torch.save(self.net.state_dict(), f"./data/model/{env_name}_2048zero.pt")

    def load(self, env_name):
        state_dict = torch.load(f"./data/model/{env_name}_2048zero.pt")
        self.net.load_state_dict(state_dict)
