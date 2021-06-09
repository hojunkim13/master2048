from Network.Network import Network
from torch.optim import Adam
from MCTS.PUCT import MCTS
import torch
import numpy as np
import random
from Utils import logger, MemoryBuffer


class Agent:
    def __init__(self, state_dim, action_dim, lr, batch_size, n_sim, maxlen):
        self.net = Network(state_dim, action_dim)
        self.optimizer = Adam(
            self.net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.8, 0.999)
        )
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=[50, 100, 150, 200, 250, 300, 400], gamma=0.77
        # )
        self.n_sim = n_sim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.mcts = MCTS(self.net)
        self.memory = MemoryBuffer(maxlen, state_dim, action_dim)
        self.step = 0

    def getAction(self, grid):
        visits = self.mcts.getAction(grid, self.n_sim)
        probs = [p/sum(visits) for p in visits]
        if self.step <= 100:
            action = np.random.choice(range(4), p = probs)
        else:
            action = np.argmax(visits)
        self.step += 1
        return action, probs

    def storeTranstion(self, state, probs):
        self.memory.stackMemory(state, probs)        

    # def pushMemory(self):
    #     length = len(self.tmp_state_memory)
    #     if length <= 50:
    #         self.tmp_state_memory.clear()
    #         return

    #     step_rewards = deque([1] * (length - 50))
    #     step_rewards += deque([-1] * 50)
    #     self.state_memory += self.tmp_state_memory
    #     self.reward_memory += step_rewards
    #     self.tmp_state_memory.clear()

    def learn(self):        
        if self.memory.mem_cntr < self.batch_size:
            return 0

        S, P, Z = self.memory.getSample(self.batch_size)
        S = torch.tensor(S, dtype=torch.float).cuda().reshape(-1, *self.state_dim)
        P = torch.tensor(P, dtype=torch.float).cuda().reshape(-1, self.action_dim)
        Z = torch.tensor(Z, dtype=torch.float).cuda().reshape(-1, 1)
        
        policy, value = self.net(S)        
        value_loss = torch.square(value - Z).mean()
        policy_loss = (-P * torch.log(policy)).mean()
        total_loss = value_loss + policy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def save(self, env_name):
        torch.save(self.net.state_dict(), f"./data/model/{env_name}_2048zero.pt")

    def load(self, env_name):
        state_dict = torch.load(f"./data/model/{env_name}_2048zero.pt")
        self.net.load_state_dict(state_dict)
