import logging
import numpy as np
import torch
from Environment.BitEnv import getTile, getFreeTile
from collections import deque
from itertools import islice

class MemoryBuffer:
    def __init__(self, mem_max):
        self.S = deque(maxlen=mem_max)        
        self.Z = deque(maxlen=mem_max)
        self.Z_temp = []

    def stackMemory(self, s, z):
        self.S.append(s)        
        self.Z_temp.append(z)

    def getNstepvalue(self, seq : list, gamma : float = 0.95):
        out = seq.copy()
        running_add = 1
        value = 0
        for i in range(len(seq)):
            value += seq[i] * running_add
            running_add *= gamma
        return value

    def updateZ(self, n_step : int = 10, gamma : float = 0.95):
        Z_temp = []
        for i in range(len(self.Z_temp)):
            value = self.getNstepvalue(self.Z_temp[i:i+n_step], gamma)
            Z_temp.append(value)                        
        self.Z += deque(Z_temp)
        self.Z_temp.clear()

    def getSample(self, n=1):        
        indice = np.random.randint(0, len(self.Z), size=n)
        
        S = torch.stack(list(self.S))[indice]    
        Z = torch.tensor(self.Z)[indice]        
        return S, Z

class MCTSLogger(logging.Logger):
    def __init__(self, name="MCTS"):
        super(MCTSLogger, self).__init__(name)
        self.setLevel(logging.INFO)
        self.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler("MCTS.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

    def reset_handler(self, episode=None):
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%y/%m/%d"
        )
        file_handler = logging.FileHandler(f"MCTS_{episode}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        if len(self.handelrs) == 0:
            self.addHandler(file_handler)
        else:
            self.handlers[0] = file_handler


def preprocessing(grid):
    """
    preprocessing function for neural network
    input : 64-bit integer grid
    output : (16, 4, 4) shape tensor state
    """
    board = torch.zeros((16, 4, 4), dtype=float)
    for row in range(4):
        for col in range(4):
            power_of_tile = getTile(grid, row, col)
            board[power_of_tile][row][col] = 1
    return board


logger = MCTSLogger()
