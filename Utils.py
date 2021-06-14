import logging
import numpy as np
import torch
from Environment.BitEnv import getTile
from collections import deque
from itertools import islice

class MemoryBuffer:
    def __init__(self, mem_max):
        self.S = deque(maxlen=mem_max)        
        self.Z = deque(maxlen=mem_max)
        self.Z_temp = deque(maxlen=mem_max)

    def stackMemory(self, s, z):
        self.S.append(s)        
        self.Z_temp.append(z)

    def convert(self, seq : list, gamma : float = 0.99):
        running_add = 0
        out = seq.copy()
        for i in reversed(range(len(out))):
            v = seq[i] + running_add * gamma
            out[i] = running_add = v
        return out

    def updateZ(self, n_step : int = 100):                
        # for i in range(len(Z)):
        #     value = 0
        #     decay = 1
        #     temp = list(self.Z_temp)[i:i+n_step]
        #     for v in temp:
        #         value += decay * v
        #         decay *= 0.99
        #     Z[i] = value
        val = (1 - 0.99 ** n_step) / (1 - 0.99)
        length = len(self.Z_temp)
        Z = [val] * length
        decay = 1
        for i in range(length - 100, length):
            Z[i] *= decay
            decay *= 0.99            
        self.Z += deque(Z)
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
