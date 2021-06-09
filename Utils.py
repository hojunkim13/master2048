import logging
import numpy as np
import torch
from Environment.BitEnv import getTile


class MemoryBuffer:
    def __init__(self, mem_max, state_shape, action_shape):
        self.mem_max = mem_max
        self.action_shape = action_shape
        self.state_shape = state_shape

        self.S = np.zeros((mem_max,) + state_shape, dtype=float)
        self.P = np.zeros((mem_max, action_shape), dtype=float)
        self.Z = np.ones((mem_max, 1), dtype=float)
        self.mem_cntr = 0

    def stackMemory(self, s, p):
        idx = self.mem_cntr % self.mem_max
        self.S[idx] = s
        self.P[idx] = p
        self.mem_cntr += 1

    def adjustZ(self, step_size):
        last_idx = self.mem_cntr % self.mem_max
        
        decay = 0
        max_step = min(step_size, 100)
        for i in range(last_idx - max_step, last_idx):
            self.Z[i] -= 0.02 * decay
            decay += 1
        print("a")

    def getSample(self, n=1):
        max_idx = min(self.mem_cntr, self.mem_max)
        indice = np.random.randint(0, max_idx, size=n)
        S = self.S[indice]
        P = self.P[indice]
        Z = self.Z[indice]        
        return S, P, Z


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
    board = torch.zeros((16,4,4), dtype = int)
    for row in range(4):
        for col in range(4):
            power_of_tile = getTile(grid, row, col)            
            board[power_of_tile][row][col] = 1
    return board

logger = MCTSLogger()
