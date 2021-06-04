import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import time

from Environment.BitEnv import *
from Environment.BitEnv import _2048
from multiprocessing import Pool


class MCTS:
    def __init__(self):
        pass
    
    def heuristic(self, grid):
        """        
        1. Decide (Up Down) or (Left Right) by sum of merged tiles
        2. Decide direction by less moved tile
        """
                
        values = [0, 0, 0, 0]        
        grids = []
        differences = [0,0,0,0]
        for move in range(0,4):
            grid_, difference = moveForHeuristic(grid, move)
            grids.append(grid_)
            if not difference:
                continue
            values[move] = len(getFreeTile(grid_))
            differences[move] = difference
        
        max_value = max(values)
        candidates = []
        for move in range(4):
            if values[move] == max_value:
                candidates.append(move)
            else:
                differences[move] = 1e+3
        
        for c in candidates:
            if differences[c] == min(differences):
                heuristic_move = c
        return grids[heuristic_move]

    def rollout(self, first_move):
        grid, _ = moveGrid(self.root_grid, first_move)
        step = 0
        done = False
        while not done:
            free_tiles = getFreeTile(grid)
            if not free_tiles and not getLegalMoves(grid):
                done = True
            move = random.randint(0,3)
            grid, changed = moveGrid(grid, move)
            if changed:
                step += 1
            # grid = self.heuristic(grid)
            # step += 1
            if step >= 100:
                break
        return step
        

    def getAction(self, root_grid, n_sim):
        self.root_grid = root_grid
        legal_moves = getLegalMoves(root_grid)
        values = {k: 0 for k in legal_moves}
        visits = {k: 0 for k in legal_moves}
        first_move_sequence = np.random.choice(legal_moves, size = n_sim)
        with Pool(6) as p:
            res = p.map(self.rollout, first_move_sequence)
        
        for value, move in zip(res, first_move_sequence):
            values[move] += value
            visits[move] += 1

        best_move = max(legal_moves, key=lambda x: values[x] / visits[x])
        return best_move
        

def main(n_episode, n_sim):
    mcts = MCTS()
    score_list = []
    for e in range(n_episode):
        start_time = time.time()
        done = False
        score = 0
        grid = env.reset()
        while not done:
            env.render()
            action = mcts.getAction(grid, n_sim)
            grid, reward, done, info = env.step(action)
            score += reward
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        spending_time = time.time() - start_time
        max_tile = np.max(grid2Board(grid))
        print(
            f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {max_tile}, Average: {average_score:.1f}"
        )
        print(f"SPENDING TIME : {spending_time:.1f} Sec\n")
    env.close()


if __name__ == "__main__":    
    env = _2048()
    n_episode = 1
    n_sim = 100
    main(n_episode=n_episode, n_sim=n_sim)
