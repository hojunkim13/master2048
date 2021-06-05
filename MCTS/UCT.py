import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Environment.BitEnv import _2048
from Environment.BitEnv import *
import numpy as np
import time
import random




class Node:
    def __init__(self, parent, move, legal_moves=[0, 1, 2, 3]):
        self.parent = parent
        self.W = 0
        self.N = 0
        self.child = {}
        self.move = move
        self.legal_moves = legal_moves
        self.untried_moves = self.legal_moves.copy()

    def getUCT(self, c_uct= 0.5):
        Q = self.W / self.N
        exp_component = c_uct * np.sqrt(np.log(self.parent.N) / self.N)
        return Q + exp_component

    def isLeaf(self):
        return self.untried_moves != []

    def isRoot(self):
        return self.parent is None


class MCTS:
    last_move = None
    def getDepth(self, node):
        depth = 0
        while not node.isRoot():
            depth += 1
            node = node.parent
        return depth

    def select(self, node, grid):
        while not node.isLeaf():
            node = max(node.child.values(), key=Node.getUCT)
            grid, _ = moveGrid(grid, node.move)
        return node, grid

    def expand(self, node):
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        child_node = Node(node, move)
        node.child[move] = child_node
        return child_node

    def evaluate(self, grid):
        step = 0
        while not isEnd(grid):
            move = random.randint(0, 3)
            grid, changed = moveGrid(grid, move)
            if changed:
                step += 1
            if step >= 100:
                break
        return step / 50 - 1

    def backpropagation(self, node, value):
        node.W += value
        node.N += 1
        if not node.isRoot():
            self.backpropagation(node.parent, value)

    def searchTree(self):
        node = self.root_node
        grid = self.root_grid
        leaf_node, grid = self.select(node, grid)

        if not isEnd(grid):
            child_node = self.expand(leaf_node)
            value = self.evaluate(grid)
            self.backpropagation(child_node, value)
        else:
            self.backpropagation(leaf_node, -1)

    def getAction(self, root_grid, n_sim):
        self.root_node = Node(None, None, getLegalMoves(root_grid))
        self.root_grid = root_grid

        # if self.last_move is None:
        #     self.root_node = Node(None, None, getLegalMoves(root_grid))
        #     self.root_grid = root_grid
        # else:
        #     new_root_node = self.root_node.child[self.last_move]
        #     new_root_node.parent = None
        #     new_root_node.move = None

        #     new_root_node.legal_moves = getLegalMoves(root_grid)
        #     unlegal_moves = list(set([0, 1, 2, 3]) - set(new_root_node.legal_moves))
        #     for move in unlegal_moves:
        #         del new_root_node.child[move]
        #     self.root_node = new_root_node
        #     self.root_grid = root_grid

        for _ in range(n_sim):
            self.searchTree()

        robust_move = max(self.root_node.child.values(), key=lambda x: x.N).move
        self.last_move = robust_move
        return robust_move


def main(n_episode, n_sim):
    env = _2048()
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
        print(
            f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}"
        )
        print(f"SPENDING TIME : {spending_time:.1f} Sec\n")
    env.close()


if __name__ == "__main__":
    main(n_episode=1, n_sim=150)
