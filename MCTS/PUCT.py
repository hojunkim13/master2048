import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Environment.BitEnv import *
from Utils import preprocessing
import numpy as np
import random
import torch


class Node:
    def __init__(self, parent, move, legal_moves=[0, 1, 2, 3]):
        self.parent = parent
        self.W = 0
        self.N = 0
        self.child = {}
        self.move = move
        self.legal_moves = legal_moves
        self.untried_moves = self.legal_moves.copy()

    def getPUCT(self, prob, c_puct=0.5):
        Q = self.W / (self.N + 1e-8)
        U = c_puct * prob * np.sqrt(self.parent.N) / (1 + self.N)
        return Q + U

    def isLeaf(self):
        return self.N == 0

    def isRoot(self):
        return self.parent is None


class MCTS:
    def __init__(self, net):
        self.net = net

    def getDepth(self, node):
        depth = 0
        while not node.isRoot():
            depth += 1
            node = node.parent
        return depth

    def select(self, node, grid):            
        while not node.isLeaf():
            with torch.no_grad():
                state = preprocessing(grid).unsqueeze(0).cuda().float()
                policy, _ = self.net(state)
                policy = policy.cpu().numpy()[0]
            values = [-1, -1, -1, -1]
            for move in node.legal_moves:
                values[move] = node.child[move].getPUCT(policy[move])
            node = node.child[np.argmax(values)]
            grid, _ = moveGrid(grid, node.move)
        return node, grid


    def expand(self, node):
        for move in range(4):            
            child_node = Node(node, move)
            node.child[move] = child_node        

    def evaluate(self, grid):
        with torch.no_grad():
            state = preprocessing(grid).unsqueeze(0).cuda().float()
            _, value = self.net(state)        
            value = value.cpu().item()
        # rollout_value = self.rollout(grid)
        # value = (value + rollout_value) / 2
        return value

    def rollout(self, grid):
        step = 0
        while isEnd(grid):
            grid, _ = (grid)
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
            self.expand(leaf_node)        
            value = self.evaluate(grid)    
            self.backpropagation(leaf_node, value)
        else:
            self.expand(leaf_node)        
            self.backpropagation(leaf_node, -1)

    def getAction(self, root_grid, n_sim):
        self.root_node = Node(None, None, getLegalMoves(root_grid))
        self.root_grid = root_grid

        # if self.last_move is None:
        #     self.root_node = Node(None, None, get_legal_moves(root_grid))
        #     self.root_grid = root_grid
        # else:
        #     new_root_node = self.root_node.child[self.last_move]
        #     new_root_node.parent = None
        #     new_root_node.move = None

        #     new_root_node.legal_moves = get_legal_moves(root_grid)
        #     unlegal_moves = list(set([0, 1, 2, 3]) - set(new_root_node.legal_moves))
        #     for move in unlegal_moves:
        #         del new_root_node.child[move]
        #     self.root_node = new_root_node
        #     self.root_grid = root_grid

        for _ in range(n_sim):
            self.searchTree()
        
        visits = []
        for i in range(4):
            try:
                visit = self.root_node.child[i].N
            except KeyError:
                visits = 0
            visits.append(visit)
            
        return visits
