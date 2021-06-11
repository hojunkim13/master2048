import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from DQN.Agent import Agent
from Utils import *
from Environment.BitEnv import _2048()


env_name = "2048"
load = False
n_state = (20,4,4)
n_action = 4
learing_rate = 1e-4
gamma = 0.9
replay_memory_buffer_size = 100000
epsilon_decay = 0.95
epsilon_decay_step = 2500
epsilon_min = 0.05
batch_size = 256
tau = 1e-3

n_episode = 10000






if __name__ == "__main__":
    env = _2048()
    agent = Agent(n_state, n_action, learing_rate, gamma, replay_memory_buffer_size,
                epsilon_decay,epsilon_min,epsilon_decay_step, batch_size, tau)
    
    scores = []    
    for episode in range(n_episode):
        grid = env.reset()
        state = preprocessing(grid)
        done = False
        score = 0
        while not done:
            action, _ = agent.getAction(state)            
            grid, reward, done, max_tile = env.step(action)
            state_ = preprocessing(grid)
            score += reward
            agent.storeTransition(state, action, reward, state_, done)
            agent.learn()
            agent.softUpdate()
            state = state_ 

        scores.append(score)
        movingAverageScore = np.mean(scores[-100:])
        
        if (episode + 1) % 100 == 0:
            agent.save(env_name)
        print(f"Episode : {episode+1}, Score : {score:.0f}, Average: {movingAverageScore:.1f} Epsilon : {agent.epsilon:.3f}, Memory : {agent.memory.tree.n_entries}, Max : {max_tile}")
    env.close()
        
            
