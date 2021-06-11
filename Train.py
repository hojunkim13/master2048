import numpy as np
from Agent import Agent
from Environment.BitEnv import _2048, grid2Board
from Utils import preprocessing, MemoryBuffer
import pickle


lr = 1e-3
batch_size = 256
n_sim = 50
maxlen = 10000

n_episode = 10000
state_dim = (16, 4, 4)
action_dim = 4
agent = Agent(state_dim, action_dim, lr, batch_size, n_sim)
env = _2048()
# agent.load("2048")

# try:
#     memory = pickle.load("Data/MemoryBuffer.pkl")
# except:
memory = MemoryBuffer(maxlen)


def main():
    score_list = []
    for e in range(n_episode):
        done = False
        grid = env.reset()
        score = 0
        loss = 0
        agent.step = 0
        while not done:
            #env.render()
            action = agent.getAction(grid)
            new_grid, reward, done, info = env.step(action)
            memory.stackMemory(preprocessing(grid), reward)
            grid = new_grid
            score += reward
            agent.step += 1
        memory.updateZ()
        loss = agent.learn(memory)
        if (e + 1) % 10 == 0:
            agent.save("2048")
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        max_tile = np.max(grid2Board(grid))
        print(
            f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {max_tile}, Average: {average_score:.1f}, Loss : {loss:.3f}"
        )


if __name__ == "__main__":
    main()
