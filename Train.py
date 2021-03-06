import numpy as np
from Agent import Agent
from Environment.BitEnv import _2048, grid2Board
from Utils import preprocessing



lr = 1e-3
batch_size = 256
n_sim = 50
maxlen = 10000

n_episode = 10000
state_dim = (16, 4, 4)
action_dim = 4


env = _2048()
agent = Agent(state_dim, action_dim, lr, batch_size, n_sim, maxlen)

#agent.load()



def main():
    score_list = []
    for e in range(n_episode):
        done = False
        grid = env.reset()
        score = 0
        loss = 0
        agent.step = 0
        while not done:
            # env.render()
            action = agent.getAction(grid)
            new_grid, reward, done, info = env.step(action)            
            agent.memory.stackMemory(preprocessing(grid), reward)
            grid = new_grid
            score += reward
            agent.step += 1
        agent.memory.updateZ()
        if (e + 1) % 10 == 0:
            loss = agent.learn()
            agent.save()
        score_list.append(agent.step)
        average_score = np.mean(score_list[-100:])
        max_tile = np.max(grid2Board(grid))
        print(
            f"Episode : {e+1} / {n_episode}, Score : {agent.step}, Max Tile : {max_tile}, Average: {average_score:.1f}, Loss : {loss:.3f}"
        )


if __name__ == "__main__":
    main()
