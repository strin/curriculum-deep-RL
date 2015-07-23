from gridworld import Grid, GridWorldMDP
from agent import ValueIterationSolver
import numpy as np

# simple world with a single wall
world = np.zeros((3, 4))
world[1, 1] = 1.

# 2 reward states each with +/- 1 reward
rewards = {(0, 3): 1., (1, 3): -1.}

grid = Grid(world, action_stoch=0.2)
mdp = GridWorldMDP(grid, rewards, wall_penalty=0., gamma=0.9)

mdp_agent = ValueIterationSolver(mdp)

mdp_agent.learn()

values = np.zeros(world.shape)
for state in range(grid.get_num_states()):
    values[grid.state_pos[state]] = mdp_agent.V[state]

print values
