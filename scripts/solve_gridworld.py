
# coding: utf-8

# In[1]:

import scriptinit
from gridworld import Grid, GridWorldMDP, GridWorld
from agent import ValueIterationSolver, TDLearner, DQN, RecurrentReinforceAgent
import numpy as np
import matplotlib
import pylab as plt
from IPython import display


# In[2]:

# set up a very simple world with a single wall
world = np.zeros((3, 4))
world[1, 1] = 1.

# 2 reward states each with +/- 1 reward
rewards = {(0, 3): 1., (1, 3): -1.}

grid = Grid(world, action_stoch=0.2)


# In[7]:

# Solve the grid using value iteration as a sanity check
mdp = GridWorldMDP(grid, rewards, wall_penalty=0., gamma=0.9)
mdp_agent = ValueIterationSolver(mdp, tol=1e-6)
mdp_agent.learn()

# visualize the results
values = np.zeros(world.shape)
for state in xrange(grid.get_num_states()):
    values[grid.state_pos[state]] = mdp_agent.V[state]

for pos, r in rewards.items():
    values[pos] = r

print values


# In[37]:

# template for interaction between the agent and the task
def grid_experiment(agent, task, num_episodes, diagnostic_callback=None, diagnostic_frequency=100):
    for episode in xrange(NUM_EPISODES):
        total_reward = 0.
        while grid_task.is_terminal():
            grid_task.reset()

        curr_state = grid_task.get_current_state()
        while True:
            action = agent.get_action(curr_state)
            next_state, reward = grid_task.perform_action(action)
            total_reward += reward
            if grid_task.is_terminal():
                agent.end_episode(reward)
                break
            else:
                agent.learn(next_state, reward)
                curr_state = next_state

        if episode % diagnostic_frequency == 0:
            if diagnostic_callback is not None:
                diagnostic_callback(episode, total_reward)


# In[32]:

# Solve gridworld using a DQN

# setup task and agent
NUM_EPISODES = 5001
grid_task = GridWorld(grid, rewards, wall_penalty=0., gamma=0.9, tabular=False)
dqn = DQN(grid_task, hidden_dim=128, l2_reg=0.0, lr=0.05, epsilon=0.1)

# diagnostic function
def compute_value_function(episode, total_reward):
    print 'Episode number, ' episode
    values = np.zeros(world.shape)
    for row in xrange(world.shape[0]):
        for col in xrange(world.shape[1]):
            if world[row, col] == 0:  # agent can occupy this state
                agent_state = np.zeros_like(world)
                agent_state[row, col] = 1.
                state = agent_state.ravel().reshape(-1, 1)

                qvals = dqn.fprop(state)
                values[row, col] = np.max(qvals)

    for pos, r in rewards.items():
        values[pos] = r

    print values, '\n'

# run the experiment
grid_experiment(dqn, grid_task, NUM_EPISODES, diagnostic_callback=compute_value_function, diagnostic_frequency=1000)


# In[73]:

# Solve gridworld using the recurrent reinforce agent (policy gradient)

# setup new task and agent
NUM_EPISODES = 4001
grid_task = GridWorld(grid, rewards, wall_penalty=0., gamma=0.9, tabular=False)
rr_agent = RecurrentReinforceAgent(grid_task, hidden_dim=1024)


# define diagnostic function (maintains internal state)
class plot_historical_avg():
    def __init__(self):
        self.running_avg_reward = 0.
        self.avg_reward_hist = []
    
    def __call__(self, episode, total_reward):
        self.running_avg_reward += (1. / (episode + 1)) * (total_reward - self.running_avg_reward)
        self.avg_reward_hist.append(self.running_avg_reward)
            self.reward_history.append(total_reward)
            if episode % 20 == 0:
                plt.plot(np.arange(episode + 1), self.avg_reward_hist)
                plt.xlabel('Iterations')
                plt.ylabel('Average Reward')
                display.display(plt.gcf())
                display.clear_output(wait=True)

    
historical_averager = plot_historical_avg()

# run the experiment!
grid_experiment(rr_agent, grid_task, NUM_EPISODES, diagnostic_callback=historical_averager, diagnostic_frequency=1)


# In[ ]:



