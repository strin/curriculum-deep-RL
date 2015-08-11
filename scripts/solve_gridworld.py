
# coding: utf-8

# In[ ]:

import scriptinit
import numpy as np
import matplotlib
import pylab as plt
from IPython import display
from gridworld import Grid, GridWorldMDP, GridWorld
from agent import ValueIterationSolver, TDLearner, DQN, RecurrentReinforceAgent


# In[ ]:

# set up a very simple world with a single wall
world = np.zeros((3, 4))
world[1, 1] = 1.

# 2 reward states each with +/- 1 reward
rewards = {(0, 3): 1., (1, 3): -1.}

grid = Grid(world, action_stoch=0.2)


# In[ ]:

# Solve the grid using value iteration as a sanity check
mdp = GridWorldMDP(grid, rewards, wall_penalty=0., gamma=0.9)
mdp_agent = ValueIterationSolver(mdp, tol=1e-6)
mdp_agent.learn()

# visualize the results
values = np.zeros(world.shape)
for state in xrange(grid.get_num_states()):
    values[grid.state_pos[state]] = mdp_agent.V[state]

expected_reward = np.sum(values) / float(world.shape[0] * world.shape[1] - np.sum(world) - len(rewards))

# put the terminal state rewards in for visualization purposes
for pos, r in rewards.items():
    values[pos] = r

print values
print 'Expected reward', expected_reward


# In[ ]:

# template for interaction between the agent and the task
def grid_experiment(agent, task, num_episodes, diagnostic_callback=None, diagnostic_frequency=100):
    for episode in xrange(NUM_EPISODES):
        total_reward = 0.
        while grid_task.is_terminal():
            grid_task.reset()

        curr_state = grid_task.get_current_state()
        num_steps = 0.
        while True:
            num_steps += 1
            if num_steps > 200:
                print 'Lying and tell the agent the episode is over!'
                agent.end_episode(0)
                num_steps = 0.

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


# In[ ]:

# Solve gridworld using a DQN

# setup task and agent
NUM_EPISODES = 5001
grid_task = GridWorld(grid, rewards, wall_penalty=0., gamma=0.9, tabular=False)
dqn = DQN(grid_task, hidden_dim=128, l2_reg=0.0, lr=0.05, epsilon=0.15)

# diagnostic function
def compute_value_function(episode, total_reward):
    print 'Episode number: ',  episode
    values = np.zeros(world.shape)
    for row in xrange(world.shape[0]):
        for col in xrange(world.shape[1]):
            if world[row, col] == 0:  # agent can occupy this state
                agent_state = np.zeros_like(world)
                agent_state[row, col] = 1.
                state = agent_state.ravel().reshape(-1, 1)

                qvals = dqn.fprop(state.T)
                values[row, col] = np.max(qvals)

    for pos, r in rewards.items():
        values[pos] = r

    print values, '\n'

# run the experiment
grid_experiment(dqn, grid_task, NUM_EPISODES, diagnostic_callback=compute_value_function, diagnostic_frequency=1000)


# In[ ]:

# Solve gridworld using the recurrent reinforce agent (policy gradient)

# setup new task and agent
NUM_EPISODES = 10001
grid = Grid(world, action_stoch=0.2)
grid_task = GridWorld(grid, rewards, wall_penalty=0., gamma=0.9, tabular=False)
rr_agent = RecurrentReinforceAgent(grid_task, hidden_dim=128, num_samples=10)


# define diagnostic function (maintains internal state)
class plot_historical_avg():
    def __init__(self):
#         self.running_avg_reward = 0.
#         self.avg_reward_hist = []
        self.reward_hist = []
        self.moving_avg_reward = []
        self.episodes = []
    
    def __call__(self, episode, total_reward):
#         self.running_avg_reward += (1. / (episode + 1)) * (total_reward - self.running_avg_reward)
#         self.avg_reward_hist.append(self.running_avg_reward)
        self.reward_hist.append(total_reward)
        if episode % 50 == 0:
            self.moving_avg_reward.append(np.mean(self.reward_hist))
            # clear history
            self.reward_hist = []
            self.episodes.append(episode)
           
            # plotting
            plt.plot(self.episodes, expected_reward * np.ones(len(self.episodes)), 'r')
            plt.plot(self.episodes, self.moving_avg_reward, 'b')
            plt.xlabel('Iterations')
            plt.ylabel('Average Reward')
            plt.title('Average Reward over The Last 30 time steps versus Iteration')
            display.display(plt.gcf())
#             display.clear_output(wait=True)

    
historical_averager = plot_historical_avg()

# run the experiment!
grid_experiment(rr_agent, grid_task, NUM_EPISODES, diagnostic_callback=historical_averager, diagnostic_frequency=1)

