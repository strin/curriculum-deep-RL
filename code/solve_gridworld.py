from gridworld import Grid, GridWorldMDP, GridWorld
from agent import ValueIterationSolver, TDLearner, DQN
import numpy as np

# simple world with a single wall
world = np.zeros((3, 4))
world[1, 1] = 1.

# 2 reward states each with +/- 1 reward
rewards = {(0, 3): 1., (1, 3): -1.}

grid = Grid(world, action_stoch=0.2)
mdp = GridWorldMDP(grid, rewards, wall_penalty=0., gamma=0.9)

print "Solving with Value Iteration"
mdp_agent = ValueIterationSolver(mdp, tol=1e-6)
mdp_agent.learn()

values = np.zeros(world.shape)
for state in xrange(grid.get_num_states()):
    values[grid.state_pos[state]] = mdp_agent.V[state]

for pos, r in rewards.items():
    values[pos] = r

print values


print "Solving with Q-learning"
NUM_EPISODES = 25000
grid_task = GridWorld(grid, rewards, wall_penalty=0., gamma=0.9)
td_agent = TDLearner(grid_task, update='q_learning', alpha=0.05, epsilon=0.1)
for i in xrange(NUM_EPISODES):
    curr_state = grid_task.get_current_state()
    while(not grid_task.is_terminal()):
        action = td_agent.get_action(curr_state)
        next_state, reward = grid_task.perform_action(action)
        td_agent.learn(next_state, reward)
        curr_state = next_state

    grid_task.reset()
    td_agent.reset_episode()


values = np.zeros(world.shape)
for state in xrange(grid.get_num_states()):
    qvals = td_agent.Q[state]
    if len(qvals) == 0:
        val = 0
    else:
        val = max(qvals)
    values[grid.state_pos[state]] = val

for pos, r in rewards.items():
    values[pos] = r

print values

print "Solving with SARSA"
NUM_EPISODES = 50000
grid_task = GridWorld(grid, rewards, wall_penalty=0., gamma=0.9)
td_agent = TDLearner(grid_task, update='sarsa', alpha=0.05, epsilon=0.05)
for i in xrange(NUM_EPISODES):
print "Solving with DQN"
NUM_EPISODES = 100000
grid = Grid(world, action_stoch=0.2)
grid_task = GridWorld(grid, rewards, wall_penalty=0., gamma=0.9, tabular=False)
dqn = DQN(grid_task, hidden_dim=128, l2_reg=0.0, lr=0.05, epsilon=0.1)
for episode in xrange(NUM_EPISODES):
    while grid_task.is_terminal():
        grid_task.reset()

    curr_state = grid_task.get_current_state()
    while True:
        action = dqn.get_action(curr_state)
        next_state, reward = grid_task.perform_action(action)
        if grid_task.is_terminal():
            dqn.end_episode(reward)
            break
        else:
            dqn.learn(next_state, reward)
            curr_state = next_state

    if episode % 100 == 0:
        values = np.zeros(world.shape)
        for row in xrange(world.shape[0]):
            for col in xrange(world.shape[1]):
                if world[row, col] == 0:  # agent can occupy this state
                    agent_state = np.zeros_like(world)
                    agent_state[row, col] = 1.
                    # state = np.concatenate((agent_state.ravel(), world.ravel())).reshape(-1, 1)
                    state = agent_state.ravel().reshape(-1, 1)

                    qvals = dqn.fprop(state)
                    values[row, col] = np.max(qvals)

        for pos, r in rewards.items():
            values[pos] = r

        print "\n"
        print values
