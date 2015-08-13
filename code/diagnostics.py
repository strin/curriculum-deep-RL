import numpy as np
import matplotlib
import pylab as plt
import time
import util


def visualize_agent(agent, task, world):
    '''
        Useful to visualize 2D Trajectories inside IPython
    '''
    if util.in_ipython():
        from IPython import display
    else:
        raise NotImplementedError()

    # sample a trajectory
    task.reset()
    current_state = task.get_current_state()
    steps = 0
    states = [current_state]
    while True:
        steps += 1
        action = agent.get_action(current_state)
        next_state, reward = task.perform_action(action)
        if next_state is not None:
            states.append(next_state)
        if task.is_terminal():
            agent.end_episode(reward)
            break

    # plot the position of the agent across time
    def show_position(position):
        x, y = int(position[0]), int(position[1])
        cells = np.copy(world)
        cells[x, y] = 1
        plt.pcolor(cells)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(0.2)

    for state in states:
        show_position(state)
