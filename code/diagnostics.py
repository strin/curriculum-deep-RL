import pylab as plt
import util


def visualize_sample_trajectory(agent, task):
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
    images = [task.visualize()]
    while True:
        action = agent.get_action(current_state)
        next_state, reward = task.perform_action(action)
        images.append(task.visualize())
        if task.is_terminal():
            agent.end_episode(reward)
            break

    # show the position of the agent across time
    def show_image(image):
        plt.pcolor(image, cmap=task.cmap, norm=task.color_norm)
        display.display(plt.gcf())
        display.clear_output(wait=True)

    for image in images:
        show_image(image)
