import os
import cPickle as pickle
import pylab as plt
import util
from experiment import Controller


def visualize_sample_trajectory(agent, task, path=None):
    '''
        Useful to visualize 2D Trajectories inside IPython
    '''
    if util.in_ipython():
        from IPython import display
    else:
        print 'Must be inside IPython!'
        raise NotImplementedError()

    if path is None:
        images = get_sample_trajectory(agent, task)
    else:
        images = pickle.load(file(path, 'r'))

    # show the position of the agent across time
    def show_image(image):
        plt.pcolor(image, cmap=task.cmap, norm=task.color_norm)
        display.display(plt.gcf())
        display.clear_output(wait=True)

    for image in images:
        show_image(image)


def get_sample_trajectory(agent, task):
    '''
        Returns a list of images (matrices) based on a single sampled
        trajectory of the agent.
    '''
    # sample a trajectory
    task.reset()
    agent.reset()
    current_state = task.get_start_state()
    images = [task.visualize()]
    while True:
        action = agent.get_action(current_state)
        next_state, reward = task.perform_action(action)
        images.append(task.visualize())
        if task.is_terminal():
            break
        current_state = next_state

    agent.reset()
    return images


class VisualizeTrajectoryController(Controller):
    def __init__(self, visualize_wait=100, dir_name='trajectories'):
        self.dir_name = dir_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.visualize_wait = visualize_wait

    def control(self, experiment):
        if experiment.num_episodes % self.visualize_wait == 0:
            trajectory = get_sample_trajectory(experiment.agent, experiment.task)
            print 'Saving trajectory...'
            file_name = 'trajectory_' + str(experiment.num_episodes) + '.cpkl'
            pickle.dump(trajectory, file(os.path.join(self.dir_name, file_name), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
