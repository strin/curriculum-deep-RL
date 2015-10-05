import numpy as np
import numpy.random as npr
import dill
import os

from gridworld import Grid, GridWorldMDP
from agent import ValueIterationSolver

CACHE = dict()

def solve_grid_by_value_iteration(grid, rewards, wall_penalty=0., gamma=0.9):
    # Solve the grid using value iteration as a sanity check
    mdp = GridWorldMDP(grid, rewards, wall_penalty=wall_penalty, gamma=gamma)
    mdp_agent = ValueIterationSolver(mdp, tol=1e-6)
    mdp_agent.learn()

    # visualize the results
    values = np.zeros(grid.shape)
    for state in xrange(grid.get_num_states()):
        values[grid.state_pos[state]] = mdp_agent.V[state]

    # put the terminal state rewards in for visualization purposes
    for pos, r in rewards.items():
        values[pos] = r

    return values

def solve_world_by_value_iteration(world, rewards, wall_penalty=0., gamma=0.9, action_stoch=0.2):
    # create a grid first.
    grid = Grid(world, action_stoch=action_stoch)

    # solve the grid.
    return solve_grid_by_value_iteration(grid, rewards, wall_penalty, gamma)

def solve_task_by_value_iteration(task):
    return solve_grid_by_value_iteration(task.env, task.rewards, task.wall_penalty, task.gamma)

def maze_template(maze):
    global world
    html = ["<table style='height: 10px;'>"]
    (H, W) = maze.shape
    for h in range(H):
        html.append("<tr style='height: 10px;'>")
        for w in range(W):
            on = maze[h, w]
            color = 'blue' if on else 'white'
            html.extend([
                    "<td style='height: 10px; width: 10px; line-height: 10px; padding: 0 0 0 0;' height='10px'>",
                    """
                    <input style='height: 10px; width: 10px; background-color: %(color)s; border: none; padding: 0 0 0 0;' type='button' id='btn_%(h)d_%(h)d'
                    onmousedown="
                    this.style['background-color'] = 'blue';
                    IPython.notebook.kernel.execute('world[%(h)d][%(w)d] = 1');
                    ">
                    </input>
                    """ %
                    dict(color=color, h=h, w=w),
                    '</td>'
                ])
        html.append('</tr>')
    return ''.join(html)

def save(key, value):
    CACHE[key] = value
    if not os.path.exists('cache/'):
        os.mkdir('cache')
    dill.dump(value, open('cache/' + key, 'w'))

def load(key):
    if key in CACHE:
        return CACHE[key]
    try:
        return dill.load(open('cache/' + key, 'r'))
    except IOError as e:
        return None

def train_test_split(dataset, training_ratio = 0.6):
    indices = npr.choice(range(len(dataset)), int(len(dataset) * 0.6), replace=True)
    train_set = [dataset[ind] for ind in indices]
    test_set = [dataset[ind] for ind in range(len(dataset)) if ind not in indices]
    return (train_set, test_set)

