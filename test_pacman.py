from pyrl.tasks.pacman.game_mdp import *
from pyrl.tasks.pacman.ghostAgents import *
import pyrl.tasks.pacman.graphicsDisplay as graphicsDisplay
import sys

if __name__ == '__main__':
    layout = layout.getLayout('pyrl/tasks/pacman/layouts/smallGrid.lay')
    ghostType = DirectionalGhost
    agents = [ghostType( i+1 ) for i in range(2)]
    display = graphicsDisplay.PacmanGraphics(zoom=1.0, frameTime = 0.1)
    task = PacmanTask(layout, agents, display)
    print task.num_actions
    while True:
        ch = sys.stdin.readline()
        reward = task.step(int(ch[0]))
        print task.valid_actions
        print 'reward', reward
        print task.curr_state





