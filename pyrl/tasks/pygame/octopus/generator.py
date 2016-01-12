# this is a game generator for Mr. Octopus.
import os
import numpy as np
import pyrl.prob as prob
import numpy.random as npr
from pyrl.tasks.pygame.octopus.task import OctopusTask, SCREEN_WIDTH, SCREEN_HEIGHT
from pyrl.utils import mkdir_if_not_exist

LEVEL_PATH = os.path.join(OctopusTask.GAME_MODULE_PATH, 'data', 'Levels')

class Generator(object):
    def sample(self):
        raise NotImplementedError('generator should have \"sample\" method')


class OctopusGenerator(Generator):
    '''
    An OctopusGenerator starts with a game layout and changes the octopus position
    uniformly at random.
    '''
    def __init__(self, name, coord=None):
        mkdir_if_not_exist(os.path.join(LEVEL_PATH, 'octopus'))
        self.data = self._parse(name)
        self.name = name
        if coord: # use coordinate from file.
            self.coord_data = self._parse(coord)
        else:
            self.coord_data = None


    def _parse(self, name):
        data = []
        with open(os.path.join(LEVEL_PATH, name + '.txt'), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                first_comma = line.find(',')
                obj = line[:first_comma]
                if obj in set(['dirt', 'grass', 'octopus', 'tp']): # this is a sprite.
                    [x, y] = line[first_comma+1:].split(',')
                    (x, y) = (int(x), int(y))
                    data.append((obj, x, y))
                else:
                    pass
        return data


    def sample(self):
        if self.coord_data:
            (_, octopus_x, octopus_y) = prob.choice(self.coord_data, 1)[0]
        else:
            octopus_x = npr.randint(0, SCREEN_WIDTH)
            octopus_y = npr.randint(0, SCREEN_HEIGHT)

        task_id = os.path.join('octopus', str(octopus_x) + '_' + str(octopus_y))
        absolute_task_path = os.path.join(LEVEL_PATH, task_id + '.txt')
        with open(absolute_task_path, 'w') as f:
            for (obj, x, y) in self.data:
                if obj == 'octopus':
                    f.write(','.join([obj, str(octopus_x), str(octopus_y)]) + '\n')
                else:
                    f.write(','.join([obj, str(x), str(y)]) + '\n')
        return OctopusTask(level=task_id)


