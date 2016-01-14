from pyrl.tasks.task import Task
from pyrl.tasks.pygame.utils import AsyncEvent, SyncEvent, encode_obj, decode_obj
from pyrl.tasks.pygame.octopus.Lake.main import setup_screen, Game, start_game, SCREEN_WIDTH, SCREEN_HEIGHT
import pyrl.tasks.pygame.octopus
import pygame
import os
import pexpect
import numpy as np
import numpy.random as npr

from skimage.transform import resize
from skimage.filters import gaussian_filter



class OctopusTask(Task):
    DISPLAY_NONE = 0
    DISPLAY_PYGAME = 1

    # path to game.
    GAME_MODULE_FILE = os.path.abspath(pyrl.tasks.pygame.octopus.__file__)
    GAME_MODULE_PATH = os.path.dirname(GAME_MODULE_FILE)
    GAME_PATH = os.path.join(GAME_MODULE_PATH, 'run_game.py')

    # all actions should be included in array `ACTIONS`.
    ACTIONS = [[pygame.K_UP],
               [pygame.K_LEFT],
               [pygame.K_RIGHT],
               [pygame.K_DOWN],
               [pygame.K_UP, pygame.K_LEFT],
               [pygame.K_UP, pygame.K_RIGHT],
               []] # None -> no action.


    def __init__(self, level=1):
        self.colors = {}
        self.level = level
        self.game_process = None

        self.num_reset = 0
        self.reset()


    def reset(self):
        self.terminate()
        self.num_reset += 1
        os.environ['level'] = self.level
        os.environ['video'] = 'video/' + str(self.level) + '/%d' % self.num_reset + '.mp4'
        self.game_process = pexpect.spawn('python %s' % OctopusTask.GAME_PATH, maxread=999999)


    def is_end(self):
        return not self.game_process.isalive()


    def terminate(self):
        if self.game_process and self.game_process.isalive():
            self.game_process.terminate()


    @property
    def curr_state(self):
        self.game_process.sendline(encode_obj({
            'type': 'state_dict'
        }))
        self.game_process.expect('output>')
        raw_data = self.game_process.readline()
        state_dict_embed = decode_obj(raw_data)
        # create state_matrix from state_dict.
        state_dict = {}
        state_stack = []
        for (key, value) in state_dict_embed.items():
            for rect in value:
                (x, xe, y, ye) = rect
                if key not in state_dict:
                    state_dict[key] = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))
                state_dict[key][y:ye, x:xe] = 1.
            # resize the representation to 32x32.
            MAP_SIZE = 32
            filter_sigma = np.sqrt((SCREEN_HEIGHT / MAP_SIZE) ** 2 + (SCREEN_WIDTH / MAP_SIZE) ** 2)
            filtered = gaussian_filter(state_dict[key], sigma=filter_sigma)
            resized = resize(filtered, (32, 32), preserve_range=True)
            # normalize so that each channel has same strength.
            resized = resized / (1e-4 + np.max(resized))
            state_dict[key] = resized
            # add to feature representation.
            state_stack.append(state_dict[key])

        return np.array(state_stack)

    @property
    def num_actions(self):
        return len(OctopusTask.ACTIONS)


    @property
    def valid_actions(self):
        return range(self.num_actions)


    def step(self, action):
        assert(action >= 0 and action < self.num_actions)
        try:
            for event_key in OctopusTask.ACTIONS[action]:
                self.game_process.sendline(encode_obj({
                    'type': 'event',
                    'event_type': pygame.KEYDOWN,
                    'key': event_key
                }))
            clock = pygame.time.Clock()
            clock.tick(60)
            for event_key in OctopusTask.ACTIONS[action]:
                self.game_process.sendline(encode_obj({
                    'type': 'event',
                    'event_type': pygame.KEYUP,
                    'key': event_key
                }))


            self.game_process.sendline(encode_obj({
                'type': 'go'
            }))
            self.game_process.expect('output>')

            self.game_process.sendline(encode_obj({
                'type': 'gameover'
            }))
            self.game_process.expect('output>')
            gameover = decode_obj(self.game_process.readline())
            return int(gameover)
        except:
            return 0.


    @property
    def state_shape(self):
        return self.curr_state.shape


    def visualize(self, state, fname=None):
        '''
        visualize the state as a static image.
        '''
        import matplotlib.pyplot as plt

        image = np.zeros((state.shape[1], state.shape[2], 3))

        for dim in range(state.shape[0]):
            if dim in self.colors:
                color = self.colors[dim]
            else:
                color = [npr.randint(0, 255),
                        npr.randint(0, 255),
                        npr.randint(0, 255)]
                self.colors[dim] = color
            for ci in range(3):
                image[:, :, ci] += color[ci] * state[dim, :, :] / float(state.shape[0])

        image = image / (1e-4 + np.max(image)) * 255
        image = np.vectorize(np.uint8)(image)

        plt.figure(0)
        fig = plt.imshow(image, interpolation='none')

        if fname:
            plt.savefig(fname)
        else:
            plt.show()











