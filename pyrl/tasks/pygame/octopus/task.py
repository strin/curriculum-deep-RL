from pyrl.tasks.task import Task
from pyrl.tasks.pygame.utils import AsyncEvent, SyncEvent, encode_obj, decode_obj
from pyrl.tasks.pygame.octopus.Lake.main import setup_screen, Game, start_game, SCREEN_WIDTH, SCREEN_HEIGHT
import pygame
import pexpect
import numpy as np


class OctopusTask(Task):
    DISPLAY_NONE = 0
    DISPLAY_PYGAME = 1

    # path to game.
    GAME_PATH = 'pyrl/tasks/pygame/octopus/run_game.py'

    # all actions should be included in array `ACTIONS`.
    ACTIONS = [[pygame.K_UP],
               [pygame.K_LEFT],
               [pygame.K_RIGHT],
               [pygame.K_DOWN],
               [pygame.K_UP, pygame.K_LEFT],
               [pygame.K_UP, pygame.K_RIGHT],
               []] # None -> no action.


    def __init__(self, level=1, display=DISPLAY_NONE):
        self.level = level
        self.game_process = pexpect.spawn('python %s' % OctopusTask.GAME_PATH, maxread=999999)


    def is_end(self):
        return not self.game_process.isalive()


    @property
    def curr_state(self):
        self.game_process.sendline(encode_obj({
            'type': 'state_dict'
        }))
        self.game_process.expect('output>')
        raw_data = self.game_process.readline()
        print 'raw_data', raw_data
        state_dict_embed = decode_obj(raw_data)
        # create state_matrix from state_dict.
        state_dict = {}
        for (key, value) in state_dict_embed.items():
            for rect in value:
                (x, xe, y, ye) = rect
                if key not in state_dict:
                    state_dict[key] = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))
                state_dict[key][y:ye, x:xe] = 1.

        return state_dict

    @property
    def num_actions(self):
        return len(OctopusTask.ACTIONS)


    @property
    def valid_actions(self):
        return range(self.num_actions)


    def step(self, action):
        assert(action >= 0 and action < self.num_actions)
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





