# an Arcade Learning Environment (ALE) wrapper.
from pyrl.common import *
from pyrl.tasks.task import Task
from pyrl.utils import rgb2yuv
from pyrl.prob import choice
from pyrl.config import floatX
from scipy.misc import imresize
import sys

from ale_python_interface import ALEInterface

class AtariGame(Task):
    ''' RL task based on Arcade Game.
    '''

    def __init__(self, rom_path, num_frames=4, live=False, skip_frame=0):
        self.ale = ALEInterface()
        if live:
            USE_SDL = True
            if USE_SDL:
                if sys.platform == 'darwin':
                    import pygame
                    pygame.init()
                    self.ale.setBool('sound', False) # Sound doesn't work on OSX
                elif sys.platform.startswith('linux'):
                    self.ale.setBool('sound', True)
            self.ale.setBool('display_screen', True)
        self.live = live
        self.ale.loadROM(rom_path)
        self.num_frames = num_frames
        self.frames = []
        self.frame_id = 0
        self.cum_reward = 0
        self.skip_frame = skip_frame
        self.img_shape = (84, 84) # image shape according to DQN Nature paper.
        while len(self.frames) < 4:
            self.step(choice(self.valid_actions, 1)[0])
        self.reset()


    def copy(self):
        import dill as pickle
        return pickle.loads(pickle.dumps(self))


    def reset(self):
        self.ale.reset_game()
        self.frame_id = 0
        self.cum_reward = 0
        if self.skip_frame:
            for frame_i in range(self.skip_frame):
                self.step(choice(self.valid_actions, 1)[0])


    @property
    def _curr_frame(self):
        img = self.ale.getScreenRGB()
        img = imresize(img, self.img_shape, interp='bilinear')
        return rgb2yuv(img)[:, :, 0] # get Y channel, according to Nature paper.


    @property
    def curr_state(self):
        '''
        return raw pixels.
        '''
        return np.array(self.frames, dtype=floatX)


    @property
    def state_shape(self):
        (screenH, screenW) = self.ale.getScreenDims()
        return (screenH, screenW, 3)


    @property
    def num_actions(self):
        return len(self.valid_actions)


    @property
    def valid_actions(self):
        return self.ale.getLegalActionSet()


    def step(self, action):
        reward = self.ale.act(action)
        if len(self.frames) == self.num_frames:
            self.frames = self.frames[1:]
        self.frames.append(self._curr_frame)
        self.frame_id += 1
        #print 'frame_id', self.frame_id
        self.cum_reward += reward
        return reward


    def is_end(self):
        if np.abs(self.cum_reward) > 0:
            return True
        return self.ale.game_over()


    def visualize(self, fig=1, fname=None, format='png'):
        fig = plt.figure(fig, figsize=(5,5))
        plt.clf()
        plt.imshow(self.getScreenRGB())
        if fname:
            plt.savefig(fname, format=format)
        else:
            plt.show()
        return res

