# an Arcade Learning Environment (ALE) wrapper.
from pyrl.tasks.task import Task

from ale_python_interface import ALEInterface

class AtariGame(Task):
    ''' RL task based on Arcade Game.
    '''

    def __init__(self, rom_path):
        self.ale = ALEInterface()
        USE_SDL = False
        if USE_SDL:
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                self.ale.setBool('sound', False) # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                self.ale.setBool('sound', True)
        self.ale.setBool('display_screen', True)
        self.ale.loadROM(rom_path)


    def copy(self):
        import dill as pickle
        return pickle.loads(pickle.dumps(self))


    def reset(self):
        self.ale.reset_game()


    @property
    def curr_state(self):
        '''
        return raw pixels.
        '''
        return self.ale.getScreenRGB()


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
        return self.ale.act(action)


    def is_end(self):
        return self.ale.game_over()



