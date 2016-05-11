''' warning: london simulator should be singleton, as it changes pygame behavior '''
from pyrl.common import *
from pyrl.tasks.pyale import PygameSimulator, function_intercept
from pyrl.evaluate import DrunkLearner
from pygame.locals import *
import pygame

_name = 'london'

WINNING_SCORE = 1

path_prefix = os.path.join(os.path.dirname(__file__), 'games', _name)
pygame.image._old_load = pygame.image.load
pygame.image.load = lambda path: pygame.image._old_load(os.path.join(path_prefix, path))
pygame.font._old_Font = pygame.font.Font
pygame.font.Font = lambda path, size: pygame.font._old_Font(os.path.join(path_prefix, path), size)

class LondonSimulator(PygameSimulator):
    def __init__(self, state_type='pixel'):
        PygameSimulator.__init__(self, 'london', [chr(ord('0') + k) for k in range(30)],
                state_type=state_type)


    def is_end(self):
        try:
            game_handler = self._get_attr('stateHandler').gameHandler
            if game_handler is None:
                return False
            print game_handler.gameOver
        except:
            return False


    def get_score(self):
        return self._get_attr('stateHandler').data.score


if __name__ == '__main__':
    from pyrl.visualize.visualize import VideoRecorder, RawVideoRecorder
    def callback():
        import pygame
        data = pygame.image.tostring(pygame.display.get_surface(), 'RGB')
        vr.write_frame(data)
    make_vr = lambda name: RawVideoRecorder(name, (600, 800))
    defender = LondonSimulator(state_type='pixel')
    vr = make_vr('test.m4v')
    defender.run(DrunkLearner(), callback=callback)
    vr.stop()
