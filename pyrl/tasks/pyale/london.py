''' warning: london simulator should be singleton, as it changes pygame behavior '''
from pyrl.common import *
from pyrl.tasks.pyale import PygameSimulator, function_intercept
from pyrl.evaluate import DrunkLearner
from pygame.locals import *
from pyrl.config import floatX
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
            return game_handler.gameOver
        except:
            return False


    def get_score(self):
        return self._get_attr('stateHandler').data.score / 100. - float(self.is_end())


class LondonRAMSimulator(LondonSimulator):
    def __init__(self):
        LondonSimulator.__init__(self, state_type='ram')


    def _get_ram_state(self):
        ram = []
        W = 600.
        H = 800.
        stateHandler = self._get_attr('stateHandler')
        bombs = stateHandler.data.bombs.sprites()
        superBombs = stateHandler.data.superBombs.sprites()
        C = 10
        ram_bombs = np.zeros(2 * C) # at most handle 10 enemy items.
        ram_superBombs = np.zeros(2 * C)

        for (i, bomb) in enumerate(bombs[:C]):
            ram_bombs[i * 2] = bomb.rect[0] / W
            ram_bombs[i * 2 + 1] = bomb.rect[1] / H

        for (i, superBomb) in enumerate(superBombs[:C]):
            ram_superBombs[i * 2] = superBomb.rect[0] / W
            ram_superBombs[i * 2 + 1] = superBomb.rect[1] / H

        ram.extend(list(ram_bombs))
        ram.extend(list(ram_superBombs))

        return np.array(ram, dtype=floatX)

    @property
    def state_shape(self):
        return self.num_frames * 40


if __name__ == '__main__':
    from pyrl.visualize.visualize import VideoRecorder, RawVideoRecorder
    def callback():
        import pygame
        data = pygame.image.tostring(pygame.display.get_surface(), 'RGB')
        vr.write_frame(data)
    make_vr = lambda name: RawVideoRecorder(name, (600, 800))
    london = LondonRAMSimulator()
    london.run(DrunkLearner(), callback=None)
    print 'screen size', london.screen_size
    os.environ.update({
        'BULLET': '2'
    })
    london.run(DrunkLearner(), callback=None)

