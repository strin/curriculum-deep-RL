''' warning: defender simulator should be singleton, as it changes pygame behavior '''
from pyrl.common import *
from pyrl.tasks.pyale import PygameSimulator, function_intercept
from pyrl.evaluate import DrunkLearner
from pygame.locals import *
import pygame

_name = 'defender'


WINNING_SCORE = 1

old_image_load = pygame.image.load
pygame.image.load = lambda path: old_image_load(os.path.join(os.path.dirname(__file__), 'games', _name, path))


class DefenderSimulator(PygameSimulator):
    def __init__(self, state_type='pixel'):
        PygameSimulator.__init__(self, 'defender', [K_DOWN, K_UP, K_LEFT, K_RIGHT, K_SPACE],
                state_type=state_type)


    def is_end(self):
        game = self._get_attr('game')
        return not game.levels


    def get_score(self):
        shot = self._get_attr('enemi').shot
        game = self._get_attr('game')
        exploded = int(game.exploded)
        level_cleared = int(not game.levels)
        score = shot * 1 + exploded * -10. + level_cleared * 10.
        print 'score', score
        return score



if __name__ == '__main__':
    defender = DefenderSimulator(state_type='pixel')
    defender.run(DrunkLearner())
