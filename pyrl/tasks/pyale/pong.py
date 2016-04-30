from pyrl.tasks.pyale import PygameSimulator
from pygame.locals import *

WINNING_SCORE = 1

class PongSimulator(PygameSimulator):
    def __init__(self, learner):
        PygameSimulator.__init__(self, learner, 'pong', [K_DOWN, K_UP])


    def is_end(self):
        bar1_score = self._get_attr('bar1_score')
        bar2_score = self._get_attr('bar2_score')
        if bar1_score >= WINNING_SCORE or bar2_score >= WINNING_SCORE:
            return True
        return False


    def get_score(self):
        bar1_score = self._get_attr('bar1_score')
        bar2_score = self._get_attr('bar2_score')
        return bar1_score - bar2_score

