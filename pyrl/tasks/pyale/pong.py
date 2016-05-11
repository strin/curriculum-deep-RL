from pyrl.tasks.pyale import PygameSimulator
from pyrl.config import floatX
from pygame.locals import *
import numpy as np

WINNING_SCORE = 1

class PongSimulator(PygameSimulator):
    def __init__(self):
        PygameSimulator.__init__(self, 'pong', [K_DOWN, K_UP],
                state_type='pixel')


    def is_end(self):
        bar1_score = self._get_attr('bar1_score')
        bar2_score = self._get_attr('bar2_score')
        if bar1_score >= WINNING_SCORE or bar2_score >= WINNING_SCORE:
            return True
        return False


    def get_score(self):
        bar1_score = self._get_attr('bar1_score')
        #bar2_score = self._get_attr('bar2_score')
        return bar1_score



class PongRAMSimulator(PongSimulator):
    def __init__(self):
        PygameSimulator.__init__(self, 'pong', [K_DOWN, K_UP],
                state_type='ram')


    def _get_ram_state(self):
        bar1_y = self._get_attr('bar1_y') / 480.
        H1 = self._get_attr('H1') / 480.
        bar2_y = self._get_attr('bar2_y') / 480.
        H2 = self._get_attr('H2') / 480.
        circle_x = self._get_attr('circle_x') / 640.
        circle_y = self._get_attr('circle_y') / 480.
        return np.array([bar1_y, bar2_y, H1, H2, circle_x, circle_y], dtype=floatX)


