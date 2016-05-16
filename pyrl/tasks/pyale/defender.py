''' warning: defender simulator should be singleton, as it changes pygame behavior '''
from pyrl.common import *
from pyrl.tasks.pyale import PygameSimulator, function_intercept
from pyrl.evaluate import DrunkLearner
from pygame.locals import *
from pyrl.config import floatX
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
        return score


class DefenderRAMSimulator(DefenderSimulator):
    def __init__(self):
        DefenderSimulator.__init__(self, state_type='ram')


    def _get_ram_state(self):
        ram = []
        W = 400.
        H = 600.
        ship = self._get_attr('ship')
        enemi = self._get_attr('enemi')
        shotami = self._get_attr('shotami')
        shotenemi = self._get_attr('shotenemi')
        ram.extend([ship.topleft[0] / W, ship.topleft[1] / H,
                    (ship.topleft[0] + ship.rect.right - ship.rect.left) / W,
                    (ship.topleft[1] + ship.rect.top - ship.rect.bottom) / W])
        C = 10
        ram_enemi = np.zeros(2 * C) # at most handle 10 enemy ships.
        for (i, e) in enumerate(enemi[:C]):
            ram_enemi[i * 2] = e.top / H
            ram_enemi[i * 2 + 1] = e.left / W
        ram.extend(list(ram_enemi))

        ram_shotami = np.zeros(2 * C) # at most handle 10 bullets.
        for (i, e) in enumerate(shotami[:C]):
            ram_shotami[i * 2] = e.top / H
            ram_shotami[i * 2 + 1] = e.left / W
        ram.extend(list(ram_shotami))

        ram_shotenemi = np.zeros(2 * C) # at most handle enemy 10 bullets.
        for (i, e) in enumerate(shotenemi[:C]):
            ram_shotenemi[i * 2] = e.top / H
            ram_shotenemi[i * 2 + 1] = e.left / W
        ram.extend(list(ram_shotenemi))

        return np.array(ram, dtype=floatX)


if __name__ == '__main__':
    # defender = DefenderSimulator(state_type='pixel')
    defender = DefenderRAMSimulator()
    import os
    os.environ.update({
        'SHIELD_SHIP': '1',
        'SHIELD_ENEMY': '30'
    })
    defender.run(DrunkLearner())
    os.environ.update({
        'SHIELD_SHIP': '1000',
        'SHIELD_ENEMY': '30'
    })
    defender.run(DrunkLearner())
