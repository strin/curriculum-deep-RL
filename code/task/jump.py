import random
import environment
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from visual.frames import FrameSource

class JumpGame(environment.Environment):
    ''' Basic Actions '''
    ACTION_JUMP = 1
    ACTION_NONE = 0

    actions = [ACTION_JUMP, ACTION_NONE]

    F = 3 # stage features.

    FEAT_BRICK = np.array([1, 0, 0])
    FEAT_TRAP = np.array([0, 1, 0])
    FEAT_COIN = np.array([0, 0, 1])
    FEAT_SPACE = np.array([0, 0, 0])

    H = 11
    W = 11

    def __init__(self):
        self.reset()

    def support_tabular(self):
        ''' does not support tabular RL '''
        return False

    def get_num_states(self):
        return

    def get_state_dimension(self):
        return 1

    def get_num_actions(self):
        return len(self.actions)

    def get_allowed_actions(self, state):
        return self.actions

    def _add_brick(self, h_start, h_end):
        ''' add a brick to the stage '''
        self._add_object(h_start, h_end, self.FEAT_BRICK)

    def _add_trap(self, h_start, h_end):
        ''' add a trap to stage '''
        self._add_object(h_start, h_end, self.FEAT_TRAP)

    def _add_coin(self, h):
        ''' add a coin to stage '''
        self._add_object(h, h+1, self.FEAT_COIN)


    def _add_space(self):
        ''' add an empty column '''
        self._add_object(0, self.H, self.FEAT_SPACE)

    def _add_object(self, h_start, h_end, obj):
        col = np.zeros((self.H, 1, self.F))
        col[h_start:h_end, 0, :] = obj
        self.stage = np.concatenate((self.stage, col), axis=1)

    def _add_brick_coin(self, h):
        ''' brick coin pattern:

            | $
            | x
            | B
            -------
        '''
        col = np.zeros((self.H, 1, self.F))
        col[h:h+1, 0, :] = self.FEAT_BRICK
        col[h-2:h-1, 0, :] = self.FEAT_COIN
        self.stage = np.concatenate((self.stage, col), axis=1)

    def _add_brick_trap(self, h):
        ''' brick trap pattern:

            | T
            | x
            | B
            -------
        '''
        col = np.zeros((self.H, 1, self.F))
        col[h:h+1, 0, :] = self.FEAT_BRICK
        col[h-2:h-1, 0, :] = self.FEAT_TRAP
        self.stage = np.concatenate((self.stage, col), axis=1)

    def _gen_scene(self):
        events = [
            (1., lambda: self._add_brick_coin(self.H-1)),
            (1., lambda: self._add_brick_trap(self.H-1)),
            (5., lambda: self._add_space())
        ]
        [probs, funcs] = zip(*events)
        probs = [prob / sum(probs) for prob in probs]
        func = npr.choice(funcs, 1, replace=True, p=probs)[0]
        func()

    def _wrap_state(self):
        sprite_state = np.zeros((self,H, self.W))
        sprite_state[self.sprite[0], self.sprite[1]] = 1
        return np.concatenate((self.screen.ravel(), sprite_state.ravel()))

    def get_current_state(self):
        return self._wrap_state()

    def perform_action(self, action):
        (H, W, F) = (self.H, self.W, self.F)
        # generate scene.
        while self.stage.shape[1] <= self.screen.shape[1] * 2:
            # add new objects.
            self._gen_scene()

        self.time += 1
        self.stage = self.stage[:, 1:, :]
        self.screen = self.stage[:, :self.W, :]

        # peform agent action.
        [y, x] = self.sprite
        if action == self.ACTION_JUMP and not self.state['in-air']:
            # jump!
            self.state['in-air'] = True
            if y > 0:
                y -= 1
        else:
            if y < H-1 and (self.screen[y+1, x, :] == self.FEAT_SPACE).all():
                # gravity.
                y += 1
            if y == H-1 or not (self.screen[y+1, x, :] == self.FEAT_SPACE).all():
                self.state['in-air'] = False
        self.sprite = [y, x]

    def render(self):
        ''' render screen and sprite into an np.ndarray '''
        # render stage.
        ret = np.zeros((self.H, self.W, 3))
        to_bool = np.vectorize(bool)
        ret[:, :, :] = 255
        ret[to_bool(self.screen[:, :, 0]), :] = np.array([0, 102, 255])
        ret[to_bool(self.screen[:, :, 1]), :] = np.array([255, 51, 0])
        ret[to_bool(self.screen[:, :, 2]), :] = np.array([255, 204, 0])

        # render sprite.
        [y, x] = self.sprite
        ret[y, x, :] = np.array([0, 153, 51])

        to_uint8 = np.vectorize(np.uint8)
        return to_uint8(ret)

    def get_start_state(self):
        return self._wrap_state()

    def reset(self):
        (H, W, F) = (self.H, self.W, self.F)
        self.screen = np.zeros((H, W, F))
        self.stage = np.zeros((H, W, F))
        self.sprite = (H-1, W/2)
        self.time = 0.
        self.score = 0.
        self.state = {
            'in-air': False
        }


class JumpGameFrameSource(FrameSource):
    def __init__(self, jump_game):
        self.jump_game = jump_game
        self.frames = []

    def generate(self, meta):
        if 'keydown' in meta and meta['keydown'] == 32: # press space bar, jump!
            self.jump_game.perform_action(JumpGame.ACTION_JUMP)
        else:
            self.jump_game.perform_action(JumpGame.ACTION_NONE)
        self.frames.append(self.jump_game.render())

    def render(self, frame_id):
        return self.frames[frame_id]

    def terminated(self):
        return False

    def __len__(self):
        return len(self.frames)
