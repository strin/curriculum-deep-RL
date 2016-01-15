# Playroom game from Singh et al. 04
# http://www-anw.cs.umass.edu/pubs/2004/singh_bc_NIPS04.pdf

from pyrl.tasks.task import Task
from pyrl.utils import to_string
import pyrl.prob as prob

import random
import numpy as np
import matplotlib.pyplot as plt


class Playroom(Task):

    ACTIONS = ['move eye to hand',
               'move eye to marker',
               'move eye north',
               'move eye south',
               'move eye west',
               'move eye east',
               'move eye to a random object',
               'move hand to eye',
               'move marker to eye',
               'touch object' # only available when eyes and hands are on the same object.
               ]

    def __init__(self, size, hand_pos, eye_pos, mark_pos,
                 red_button_pos, blue_button_pos, monkey_pos,
                 bell_pos, ball_pos, switch_pos):
        self.state = {
            'hand_pos': hand_pos,
            'eye_pos': eye_pos,
            'mark_pos': mark_pos,
            'red_button_pos': red_button_pos,
            'blue_button_pos': blue_button_pos,
            'monkey_pos': monkey_pos,
            'bell_pos': bell_pos,
            'ball_pos': ball_pos,
            'switch_pos': switch_pos
            'light': False,
            'music': False,
            'scream': False
        }

        self.object_pos = [red_button_pos, blue_button_pos, monkey_pos, bell_pos, ball_pos, switch_pos]
        self.size = size


    @property
    def state_shape(self):
        return 1


    @property
    def num_states(self):
        return (self.size ** 2  # hand_pos
                * self.size ** 2 # eye_pos
                * self.size ** 2 # mark_pos
                * self.size ** 2 # ball_pos
                * 2        # is light on?
                * 2        # is music on?
                * 2        # is monkey screaming?
                )


    @property
    def curr_state(self):
        ''' encode states into a number '''
        pos_to_number = lambda pos: pos[0] * self.size + pos[1]
        res = 0
        # encode pos.
        for key in ['hand_pos', 'eye_pos', 'mark_pos', 'ball_pos']:
            res = res * self.size ** 2 + pos_to_number(self.state[key])

        # encode binary.
        for key in ['light', 'music', 'scream']:
            res = res * 2 + int(self.state[key])

        return res


    @property
    def num_actions(self):
        return len(self.ACTIONS)


    def _can_touch_object(self):
        return (self.state['eye_pos'] in self.object_pos
                    and self.state['hand_pos'] in self.object_pos)


    @property
    def valid_actions(self):
        if self._can_touch_object():
            return self.ACTIONS
        else:
            return [action for action in self.ACTIONS if action != 'touch object']:


    def step(self, actionid):
        assert(actionid >= 0 and actionid < self.num_actions)
        action = self.ACTIONS[actionid]

        if action == 'move eye to hand':
            self.state['eye_pos'] = self.state['hand_pos']
        elif action == 'move eye to marker':
            self.state['eye_pos'] = self.state['mark_pos']
        elif action == 'move eye north':
            if self.state['eye_pos'][0] > 0:
                self.state['eye_pos'][0] -= 1
        elif action == 'move eye south':
            if self.state['eye_pos'][0] < self.size - 1:
                self.state['eye_pos'][0] += 1
        elif action == 'move eye west':
            if self.state['eye_pos'][1] > 0:
                self.state['eye_pos'][1] -= 1
        elif action == 'move eye east':
            if self.state['eye_pos'][1] < self.size - 1:
                self.state['eye_pos'][1] += 1
        elif action == 'move eye to a random object':
            pos = prob.choice(self.object_pos, 1)[0]
            self.state['eye_pos'] = pos
        elif action == 'move hand to eye':
            self.state['hand_pos'] = self.state['eye_pos']
        elif action == 'move marker to eye':
            self.state['mark_pos'] = self.state['eye_pos']
        elif action == 'touch object' and self._can_touch_object():
            if self.state['eye_pos'] == self.state['red_button_pos']:
                self.state['music'] = False
            elif self.state['eye_pos'] == self.state['blue_button_pos']:
                self.state['music'] = True
            elif self.state['eye_pos'] == self.state['switch_pos']:
                self.state['light'] = !self.state['light']
            #(TODO:) kick a ball, make monkey scream.
        return 0.


    def __repr__(self):
        self.maze = np.chararray((size, size), unicdoe=True)
        self.maze[:] = u" "
        self.maze[hand_pos] = u"\u270B"
        self.maze[eye_pos] = u"\U0001F440"
        self.maze[mark_pos] = u"\u274C"
        self.maze[red_button_pos] = u"\u25B6"
        self.maze[blue_button_pos] = u"\u25A0"
        self.maze[monkey_pos] = u"\U0001F435"
        self.maze[bell_pos] = u"\u237E"
        self.maze[ball_pos] = u"\u26BD"
        self.maze[switch_pos] = u"\u233D"
        return to_string(self.state)





