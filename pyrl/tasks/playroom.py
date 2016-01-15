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
            'hand_pos': list(hand_pos),
            'eye_pos': list(eye_pos),
            'mark_pos': list(mark_pos),
            'red_button_pos': list(red_button_pos),
            'blue_button_pos': list(blue_button_pos),
            'monkey_pos': list(monkey_pos),
            'bell_pos': list(bell_pos),
            'ball_pos': list(ball_pos),
            'switch_pos': list(switch_pos),
            'light': False,
            'music': False,
        }

        self.object_pos = [tuple(red_button_pos), tuple(blue_button_pos),
                           tuple(monkey_pos), tuple(bell_pos),
                           tuple(ball_pos), tuple(switch_pos)
                           ]
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
                )


    def _state_to_id(self, state):
        ''' convert state to integer represtnation '''
        pos_to_number = lambda pos: pos[0] * self.size + pos[1]
        res = 0
        # encode pos.
        for key in ['hand_pos', 'eye_pos', 'mark_pos', 'ball_pos']:
            res = res * self.size ** 2 + pos_to_number(state[key])

        # encode binary.
        for key in ['light', 'music']:
            res = res * 2 + int(state[key])

        return res


    def _id_to_state(self, res):
        number_to_pos = lambda number: (number % self.size, int(number / self.size))
        state= self.state

        # decode binary.
        for key in ['light', 'music']:
            state[key] = res % 2
            res = int(res / 2)

        # decode pos.
        for key in ['hand_pos', 'eye_pos', 'mark_pos', 'ball_pos']:
            state[key] = number_to_pos(res % (self.size ** 2))
            res = int(res / (self.size ** 2))

        return state


    @property
    def curr_state(self):
        ''' encode states into a number '''
        return self._state_to_id(self.state)


    @property
    def num_actions(self):
        return len(self.ACTIONS)


    def _can_touch_object(self):
        return (tuple(self.state['eye_pos']) in self.object_pos
                    and tuple(self.state['hand_pos']) in self.object_pos)


    def _action_id(self, action):
        return self.ACTIONS.index(action)


    def _action_ids(self, actions):
        return [self._action_id(action) for action in actions]


    @property
    def valid_actions(self):
        if self._can_touch_object():
            return self._action_ids(self.ACTIONS)
        else:
            return self._action_ids([action for action in self.ACTIONS if action != 'touch object'])


    def salient_event(self, state, next_state):
        '''
        tell if the transition from state to next_state is a salient event.
        '''
        state_repr = self._id_to_state(state)
        next_state_repr = self._id_to_state(next_state)

        whitelist = ['light', 'music']

        for key in whitelist:
            if state_repr[key] != next_state_repr[key]:
                return [key]

        if state_repr['ball_pos'] != next_state_repr['ball_pos']: # kick the ball.
            events = ['ball']

            if ((state_repr['ball_pos'][0] == next_state_repr['ball_pos'][0]
                  and state_repr['ball_pos'][0] == state_repr['bell_pos'][0]
                  )
                 or
                 (state_repr['ball_pos'][1] == next_state_repr['ball_pos'][1]
                  and state_repr['ball_pos'][1] == state_repr['bell_pos'][1]
                  )
                 ): # ring the bell.

                events.append('bell')

                if state_repr['music'] == True and state_repr['light'] == False:
                    events.append('scream')

            return events

        return ''


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
                self.state['light'] = not self.state['light']
            elif (self.state['eye_pos'] == self.state['ball_pos']
                  and (self.state['mark_pos'][0] == self.state['ball_pos'][0]
                       or self.state['mark_pos'][1] == self.state['ball_pos'][1])
                  ): # kick the ball if ball and mark are on a straight line.
                self.state['ball_pos'] = self.state['mark_pos']

        return 0.


    def __repr__(self):
        maze = np.chararray((self.size, self.size), unicode=True)
        maze[:] = u"_"
        maze[self.state['hand_pos']] = u"\u270B"
        maze[self.state['eye_pos']] = u"\U0001F440"
        maze[self.state['mark_pos']] = u"\u274C"
        maze[self.state['red_button_pos']] = u"\u25B6"
        maze[self.state['blue_button_pos']] = u"\u25A0"
        maze[self.state['monkey_pos']] = u"\U0001F435"
        maze[self.state['bell_pos']] = u"\u237E"
        maze[self.state['ball_pos']] = u"\u26BD"
        maze[self.state['switch_pos']] = u"\u233D"

        res = []
        for i in range(self.size):
            for j in range(self.size):
                res.append(maze[i, j])
            res.append(u'\n')

        print res

        return u"".join(res).encode('utf-8')



