class TaskSimulator(object):
    def __init__(self, learner, task):
        '''
        learner:
            an abstraction of RL learning agents with
                get_action(curr_states) # get action on current state.
                send_feedback(reward)   # send reward to agent for last action.
        '''
        self.task = task
        self.learner = learner


    def run(self, callback=None):
        task = self.task
        num_steps = 0.
        cum_reward = 0.
        factor = 1.

        task.reset()
        curr_state = task.curr_state
        while True:
            # TODO: Hack!
            # print 'Lying and tell the agent the episode is over!'
            #if self.gamma < 1. and num_steps >= np.log(tol) / np.log(self.gamma):
            #    meta['curr_valid_actions'] = []
            #    self._end_episode(0, meta)
            #    break

            action = self.learner.get_action(curr_state, task.valid_actions)
            reward = task.step(action)
            cum_reward += reward
            try:
                next_state = task.curr_state
                next_valid_actions = task.valid_actions
            except: # session has ended.
                next_state = None
                next_valid_actions = None
            self.learner.send_feedback(reward, next_state, next_valid_actions, task.is_end())

            if callback:
                callback(task)

            if task.is_end():
                break

            curr_state = next_state
        task.reset()
        return cum_reward

import pygame
import pygame.key
import pygame.surfarray


class PygameSimulator(object):
    def __init__(self, game_module):
        self.game_module = game_module


    def _on_screen_update(self, _, *args, **kwargs):
        print 'screen update!'
        pass


    def _on_event_get(self, _, *args, **kwargs):
        print 'on event get'
        pass


    def _on_time_clock(self, real_clock, *args, **kwargs):
        pass

    def run(self):
        reload(self.game_module)
        pygame.display.flip = function_intercept(pygame.display.flip, self._on_screen_update)
        pygame.display.update = function_intercept(pygame.display.update, self._on_screen_update)
        pygame.event.get = function_intercept(pygame.event.get, self._on_event_get)
        pygame.time.Clock = function_intercept(pygame.time.Clock, self._on_time_clock)
        #pygame.time.get_ticks = function_intercept(pygame.time.get_ticks, self.get_game_time_ms)
