from pyrl.common import *
from pyrl.tasks.task import Task
from pyrl.utils import rgb2yuv, Timer
from pyrl.prob import choice
from pyrl.config import floatX
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
import sys
import pexpect.popen_spawn
import subprocess
import dill
import base64
import select
import traceback

import pygame.image
from pygame.event import Event

from scipy.misc import imread
from StringIO import StringIO


import pygame
from pygame.locals import *
import pygame.key
import pygame.surfarray
import imp
import os
import re
import inspect

def function_intercept(intercepted_func, intercepting_func, pass_on=False):
    """
    Intercepts a method call and calls the supplied intercepting_func with the result of it's call and it's arguments
    Example:
        def get_event(result_of_real_event_get, *args, **kwargs):
            # do work
            return result_of_real_event_get
        pygame.event.get = function_intercept(pygame.event.get, get_event)
    :param intercepted_func: The function we are going to intercept
    :param intercepting_func:   The function that will get called after the intercepted func. It is supplied the return
    value of the intercepted_func as the first argument and it's args and kwargs.
    :return: a function that combines the intercepting and intercepted function, should normally be set to the
             intercepted_functions location
    """

    def wrap(*args, **kwargs):
        real_results = intercepted_func(*args, **kwargs)  # call the function we are intercepting and get it's result
        intercepted_results = intercepting_func(real_results, *args, **kwargs)  # call our own function a
        if pass_on:
            return real_results + intercepted_results
        return intercepted_results

    return wrap


class PygameSimulator(object):
    def __init__(self, game_module_name, valid_events, state_type='pixel', frames_per_action=2, pass_event=True):
        self.game_module_name = game_module_name
        self.game_module = None # cached game module
        self.game_code = None
        self.valid_actions = range(len(valid_events))
        self.num_actions = len(self.valid_actions)
        self.valid_events = valid_events
        self.curr_screen_rgb = None
        self.learner = None
        self.state_type = state_type
        self.frames_per_action = frames_per_action
        self.num_frames = 4
        self.pass_event = pass_event

    def _get_attr(self, name):
        if not self.game_module: # dynamically load library if not found.
            game_frame = [frame for frame in inspect.stack()
                          if frame[1].find('pyrl/tasks/pyale/games') != -1][-1][0]
            self.game_module = inspect.getmodule(game_frame)
        return getattr(self.game_module, name)


    def _get_frame(self):
        if self.state_type == 'pixel':
            from scipy.misc import imresize
            img = self.curr_screen_rgb
            img = rgb2yuv(img)[:, :, 0] # get Y channel, according to Nature paper.
            img = imresize(img, (84, 84), interp='bicubic')
            return img / floatX(255.0)
        elif self.state_type == 'ram':
            return self._get_ram_state()
        else:
            raise NotImplementedError()


    def _get_ram_state(self):
        raise NotImplementedError()


    def _get_state(self):
        return np.array(self.frames)

    
    @property
    def screen_size(self):
        surface = pygame.display.get_surface()
        if not surface:
            return None
        width = surface.get_width()
        height = surface.get_height()
        return (width, height)


    def _on_screen_update(self, _, *args, **kwargs):
        self.total_frames += 1
        is_end = self.is_end()

        if not is_end and (self.total_frames-1) % self.frames_per_action > 0:
            if self.callback: # TODO: callback on skip steps. now callback is only used for videos.
                self.callback()

            return


        score = self.get_score()
        reward = score - self.curr_score
        self.cum_reward += reward
        self.curr_score = score

        if self.state_type == 'pixel':
            self.curr_screen_rgb = pygame.surfarray.array3d(pygame.display.get_surface())

        frame = self._get_frame()
        self.frames.append(frame)

        if len(self.frames) < self.num_frames:
            action = choice(self.valid_actions, 1)[0]
        else:
            if len(self.frames) > self.num_frames:
                self.frames = self.frames[-4:]
            curr_state = self._get_state()

            if self.callback:
                self.callback()

            if self.last_action != None:
                self.learner.send_feedback(reward, curr_state, self.valid_actions, is_end)
            if is_end:
                return

            action = self.learner.get_action(curr_state, self.valid_actions)
            self.total_steps += 1
            self.last_action = action

        self._last_keys_pressed = self._keys_pressed
        self._keys_pressed = [self.valid_events[action]]


    def _on_event_get(self, _, *args, **kwargs):
        if self.is_end():
            return [pygame.event.Event(QUIT, {})]

        key_down_events = [pygame.event.Event(KEYDOWN, {"key": x})
                           for x in self._keys_pressed if x not in self._last_keys_pressed]
        key_up_events = [pygame.event.Event(KEYUP, {"key": x})
                         for x in self._last_keys_pressed if x not in self._keys_pressed]

        result = []
        if args:
            if hasattr(args[0], "__iter__"):
                args = args[0]

            for type_filter in args:
                if type_filter == QUIT:
                    if type_filter == QUIT:
                        if self.pass_event:
                            for e in _:
                                if e.type == QUIT:
                                    result.append(e)
                    else:
                        pass  # never quit
                elif type_filter == KEYUP:
                    result = result + key_up_events
                elif type_filter == KEYDOWN:
                    result = result + key_down_events
        else:
            result = key_up_events + key_down_events
            for e in _:
                if e.type == QUIT:
                    result.append(e)

        return result


    def _on_time_clock(self, real_clock, *args, **kwargs):
        pass


    def _on_exit(self):
        print 'exit event'
        pass


    def run(self, learner, max_steps=None, callback=None):
        self.learner = learner
        self.callback = callback
        self._keys_pressed = []
        self._last_keys_pressed = []
        self.last_action = None
        self.cum_reward = 0
        self.curr_score = 0
        self.total_frames = 0
        self.total_steps = 0
        self.frames = []
        #pygame.time.get_ticks = function_intercept(pygame.time.get_ticks, self.get_game_time_ms)
        # run game using dynamic importing.
        ## crude way of writing load.
        #self.game_module = imp.new_module(self.game_module_name)
        #
        #if not self.game_code:
        #    game_path = os.path.join(os.path.dirname(__file__),
        #                            'games'
        #                            self.game_module_name + '.py')
        #    with open(game_path, 'r') as f:
        #        self.game_code = compile(f.read(), game_path, 'exec')
        #        print 'game code', self.game_code
        #exec(self.game_code, self.game_module.__dict__)


        try:
            if self.game_module:
                from pyrl.tasks.pyale.reload import reload
                reload(self.game_module, exclude=['sys', 'os.path', 'builtins', '__main__', 'pygame'])
            else:
                pygame.display.flip = function_intercept(pygame.display.flip, self._on_screen_update)
                pygame.display.update = function_intercept(pygame.display.update, self._on_screen_update)
                pygame.event.get = function_intercept(pygame.event.get, self._on_event_get)
                pygame.time.Clock = function_intercept(pygame.time.Clock, self._on_time_clock)
                sys.exit = function_intercept(sys.exit, self._on_exit) # TODO: this doesn't work.
                with Timer('running game ' + self.game_module_name):
                    exec('''import pyrl.tasks.pyale.games.%s as the_game''' % self.game_module_name)
                self.game_module = the_game
        except Exception as e:
            print '[Exception]', e.message
            traceback.print_exc()

        return self.cum_reward


