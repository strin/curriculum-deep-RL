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
    def __init__(self, game_module_name, valid_events, state_type='pixel', frames_per_action=4):
        self.game_module_name = game_module_name
        self.game_module = None # cached game module
        self.game_code = None
        self.valid_actions = range(len(valid_events))
        self.valid_events = valid_events
        self.curr_screen_rgb = None
        self.learner = None
        self.state_type = state_type
        self.frames_per_action = frames_per_action
        self.num_frames = 4


    def _get_attr(self, name):
        if not self.game_module: # dynamically load library if not found.
            game_frame = [frame for frame in inspect.stack()
                          if frame[1].find('pyrl/tasks/pyale/games') != -1][0][0]
            self.game_module = inspect.getmodule(game_frame)
        return getattr(self.game_module, name)


    def _get_frame(self):
        if self.state_type == 'pixel':
            from scipy.misc import imresize
            img = self.curr_screen_rgb
            img = rgb2yuv(img)[:, :, 0] # get Y channel, according to Nature paper.
            img = imresize(img, (84, 84), interp='bicubic')
            return img / floatX(255.0)


    def _get_state(self):
        return np.array(self.frames)


    def _on_screen_update(self, _, *args, **kwargs):
        self.num_steps += 1
        is_end = self.is_end()

        if not is_end and (self.num_steps-1) % self.frames_per_action > 0:
            if self.callback: # TODO: callback on skip steps. now callback is only used for videos.
                self.callback()

            return

        score = self.get_score()
        reward = score - self.curr_score
        self.cum_reward += reward
        self.curr_score = score
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

            if self.last_action:
                self.learner.send_feedback(reward, curr_state, self.valid_actions, is_end)
            if is_end:
                return

            action = self.learner.get_action(curr_state, self.valid_actions)
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
                        if self.pass_quit_event:
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


    def run(self, learner, callback=None):
        self.learner = learner
        self.callback = callback
        self._keys_pressed = []
        self._last_keys_pressed = []
        self.last_action = None
        self.cum_reward = 0
        self.curr_score = 0
        self.num_steps = 0
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


        if self.game_module:
            reload(self.game_module)
        else:
            pygame.display.flip = function_intercept(pygame.display.flip, self._on_screen_update)
            pygame.display.update = function_intercept(pygame.display.update, self._on_screen_update)
            pygame.event.get = function_intercept(pygame.event.get, self._on_event_get)
            pygame.time.Clock = function_intercept(pygame.time.Clock, self._on_time_clock)
            sys.exit = function_intercept(sys.exit, self._on_exit) # TODO: this doesn't work.
            with Timer('running game ' + self.game_module_name):
                exec('''import pyrl.tasks.pyale.games.%s as the_game''' % self.game_module_name)
            self.game_module = the_game

        return self.cum_reward







def encode_obj(obj):
    '''
    encode object in one-line string.
    so we can send it over stdin.
    '''
    return base64.b64encode(dill.dumps(obj))


def decode_obj(obj):
    '''
    decode the one-line object sent over stdin.
    '''
    return dill.loads(base64.b64decode(obj))


class AsyncEvent(object):
    '''
    async event, wrapper around pygame.event
    '''
    def get(self):
        return pygame.event.get()

    def mount(self, message_type, func):
        pass # ignore.


def respond_ale(stdout, ret):
    stdout.write('>' + encode_obj(ret) + '\r\n')
    stdout.flush()


def respond_ok(stdout):
    respond_ale(stdout, 'ok')


class SyncEvent(object):
    '''
    synchronous event, used for MDP.
    '''
    def __init__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open('game.out', 'w')
        sys.stderr = open('game.err', 'w')
        self.mounted = {}
        self.frames_togo = 0


    def get(self):
        for event in pygame.event.get(): # first yield all pygame events, such as window active.
            yield event

        if self.frames_togo > 0: # if in state 'go', skip interaction with ALE.
            self.frames_togo -= 1
            if self.frames_togo == 0:
                respond_ok(self.stdout)

        elif 'ALE' in os.environ:  # Arcade leanring experiment.
            while self.frames_togo == 0: # wait for 'go' signal.
                while sys.stdin in select.select([sys.stdin], [], [], 99999)[0]:
                    raw_message = sys.stdin.readline()
                    message = decode_obj(raw_message)
                    if message['type'] == 'event':
                        yield Event(message['event'][0], {
                            'key': message['event'][1]
                        })
                        respond_ok(self.stdout)
                    elif message['type'] == 'go':
                        self.frames_togo = message['frame']
                        break
                    elif message['type'] in self.mounted:
                        func = self.mounted[message['type']]
                        ret = func()
                        respond_ale(self.stdout, ret)
                        if message['type'] == 'is_end' and ret == True:
                            exit(0)   # terminate the game.


    def mount(self, message_type, func):
        self.mounted[message_type] = func


def query_process(process, message):
    process.sendline(encode_obj(message))
    process.expect('>')
    raw_data = process.readline()
    data = decode_obj(raw_data)
    return data


def tell_process(process, message):
    resp = query_process(process, message)
    assert(resp == 'ok')


def query_valid_events(process):
    return query_process(process, {
            'type': 'valid_events'
        })


def query_process_score(process):
    return query_process(process, {
            'type': 'score'
        })


def query_process_is_end(process):
    return query_process(process, {
            'type': 'is_end'
        })


def query_process_screen(process):
    img_data = query_process(process, {
            'type': 'screen'
        })
    return img_data['data']


def query_process_state(process):
    return query_process(process, {
            'type': 'state'
        })


def tell_process_event(process, event):
    return tell_process(process, {
            'type': 'event',
            'event': event
        })


def tell_process_go(process, num_frames):
    return tell_process(process, {
            'type': 'go',
            'frame': num_frames
        })



class PythonGame(Task):
    def __init__(self, game_path, frames_per_action=1):
        '''
        go_frame: how many frames each action last.
        '''
        self.game_path = os.path.join(os.path.dirname(__file__),
                     'games',
                     game_path
                     )
        self.pyrl_root = __file__[:__file__.find('pyrl/tasks/pyale')-1]
        self.game_process = None
        self.frames_per_action = frames_per_action
        self.num_reset = 0
        self.img_shape = (84, 84)
        self.reset()
        self.valid_events = query_valid_events(self.game_process)


    def reset(self):
        self.terminate()
        self.has_ended = False
        self.curr_score = 0.
        self.num_reset += 1
        env = {"ALE": "true",
                "PYRL": self.pyrl_root + '/',
                }
        env.update({key: os.environ[key] for key in os.environ.keys()})
        self.game_process = pexpect.spawn('python %s' % self.game_path,
                                          maxread=9999999,
                                          env=env)
        #self.game_process = subprocess.Popen(['python', self.game_path],
        #                                     bufsize=-1,
        #                                     stdin=subprocess.PIPE,
        #                                     stdout=subprocess.PIPE)


    def is_end(self):
        if self.has_ended:
            return True
        is_end = query_process_is_end(self.game_process)
        self.has_ended = is_end
        return is_end


    def terminate(self):
        if self.game_process and self.game_process.isalive():
            self.game_process.terminate()


    @property
    def _curr_rgb_screen(self):
        raise NotImplementedError('rgb input not defined')


    @property
    def _curr_frame(self):
        img = self._curr_rgb_screen
        return rgb2yuv(img)[:, :, 0] # get Y channel, according to Nature paper.


    @property
    def curr_state(self):
        '''
        return raw pixels.
        '''
        return query_process_state(self.game_process)


    @property
    def state_shape(self):
        return self.curr_state.shape


    @property
    def num_actions(self):
        return len(self.valid_actions)


    @property
    def valid_actions(self):
        return range(len(self.valid_events))


    def step(self, action):
        if self.has_ended:
            assert(False)
        assert(action >= 0 and action < self.num_actions)
        event = self.valid_events[action]
        tell_process_event(self.game_process, event)
        tell_process_go(self.game_process, self.frames_per_action)
        score = query_process_score(self.game_process)
        reward = score - self.curr_score
        self.curr_score = score
        return reward


    #def visualize(self, fig=1, fname=None, format='png'):
    #    import matplotlib.pyplot as plt
    #    fig = plt.figure(fig, figsize=(5,5))
    #    plt.clf()
    #    plt.axis('off')
    #    res = plt.imshow(self._curr_rgb_screen)
    #    if fname:
    #        plt.savefig(fname, format=format)
    #    else:
    #        plt.show()
    #    return res


    def visualize_raw(self):
        data = query_process_screen(self.game_process)
        return data
