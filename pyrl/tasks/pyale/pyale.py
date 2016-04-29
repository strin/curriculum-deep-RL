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

import pygame
import pygame.image
from pygame.event import Event
from pygame.locals import *

from scipy.misc import imread
from StringIO import StringIO


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
