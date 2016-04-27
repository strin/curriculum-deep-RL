from pyrl.common import *
from pyrl.tasks.task import Task
from pyrl.utils import rgb2yuv
from pyrl.prob import choice
from pyrl.config import floatX
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
import sys
import pexpect
import dill
import base64
import select


import pygame
import pygame.image
from pygame.event import Event

class AsyncEvent(object):
    '''
    async event, wrapper around pygame.event
    '''
    def get(self):
        return pygame.event.get()

    def mount(self, message_type, func):
        pass # ignore.


class SyncEvent(object):
    '''
    synchronous event, used for MDP.
    '''
    def __init__(self):
        print 'sync __init__'
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open('octopus.out', 'w')
        sys.stderr = open('octopus.err', 'w')
        self.mounted = {}


    def get(self):
        for event in pygame.event.get(): # first yield all pygame events, such as window active.
            yield event

        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        #while True: # loop until a GO event is emitted.
            raw_message = sys.stdin.readline()
            message = dill.loads(base64.b64decode(raw_message))
            print 'new message', message
            if message['type'] == 'event':
                yield Event(message['event_type'], {
                    'key': message['key']
                })
            elif message['type'] in self.mounted:
                func = self.mounted[message['type']]
                ret = func()
                self.stdout.write('output>'+ encode_obj(ret) + '\r\n')


    def mount(self, message_type, func):
        self.mounted[message_type] = func


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


def query_process(process, message):
    process.sendline(encode_obj(message))
    process.expect('output>')
    raw_data = process.readline()
    return decode_obj(raw_data)


class PythonGame(Task):
    def __init__(self, game_path):
        self.game_process = None
        self.num_reset = 0
        self.game_path = game_path
        self.curr_score = 0.


    def reset(self):
        self.terminate()
        self.num_reset += 1
        self.game_process = pexpect.spawn('python %s' % self.game_path, maxread=999999)


    def is_end(self):
        return not self.game_process.isalive()


    def terminate(self):
        if self.game_process and self.game_process.isalive():
            self.game_process.terminate()


    @property
    def _curr_rgb_screen(self):
        img = query_process(self.game_process, {
            'type': 'state'
        })
        img = imresize(img, self.img_shape, interp='bilinear')
        return img

    @property
    def _curr_frame(self):
        img = self._curr_rgb_screen
        return rgb2yuv(img)[:, :, 0] # get Y channel, according to Nature paper.


    @property
    def curr_state(self):
        '''
        return raw pixels.
        '''
        return np.array(self.frames, dtype=floatX) / floatX(255.) # normalize


    @property
    def state_shape(self):
        return self.curr_state.shape


    @property
    def num_actions(self):
        return len(self.valid_actions)


    @property
    def valid_actions(self):
        return query_process(self.game_process, {
            'type': 'valid_actions':
        }


    def step(self, action):
        assert(action >= 0 and action < self.num_actions)
        score = query_process(self.game_process, {
            'type': 'action',
            'action': action
        })

        reward = score - self.curr_score
        return reward


    def visualize(self, fig=1, fname=None, format='png'):
        import matplotlib.pyplot as plt
        fig = plt.figure(fig, figsize=(5,5))
        plt.clf()
        plt.axis('off')
        res = plt.imshow(self._curr_rgb_screen)
        if fname:
            plt.savefig(fname, format=format)
        else:
            plt.show()
        return res

