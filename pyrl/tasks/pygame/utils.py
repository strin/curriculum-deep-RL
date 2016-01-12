# shared utils for pygames.
import pygame
from pygame.event import Event
from multiprocessing import Queue
import dill
import base64
import sys
import select


class AsyncEvent(object):
    '''
    async event, wrapper around pygame.event
    '''
    def get(self):
        return pygame.event.get()


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



def play_task_human(task, self):
    while not task.is_end():
        for event in pygame.event.get():
            action = task.ACTIONS.index(event)
            task.step(action)


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




