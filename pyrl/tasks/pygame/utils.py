# shared utils for pygames.
from pyrl.utils import mkdir_if_not_exist

import os
import pygame
import pygame.image
from pygame.event import Event

import threading
import dill
import base64
import sys
import select
import subprocess
import time
import pexpect


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



class VideoRecorder(object):
    '''
    record a video from pygame session.
    requires ffmpeg.
    '''
    FFMPEG_BIN = 'ffmpeg'

    def __init__(self, surface, fname):
        screen_size = (surface.get_width(), surface.get_height())

        mkdir_if_not_exist(os.path.dirname(fname))
        command = [ VideoRecorder.FFMPEG_BIN,
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', '%sx%s' % (screen_size[0], screen_size[1]), # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', '24', # frames per second
            '-i', '-', # The input comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', 'mpeg4',
            fname ]

        movie = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=None)

        self.surface = surface
        self.movie = movie
        self.running = True

        thread = threading.Thread(target=self.record)
        thread.start()


    def record(self):
        while self.running:
            data = pygame.image.tostring(self.surface, 'RGB')
            self.movie.stdin.write(data)
            time.sleep(1. / 24)


    def stop(self):
        self.running = False

