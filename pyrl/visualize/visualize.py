import numpy as np
import numpy.random as npr
import time
import os
from os import path
import subprocess
from pyrl.tasks.task import Task
from pyrl.algorithms.valueiter import compute_tabular_value
from pyrl.utils import mkdir_if_not_exist
from matplotlib import pyplot

def record_game(policy, task, output_path, max_step=9999, **policy_args):
    '''
    record the play of policy on task, and save each frame to *output_path*.

    Requirements:
        task should have visualize(self, fname=None) implemented.
    '''
    step = 0
    mkdir_if_not_exist(output_path)
    task.reset()

    while step < max_step:
        task.visualize(path.join(output_path, '%d' % step))
        curr_state = task.curr_state
        action = policy.get_action(curr_state, valid_action=task.valid_actions, **policy_args)
        task.step(action)
        step += 1

    task.reset()


def record_game_multi(policy, task, output_path, num_times=5, max_step=9999, **policy_args):
    for ni in range(num_times):
        record_game(policy, task, path.join(output_path, '%d' % ni), max_step, **policy_args)


def replay_game(output_path):
    '''
    replay the frames of game play saved in *output_path*.
    '''
    step = 0
    get_path = lambda step: path.join(output_path, '%d' % step)
    while path.exists(get_path(step)):
        image = pyplot.imread(get_path(step))
        pyplot.imshow(image)
        time.sleep(0.1)


class VideoRecorder(object):
    '''
    record a video from pygame session.
    requires ffmpeg.
    '''
    FFMPEG_BIN = 'ffmpeg'

    def __init__(self, fname):
        mkdir_if_not_exist(os.path.dirname(fname))
        command = [ VideoRecorder.FFMPEG_BIN,
            '-y', # (optional) overwrite output file if it exists
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-r', '30', # frames per second
            '-i', '-', # The input comes from a pipe
            '-vcodec', 'libx264',
            '-an', # Tells FFMPEG not to expect any audio
            fname ]

        self.output = open('_video_recorder.out', 'w')
        movie = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=self.output)

        self.movie = movie
        self.finished = False

    def write_frame(self, data):
        if self.finished:
            return
        self.movie.stdin.write(data)

    def stop(self):
        self.movie.communicate()
        self.finished = True

def html_embed_mp4(video_path):
    VIDEO_TAG = """<video controls>
     <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
     Your browser does not support the video tag.
    </video>"""
    video = open(video_path, "rb").read()
    _encoded_video = video.encode("base64")
    return VIDEO_TAG.format(_encoded_video)

def html_dbx_mp4(video_path):
    """
    send the video to dropbox and returned a link to the file.
    """
    from pyrl.storage.dropbox import put_file, shared_link
    with open(video_path, 'rb') as f:
        print 'uploading video to cloud'
        put_file('pyrl/' + video_path, f)
    link = shared_link('pyrl/' + video_path)
    return html_mp4(link)

def html_mp4(video_path):
    VIDEO_TAG = """<video controls>
     <source src="{0}" type="video/mp4">
     Your browser does not support the video tag.
    </video>"""
    return VIDEO_TAG.format(video_path)



