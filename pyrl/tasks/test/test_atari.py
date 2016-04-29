from pyrl.tasks.atari import AtariGame
from pyrl.prob import choice
from pyrl.visualize.visualize import VideoRecorder
from StringIO import StringIO
import matplotlib.pyplot as plt

def callback(task):
    imgbuf = StringIO()
    task.visualize(fig = 1, fname='__cache__.jpg', format='jpg')
    with open('__cache__.jpg', 'rb') as imgbuf:
        data = imgbuf.read()
    vr.write_frame(data)

game = AtariGame('data/roms/pong.bin', live=True, skip_frame=65)
#plt.imshow(game._curr_frame, cmap='Greys_r', interpolation='none')
#plt.show()

vr = VideoRecorder('video.m4v')

buf = StringIO()
count = 0
while not game.is_end():
    count += 1
    a = choice(game.valid_actions, 1)[0]
    print 'action', a
    print 'game', game.valid_actions
    a = 12
    #game.visualize(fig=1, fname=buf, format='jpg')
    print game.curr_state.shape
    game.step(a)
    callback(game)

vr.stop()
