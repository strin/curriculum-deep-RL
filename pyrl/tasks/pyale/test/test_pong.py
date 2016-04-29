from pyrl.tasks.pyale import PythonGame
from pyrl.tasks.pyale.pong import PongGame
from pyrl.utils import Timer
from pyrl.visualize.visualize import *
from pyrl.prob import choice

game = PongGame()

with Timer('valid actions'):
    for it in range(100):
        print 'valid_actions', game.valid_actions

vr = RawVideoRecorder('video.m4v', (640, 480))
for it in range(100):
    action = choice(range(game.num_actions), 1)[0]
    reward = game.step(action)
    print 'state', game.curr_state
    print 'is_end', game.is_end()
    #vr.write_frame(game.visualize_raw())
    print 'action', action, 'reward', reward
vr.stop()
