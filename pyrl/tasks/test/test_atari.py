from pyrl.tasks.atari import AtariGame
from pyrl.prob import choice
from StringIO import StringIO
import matplotlib.pyplot as plt

game = AtariGame('data/roms/pong.bin', live=True, skip_frame=65)
#plt.imshow(game._curr_frame, cmap='Greys_r', interpolation='none')
#plt.show()

buf = StringIO()
while not game.is_end():
    a = choice(game.valid_actions, 1)[0]
    #game.visualize(fig=1, fname=buf, format='jpg')
    print game.curr_state.shape
    game.step(a)


