from pyrl.tasks.atari import AtariGame
from pyrl.prob import choice

game = AtariGame('data/roms/pong.bin', live=True, skip_frame=65)
while not game.is_end():
    a = choice(game.valid_actions, 1)[0]
    print game.curr_state.shape
    game.step(a)


