from pyrl.tasks.atari import AtariGame
from pyrl.prob import choice

game = AtariGame('data/roms/pong.bin')
while not game.is_end():
    a = choice(game.valid_actions, 1)[0]
    game.step(a)


