import pytest

from pyrl.tasks.playroom import Playroom

def make_playroom():
    playroom=Playroom(size=5,
                        hand_pos=(0, 3), eye_pos=(1, 3), mark_pos=(2, 4),
                        red_button_pos=(4, 4), blue_button_pos=(4, 0),
                        monkey_pos=(1, 2), bell_pos=(1, 4), ball_pos=(1, 0), switch_pos=(2, 2)
                        )
    return playroom


def test_repr():
    playroom = make_playroom()


def test_step_eye():
    playroom = make_playroom()
    playroom.step(playroom.ACTIONS.index('move eye north'))
    eye_pos = playroom.state['eye_pos']
    assert(eye_pos[0] == 0 and eye_pos[1] == 3)


def test_step_eye_to_hand():
    playroom = make_playroom()
    playroom.step(playroom.ACTIONS.index('move eye to hand'))
    eye_pos = playroom.state['eye_pos']
    assert(eye_pos[0] == 0 and eye_pos[1] == 3)


def test_step_hand_to_eye():
    playroom = make_playroom()
    playroom.step(playroom.ACTIONS.index('move hand to eye'))
    hand_pos = playroom.state['hand_pos']
    assert(hand_pos[0] == 1 and hand_pos[1] == 3)


def test_touch_switch():
    playroom = make_playroom()
    action = playroom.ACTIONS.index('touch object')
    assert(action not in playroom.valid_actions)

    playroom.step(playroom.ACTIONS.index('move eye south'))
    playroom.step(playroom.ACTIONS.index('move eye west'))
    eye_pos = playroom.state['eye_pos']
    assert(tuple(eye_pos) == (2, 2))
    assert(action not in playroom.valid_actions)

    playroom.step(playroom.ACTIONS.index('move hand to eye'))
    hand_pos = playroom.state['hand_pos']
    assert(tuple(hand_pos) == (2, 2))

    assert(action in playroom.valid_actions)
    playroom.step(playroom.ACTIONS.index('touch object'))

    assert(playroom.state['light'] == True)

    return playroom


def test_monkey():
    playroom = test_touch_switch()

    playroom.step(playroom.ACTIONS.index('move eye south'))
    playroom.step(playroom.ACTIONS.index('move eye south'))
    playroom.step(playroom.ACTIONS.index('move eye west'))
    playroom.step(playroom.ACTIONS.index('move eye west'))

    assert(tuple(playroom.state['eye_pos']) == (4, 0))

    playroom.step(playroom.ACTIONS.index('move hand to eye'))

    assert(tuple(playroom.state['hand_pos']) == (4, 0))

    playroom.step(playroom.ACTIONS.index('touch object'))
