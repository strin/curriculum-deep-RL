import pyrl
from pyrl.tasks.task import Task
from util import *
from pacman import GameState, ClassicGameRules, loadAgent
from game import Game, Directions, Actions, Agent
import layout
import numpy as np
from threading import Thread, Event

class GameWaitAgent(Agent):
    """
    GameWaitAgent is a blocking agent.
    During a run of a game, if GameWaitAgent is called to
    get the action, it blocks to wait for a trigger event.


    """
    def __init__(self, action_event, done_event):
        Agent.__init__(self)
        self.next_action = None
        self.action_event = action_event
        self.done_event = done_event
        self.kill = False

    def getAction(self, state):
        self.action_event.wait()
        self.action_event.clear()
        if self.kill:
            return None
        action = self.next_action
        self.done_event.set()
        return action

class GameNoWaitAgent(Agent):
    """
    GameNoWaitAgent is a nonblocking agent.
    During a run of a game, if GameNoWaitAgent is called to
    get the action, it produces the cached next_action.

    """
    def __init__(self):
        Agent.__init__(self)
        self.next_action = None

    def getAction(self, state):
        return self.next_action

class PacmanTask(Task):
    def __init__(self, layout, agents, display, muteAgents=False, catchExceptions=False):
        self.init_state = GameState()
        self.init_state.initialize(layout, len(agents))
        self.game_rule = ClassicGameRules(timeout=100)
        # action mapping.
        self.all_actions = Actions._directions.keys()
        self.action_to_dir = {action_i: action
            for (action_i, action) in enumerate(self.all_actions)}
        self.dir_to_action = {action: action_i
            for (action_i, action) in enumerate(self.all_actions)}

        def start_game():
            self.action_event = Event()
            self.done_event = Event()
            self.myagent = GameNoWaitAgent()
            self.game = Game([self.myagent] + agents[:layout.getNumGhosts()],
                            display,
                            self.game_rule,
                            catchExceptions = catchExceptions)
            self.game.state = self.init_state
            self.game.init()

        self.start_game = start_game
        start_game()

    @property
    def curr_state_dict(self):
        return self.game.state.data.array()

    @property
    def curr_state(self):
        state_dict = self.curr_state_dict
        state = np.array(
            [
                state_dict['food'],
                state_dict['wall']
            ]
            +
            [
                state_dict['pacman'][:, :, i] for i in range(4)
            ]
            +
            sum(
                [
                    [
                        ghost[:, :, i] for i in range(4)
                    ]
                    for ghost in state_dict['ghosts']
                ], []
            )
        )
        return state

    def is_end(self):
        return self.game.gameOver

    @property
    def num_actions(self):
        return len(self.all_actions)

    @property
    def valid_actions(self):
        dirs = self.game.state.getLegalPacmanActions()
        return [self.dir_to_action[dir] for dir in dirs]

    def step(self, action):
        if action not in self.valid_actions: # TODO: hack.
            action = self.dir_to_action[Directions.STOP]
        direction = self.action_to_dir[action]
        old_score = self.game.state.data.score
        self.myagent.next_action = direction
        self.game.run_one()
        new_score = self.game.state.data.score
        reward = new_score - old_score
        if self.is_end():
            self.game.finalize()
        return reward

    def reset(self):
        self.start_game()

    @property
    def state_shape(self):
        return self.curr_state.shape

    def __str__(self):
        return str(self.game.state)


