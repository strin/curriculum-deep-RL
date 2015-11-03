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
    def __init__(self, layout, agents, display, state_repr='stack', muteAgents=False, catchExceptions=False):
        '''
        state_repr: state representation, possible values ['stack', 'k-frames', 'dict']
            'stack' - stack walls, food, ghost and pacman representation into a 4D tensor.
            'dict' - return the raw dict representation keys=['walls', 'food', 'ghost', 'pacman'], values are matrix/tensor.
            'k-frames' - instead of directional descriptors for pacman and ghost, use static descriptors and capture past k frames.
        '''
        # parse state representation.
        self.state_repr = state_repr
        if self.state_repr.endswith('frames'):
            bar_pos = self.state_repr.rfind('frames')
            self.state_k = int(self.state_repr[:bar_pos-1])
            self.state_history = []
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
        if self.state_repr == 'dict':
            return self.curr_state_dict
        elif self.state_repr == 'stack':
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
        elif hasattr(self, 'state_k'):
            state_history = self.state_history + [self.curr_state_dict]
            def stack_direction(state_agent):
                return np.sum(state_agent, axis=2)
            state = []
            k = 0
            for hist_dict in state_history[::-1]:
                state.extend([
                        hist_dict['food'],
                        hist_dict['wall'],
                        stack_direction(hist_dict['pacman'])

                    ]
                        +
                    sum(
                        [
                                [stack_direction(ghost) for ghost in hist_dict['ghosts']]
                        ], []
                    )
                )
                k += 1
            state = np.array(state)
            frame_dim = state.shape[0] / k
            for ki in range(k + 1, self.state_k + 1):
                state = np.concatenate((state, np.zeros_like(state[:frame_dim, :, :])), axis=0)
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
        if hasattr(self, 'state_k'): # if we use past frames.
            self.state_history.append(self.curr_state_dict)
            if len(self.state_history) > self.state_k:
                self.state_history = self.state_history[-self.state_k:]

        if action not in self.valid_actions: # TODO: hack.
            action = self.dir_to_action[Directions.STOP]

        # convert action to direction.
        direction = self.action_to_dir[action]
        old_score = self.game.state.data.score

        # run the game using the direction.
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


