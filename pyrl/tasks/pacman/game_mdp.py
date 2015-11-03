import pyrl
from pyrl.tasks.task import Task
from util import *
from pacman import GameState, ClassicGameRules, loadAgent
from game import Game, Directions, Actions, Agent, AgentState, Configuration
from pyrl.tasks.pacman.ghostAgents import DirectionalGhost
import layout
import numpy as np
import cStringIO
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

    def deepCopy(self):
        agent = GameNoWaitAgent()
        agent.next_action = self.next_action
        return agent

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
        self.layout = layout
        self.agents = agents
        self.display = display
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions

        if self.state_repr.endswith('frames'):
            bar_pos = self.state_repr.rfind('frames')
            self.state_k = int(self.state_repr[:bar_pos-1])
            self.state_history = []
        self.init_state = GameState()
        self.init_state.initialize(layout, len(agents))
        self.game_rule = ClassicGameRules(timeout=100)
        self.myagent = GameNoWaitAgent()
        self.init_game = Game([self.myagent] + agents[:layout.getNumGhosts()],
                        display,
                        self.game_rule,
                        catchExceptions = catchExceptions)
        self.init_game.state = self.init_state

        # action mapping.
        self.all_actions = Actions._directions.keys()
        self.action_to_dir = {action_i: action
            for (action_i, action) in enumerate(self.all_actions)}
        self.dir_to_action = {action: action_i
            for (action_i, action) in enumerate(self.all_actions)}

        def start_game():
            self.game = self.init_game.deepCopy()
            self.game.init()

        self.start_game = start_game
        start_game()

    def deep_copy(self):
        task = PacmanTask(self.layout, self.agents, self.display, self.state_repr, self.muteAgents, self.catchExceptions)
        task.game = self.game.deepCopy()
        task.myagent = self.myagent # TODO: agents not deep copy.
        return task

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
            for ki in range(k, self.state_k + 1):
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


class GameEditor(object):
    """
    provide primitive edit functions to game state data.
    """
    @staticmethod
    def _find_free_pos(data):
        config = data.array()
        width, height = data.layout.width, data.layout.height
        for x in range(width):
            for y in range(height):
                if (config['wall'][x, y] == 1 or
                    config['food'][x, y] == 1 or
                    any(map(lambda ghost: ghost[x, y].any() == 1, config['ghosts']))):
                    continue
                else:
                    yield (x, y)

    @staticmethod
    def move_pacman(game):
        '''
        move pacman to any free pos on the map.
        '''
        data = game.state.data
        # find pacman.
        def find_pacman(new_data):
            pacmanState = None
            for agentState in new_data.agentStates:
                if agentState.isPacman:
                    pacmanState = agentState
                    break
            assert(pacmanState)
            return pacmanState

        # create all valid modifications of pacman.
        game_nb = []
        for (x, y) in GameEditor._find_free_pos(data):
                for dir in Directions.ALL:
                    new_data = data.deepCopy()
                    pacmanState = find_pacman(new_data)
                    pacmanState.configuration.pos = (x, y) # change the state.
                    pacmanState.configuration.direction = dir
                    new_game = game.deepCopy()
                    new_game.state.data = new_data
                    game_nb.append(new_game)
        return game_nb

    @staticmethod
    def del_ghost(game):
        data = game.state.data
        game_nb = []
        if len(data.agentStates) == 1:
            return game_nb
        # try to delete one of the ghosts.
        for ind in range(1, len(data.agentStates)): # TODO: abusing variable naming convention here, CS188 and pyrl use different conventions.
            if ind == 0:
                continue
            new_game = game.deepCopy()
            rm_ind = lambda xs: [x for (xi, x) in enumerate(xs) if xi != ind]
            new_game.state.data.agentStates = rm_ind(data.agentStates)
            new_game.agents = rm_ind(game.agents)
            new_game.totalAgentTimes = rm_ind(game.totalAgentTimes)
            new_game.totalAgentTimeWarnings = rm_ind(game.totalAgentTimeWarnings)
            new_game.agentOutput = rm_ind(game.agentOutput)
            game_nb.append(new_game)
        return game_nb

    @staticmethod
    def add_ghost(game, ghost_type=DirectionalGhost):
        data = game.state.data
        game_nb = []
        for (x, y) in GameEditor._find_free_pos(data):
            for dir in Directions.ALL:
                agentState = AgentState(Configuration((x, y), dir), isPacman=False)
                new_game = game.deepCopy()
                new_game.state.data.agentStates.append(agentState)
                new_game.agents.append(ghost_type(len(new_game.agents)))
                new_game.totalAgentTimes.append(0.)
                new_game.totalAgentTimeWarnings.append(0.)
                new_game.agentOutput.append(cStringIO.StringIO())
                game_nb.append(new_game)
            break
        return game_nb

    @staticmethod
    def move_ghost(game, ghost_index=1):
        data = game.state.data
        # create all valid modifications of pacman.
        game_nb = []
        for (x, y) in GameEditor._find_free_pos(data):
                for dir in Directions.ALL:
                    new_data = data.deepCopy()
                    ghostState =  data.agentStates[ghost_index]
                    ghostState.configuration.pos = (x, y) # change the state.
                    ghostState.configuration.direction = dir
                    new_game = game.deepCopy()
                    new_game.state.data = new_data
                    game_nb.append(new_game)
        return game_nb

    @staticmethod
    def move_ghosts(game):
        game_nb = []
        for ghost_index in range(1, len(game.agents)):
            game_nb.extend(GameEditor.move_ghost(game, ghost_index))
        return game_nb


class PacmanTaskShifter(object):
    """
    creates local edits to a PacmanTask
    """
    @staticmethod
    def neighbors(task, axis=['move_pacman']):
        game = task.game
        game_nb = []
        for ax in axis:
            game_nb += getattr(GameEditor, ax)(game)
        task_nb = []
        for new_game in game_nb:
            new_task = task.deep_copy()
            new_task.init_game = new_game
            new_task.reset()
            task_nb.append(new_task)
        return task_nb





