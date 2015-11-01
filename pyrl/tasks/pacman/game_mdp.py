import pyrl
from pyrl.tasks.task import Task
from util import *
from pacman import GameState, ClassicGameRules, loadAgent
from game import Game, Directions, Actions, Agent
import layout
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

    def getAction(self, state):
        print 'get action'
        self.action_event.wait()
        self.action_event.clear()
        action = self.next_action
        self.done_event.set()
        return action

class PacmanTask(Task):
    def __init__(self, layout, agents, display, muteAgents=False, catchExceptions=False ):
        self.init_state = GameState()
        self.init_state.initialize(layout, len(agents))
        self.action_event = Event()
        self.done_event = Event()
        self.myagent = GameWaitAgent(self.action_event, self.done_event)
        self.game_rule = ClassicGameRules(timeout=100)
        self.game = Game([self.myagent] + agents[:layout.getNumGhosts()],
                         display,
                         self.game_rule,
                         catchExceptions = catchExceptions)
        self.game.state = self.init_state
        # action mapping.
        self.all_actions = Actions._directions.keys()
        self.action_to_dir = {action_i: action
            for (action_i, action) in enumerate(self.all_actions)}
        self.dir_to_action = {action: action_i
            for (action_i, action) in enumerate(self.all_actions)}
        # start running game.
        def run_game():
            self.game.run()
        self.game_thread = Thread(target=run_game)
        self.game_thread.start()

    @property
    def curr_state(self):
        return self.game.state.data.array()

    @property
    def num_actions(self):
        return len(self.all_actions)

    @property
    def valid_actions(self):
        dirs = self.game.state.getLegalPacmanActions()
        return [self.dir_to_action[dir] for dir in dirs]

    def step(self, action):
        direction = self.action_to_dir[action]
        old_score = self.game.state.data.score
        self.myagent.next_action = direction
        self.action_event.set()
        self.done_event.wait()
        self.done_event.clear()
        new_score = self.game.state.data.score
        reward = new_score - old_score
        return reward


