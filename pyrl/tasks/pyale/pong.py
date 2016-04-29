from pyrl.tasks.task import Task
from pyrl.utils import mkdir_if_not_exist
from pyrl.common import *
from pyrl.config import floatX
from pyrl.prob import choice
from pyrl.tasks.pyale import PythonGame

class PongGame(PythonGame):
    def __init__(self, frames_per_action=1, state_type='ram', num_frames=4):
        PythonGame.__init__(self, game_path='pong.py', frames_per_action=frames_per_action)
        self.frames = []
        self.num_frames = num_frames
        self.state_type = state_type
        self.curr_raw_state = self.curr_state

        while len(self.frames) < self.num_frames:
            self.step(choice(self.valid_actions, 1)[0])
            if self.is_end():
                self.reset()

    @property
    def _curr_frame(self):
        state = self.curr_raw_state
        if self.state_type == 'ram':
            frame = np.array(state.values(), dtype=floatX)
        elif self.state_type == '1hot': # 1-hot representation.
            raise NotImplementedError()
        return frame


    @property
    def curr_state(self):
        return np.array(self.frames)


    def step(self, action):
        reward = PythonGame.step(self, action)
        self.curr_raw_state = PythonGame.curr_state.fget(self)
        if len(self.frames) == self.num_frames:
            self.frames = self.frames[1:]
        self.frames.append(self._curr_frame)
        return reward


