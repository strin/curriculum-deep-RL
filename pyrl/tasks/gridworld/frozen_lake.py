# this task is borrowed from Berkeley CS294 Deep Reinforcement Learning course.
from pyrl.tasks.task import DiscreteMDP
import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class FrozenLake(DiscreteMDP):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    def __init__(self, desc):
        self.desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = self.desc.shape
        self.maxxy = np.array([nrow-1, ncol-1])
        (startx,), (starty,) = np.nonzero(self.desc=='S')
        self.startstate = np.array([startx,starty])
        self.increments = np.array([[0,-1],[1,0],[0,1],[-1,0]])

    def start_state(self):
        return self.startstate

    def end_state(self, state):
        pos = self.state_to_pos(state)
        state_type = self.desc[pos]
        return state_type in 'GH'

    @property
    def num_actions(self):
        return 4

    @property
    def valid_actions(self, state):
        return range(4)

    @property
    def num_states(self):
        return self.nrow * self.ncol

    def step(self, state, a):
        pos = np.array(self.state_to_pos(state))
        a = (a + np.random.randint(-1,2)) % 4
        state = r, c = np.clip(pos + self.increments[a], [0,0], self.maxxy)
        return self.pos_to_state((r, c))

    def state_to_pos(self, state):
        return (int(state / self.ncol), state % self.ncol)

    def pos_to_state(self, pos):
        (r, c) = pos
        return self.ncol * r + c

    def reward(self, s, a, ns):
        state_type = self.desc[s]
        return float(state_type == 'G')

    def visualize(self, state):
        from jinja2 import Template
        from StringIO import StringIO
        import urllib, base64
        import matplotlib.pyplot as plt
        # create image.
        vis = np.zeros((self.nrow, self.ncol, 3), dtype=float)
        pos = self.state_to_pos(state)
        vis[self.desc == 'S', :] = np.array([248, 255, 255])
        vis[self.desc == 'F', :] = np.array([248, 247, 255])
        vis[self.desc == 'G', :] = np.array([255, 215, 55])
        vis[self.desc == 'H', :] = np.array([37, 39, 69])
        vis[pos[0], pos[1], :] = np.array([255, 80, 39])
        vis = vis / 255
        fig = plt.figure()
        plt.imshow(vis, interpolation='none')
        plt.axis('off')
        imgdata = StringIO()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0)
        imgdata = urllib.quote(base64.b64encode(imgdata.buf))
        plt.close()
        # create HTML content.
        template = Template('''
                            <center>
                                Frozen Lake ({{height}} x {{width}})
                                <img style="width: 200px" src="data:image/png;base64,{{imgdata}}"></img>
                            </center>
                            ''')
        return template.render(imgdata=imgdata, height=self.nrow, width=self.ncol)





