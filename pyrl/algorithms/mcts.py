# monte carlo tree search algorithm.
from pyrl.common import *
from pyrl.agents.agent import RandomPolicy

class MCTS(object):
    '''
    simple monte carlo tree search algorithm.
    '''
    def __init__(self, qval, uct, gamma=0.95, param_c=0.1):
        self.qval = qval
        self.random_policy = RandomPolicy()
        self.uct = uct
        self.gamma = gamma
        self.param_c = param_c


    def backprop(self, history):
        '''
        given an episode, update qval and uct.
        '''
        reward = 0.
        for (state, action, r, meta) in history[::-1]:
            reward = r + reward * self.gamma
            curr_qval = self.qval.get(state, action):
            curr_count = self.uct.count_sa(state, action)
            if not curr_qval:
                assert(curr_count == 0)
                curr_qval = 0.
            next_qval = float(curr_qval * curr_count + reward) / (curr_count + 1)
            self.uct.visit(state, action)


    def run(self, task, num_episodes=100, tol=1e-4):
        '''
        update qval every *num_epoch*
        for every *num_epoch*, run *num_episodes* of MCTS.
        '''
        cum_rewards = []

        for ei in range(num_episodes):
            num_steps = 0.
            cum_reward = 0.
            history = []
            phase_expansion = False

            while True:
                if num_steps >= np.log(tol) / np.log(self.gamma) or task.is_end():
                    self.backprop(history)
                    break

                curr_state = task.curr_state
                meta = {}

                unvisited_actions = [action for action in task.valid_actions if not self.qval.get(curr_state, action)]

                if unvisited_actions: # can we switch back to qval if unvisited is empty?
                    phase_expansion = True

                if phase_expansion: # expand.
                    policy = random_policy()
                    meta['phase'] = 'expansion'
                    action = self.dqn.get_action(curr_state, valid_actions=task.valid_actions)
                else: # select.
                    policy = self.qval
                    meta['phase'] = 'selection'
                    action = self.dqn.get_action(curr_state, valid_actions=task.valid_actions, method='uct', uct=self.uct, param_c=param_c)

                reward = task.step(action)
                cum_reward += reward

                history.append((curr_state, action, reward, meta))
                cum_rewards.append(cum_reward)

        return np.mean(cum_rewards)

