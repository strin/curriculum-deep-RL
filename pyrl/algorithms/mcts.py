# monte carlo tree search algorithm.
from pyrl.common import *
from pyrl.agents.agent import RandomPolicy
import pyrl.prob as prob

class MCTS(object):
    '''
    simple monte carlo tree search algorithm.
    '''
    def __init__(self, qval, uct, gamma=0.95, param_c=0.1, rb=None, lam=1., default_policy='random'):
        self.qval = qval
        self.random_policy = RandomPolicy()
        self.default_policy = default_policy
        self.uct = uct
        self.gamma = gamma
        self.param_c = param_c
        self.lam = lam
        self.rb = rb


    def backprop(self, history):
        '''
        given an episode, update qval and uct.
        '''
        reward = 0.
        V = 0.
        Vx = None
        next_state = None
        next_vas = None
        has_next = False
        for (state, action, r, meta) in history[::-1]:
            if meta['phase'] == 'expansion':
                reward = r + reward * self.gamma * self.lam
                if self.rb and self.lam < 1. and has_next:
                    rv = max([self.rb.av(next_state)[a] for a in next_vas])
                    V = rv + self.lam * V

            elif meta['phase'] == 'selection':
                if not Vx:
                    if self.rb and self.lam < 1. and has_next: # 0-th TD backup.
                        rv = max([self.rb.av(next_state)[a] for a in next_vas])
                        V = rv + self.lam * V
                    Vx = (1 - self.lam) * V + self.lam * reward
                Vx = r + self.gamma * Vx
                curr_qval = self.qval.get(state, action)
                curr_count = self.uct.count_sa(state, action)
                if curr_qval == None:
                    assert(curr_count == 0)
                    curr_qval = 0.
                next_qval = float(curr_qval * curr_count + Vx) / (curr_count + 1)
                self.qval.set(state, action, next_qval)
                self.uct.visit(state, action)

            next_state = state
            next_vas = meta['valid_actions']
            has_next = True



    def run(self, task, num_episodes=100, num_steps=float('inf'), tol=1e-4, debug=False):
        '''
        update qval every *num_epoch*
        for every *num_epoch*, run *num_episodes* of MCTS.
        '''
        cum_rewards = []
        total_steps = 0.

        for ei in range(num_episodes):
            count_steps = 0.
            cum_reward = 0.
            factor = 1.
            history = []
            phase_expansion = False

            task.reset()

            while True:
                if total_steps > num_steps or count_steps >= np.log(tol) / np.log(self.gamma) or task.is_end():
                    self.backprop(history)
                    break

                curr_state = task.curr_state
                meta = {}

                unvisited_actions = [action for action in task.valid_actions if not self.qval.get(curr_state, action)]

                if not phase_expansion and unvisited_actions: # can we switch back to qval if unvisited is empty?
                    phase_expansion = True
                    action = prob.choice(unvisited_actions, 1)[0]
                    meta['phase'] = 'selection'
                elif phase_expansion: # expand.
                    meta['phase'] = 'expansion'
                    if self.default_policy == 'random':
                        action = self.random_policy.get_action(curr_state, valid_actions=task.valid_actions)
                    elif self.default_policy == 'rb-eps':
                        action = self.rb.get_action(curr_state, valid_actions=task.valid_actions, method='eps-greedy', epsilon=0.05)

                else: # select.
                    meta['phase'] = 'selection'
                    action = self.qval.get_action(curr_state, valid_actions=task.valid_actions, method='uct', uct=self.uct, param_c=self.param_c)

                meta['valid_actions'] = task.valid_actions

                reward = task.step(action)
                cum_reward = cum_reward + factor * reward
                factor *= self.gamma

                history.append((curr_state, action, reward, meta))
                count_steps += 1
                total_steps += 1

            cum_rewards.append(cum_reward)

            if total_steps > num_steps:
                break

        task.reset()
        print 'ei', ei
        return np.mean(cum_rewards)


