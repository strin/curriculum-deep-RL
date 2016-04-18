# monte carlo tree search algorithm.
from pyrl.common import *
from pyrl.agents.agent import RandomPolicy
import pyrl.prob as prob

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
            curr_qval = self.qval.get(state, action)
            curr_count = self.uct.count_sa(state, action)
            if curr_qval == None:
                assert(curr_count == 0)
                curr_qval = 0.
            next_qval = float(curr_qval * curr_count + reward) / (curr_count + 1)
            self.qval.set(state, action, next_qval)
            self.uct.visit(state, action)


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
                    meta['phase'] = 'expansion'
                elif phase_expansion: # expand.
                    meta['phase'] = 'expansion'
                    action = self.random_policy.get_action(curr_state, valid_actions=task.valid_actions)
                else: # select.
                    meta['phase'] = 'selection'
                    action = self.qval.get_action(curr_state, valid_actions=task.valid_actions, method='uct', uct=self.uct, param_c=self.param_c)

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


class Astar(object):
    '''
    simple monte carlo tree search algorithm.
    '''
    def __init__(self, qval, uct, rb, gamma=0.95, param_c=0.1):
        self.qval = qval
        self.random_policy = RandomPolicy()
        self.uct = uct
        self.gamma = gamma
        self.param_c = param_c
        self.rb = rb


    def backprop(self, history):
        '''
        given an episode, update qval and uct.
        '''
        reward = 0.
        for (state, action, r, meta) in history[::-1]:
            reward = r + reward * self.gamma
            curr_qval = self.qval.get(state, action)
            curr_count = self.uct.count_sa(state, action)
            if curr_qval == None:
                assert(curr_count == 0)
                curr_qval = 0.
            next_qval = float(curr_qval * curr_count + reward) / (curr_count + 1)
            self.qval.set(state, action, next_qval)
            self.uct.visit(state, action)


    def run(self, task, num_episodes=100, tol=1e-4, debug=False):
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

            task.reset()


            while True:
                if num_steps >= np.log(tol) / np.log(self.gamma) or task.is_end():
                    self.backprop(history)
                    break

                curr_state = task.curr_state
                meta = {}

                unvisited_actions = [action for action in task.valid_actions if not self.qval.get(curr_state, action)]

                if not phase_expansion and unvisited_actions: # can we switch back to qval if unvisited is empty?
                    phase_expansion = True

                if phase_expansion: # expand.
                    meta['phase'] = 'expansion'
                    last_action = prob.choice(unvisited_actions, 1)[0]
                    last_state = curr_state
                    task.step(last_action)
                    bound = self.rb.av(curr_state)
                    value = max([bound[action] for action in task.valid_actions])
                    history.append((last_state, last_action, value, meta))
                    self.backprop(history)
                    break
                else: # select.
                    meta['phase'] = 'selection'
                    action = self.qval.get_action(curr_state, valid_actions=task.valid_actions, method='eps-greedy', epsilon=0.)

                    reward = task.step(action)
                    cum_reward += reward

                    history.append((curr_state, action, reward, meta))
                num_steps += 1

            cum_rewards.append(cum_reward)
            print 'num_steps', num_steps

        task.reset()
        return np.mean(cum_rewards)
