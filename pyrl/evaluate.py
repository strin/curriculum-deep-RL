import numpy as np
import numpy.random as npr
from pyrl.tasks.task import Task
from pyrl.algorithms.valueiter import compute_tabular_value

def reward_stochastic_samples(policy, task, gamma=0.95, num_trials = 100, budget=None, tol=1e-6, **args):
    total_reward = []

    for ni in range(num_trials):
        num_steps = 0
        task.reset()
        reward = 0.
        factor = 1.
        while num_steps < np.log(tol) / np.log(gamma):
            if task.is_end():
                break

            if budget and num_steps >= budget:
                break

            curr_state = task.curr_state
            # action = policy.get_action(curr_state, method='eps-greedy', epsilon=0., valid_actions=task.valid_actions)
            action = policy.get_action(curr_state, valid_actions=task.valid_actions, **args)
            reward += factor * task.step(action)
            factor *= gamma
            num_steps += 1
        total_reward.append(reward)
    task.reset()
    return total_reward

def qval_stochastic_samples(dqn, task, gamma=0.95, num_trials = 100, budget=20, **args):
    total_reward = []

    for ni in range(num_trials):
        num_steps = 0
        task.reset()
        reward = 0.
        factor = 1.
        while True:
            if task.is_end():
                break

            if num_steps >= budget:
                action_values = dqn.fprop(np.array([task.curr_state]))[0]
                value = max([action_values[act] for act in task.valid_actions])
                reward += factor * value
                break

            curr_state = task.curr_state
            # action = policy.get_action(curr_state, method='eps-greedy', epsilon=0., valid_actions=task.valid_actions)
            action = dqn.get_action(curr_state, valid_actions=task.valid_actions, **args)
            reward += factor * task.step(action)
            factor *= gamma
            num_steps += 1
        total_reward.append(reward)
    task.reset()
    return total_reward

def reward_stochastic(policy, task, gamma=0.95, num_trials=100, budget=None, tol=1e-6, **args):
    total_reward = reward_stochastic_samples(policy, task, gamma, num_trials, budget, tol, **args)
    return np.mean(total_reward)

def qval_stochastic(policy, task, gamma=0.95, num_trials=100, budget=20, **args):
    total_reward = qval_stochastic_samples(policy, task, gamma, num_trials, budget, **args)
    return np.mean(total_reward)

def reward_stochastic_mean_std(policy, task, gamma=0.05, num_trials=100, tol=1e-6, **args):
    total_reward = reward_stochastic_samples(policy, task, gamma, num_trials, tol, **args)
    return (np.mean(total_reward), np.std(total_reward) / np.sqrt(num_trials))

def reward_tabular(policy, task, tol=1e-4):
    '''
    compute exactly expected rewards averaged over start states.
    '''
    policy.task = task # configure the policy task in the multi-task setting.
    V = np.zeros(task.get_num_states())
    while True:
        # repeatedly perform reward bootstrapping on each state
        # given actions produced from the policy.
        max_diff = 0.

        # TODO: Add priority sweeping for state in xrange(self.num_states):
        for state in task.get_valid_states():
            if not policy.is_tabular():
                state_vector = task.wrap_stateid(state)
                poss_actions = policy.get_action_distribution(state_vector, method='eps-greedy', epsilon=0.01)
                # poss_actions = policy.get_action_distribution(state_vector, method='softmax', temperature=5e-2)
            else:
                poss_actions = policy.get_action_distribution(state)


            total_val = 0.
            for action, action_prob in poss_actions.items():
                val = 0.
                ns_dist = task.next_state_distribution(state, action)
                for ns, prob in ns_dist:
                    val += prob * (task.get_reward(state, action, ns) +
                                    task.gamma * V[ns])
                total_val += action_prob * val

            diff = abs(V[state] - total_val)
            V[state] = total_val

            if diff > max_diff:
                max_diff = diff

        if max_diff < tol:
            break
    return V

def expected_reward_tabular(policy, task, tol=1e-4):
    '''
    compute exactly expected rewards averaged over start states.
    '''
    V = reward_tabular(policy, task, tol)
    rewards = [V[state] for state in task.get_valid_states()]
    return np.mean(rewards)

def expected_reward_tabular_normalized(policy, task, tol=1e-4):
    '''
    compute the expected reward / reward by value iteration
    averaged over states.
    '''
    gtV = compute_tabular_value(task, tol) # ground truth values by value iteration.
    V = reward_tabular(policy, task, tol)
    rewards = [V[state] / gtV[state] for state in task.get_valid_states()]
    return np.mean(rewards)

def reward_tabular_normalized(policy, task, tol=1e-4):
    '''
    compute the expected reward / reward by value iteration
    averaged over states.
    '''
    gtV = compute_tabular_value(task, tol) # ground truth values by value iteration.
    V = reward_tabular(policy, task, tol)
    return V / gtV

def reward_tabular_normalized_fix_start(policy, task, tol=1e-4):
    '''
    compute the expected reward / reward by value iteration
    averaged over states.
    '''
    states = [task.start_state]
    gtV = compute_tabular_value(task, tol) # ground truth values by value iteration.
    V = reward_tabular(policy, task, tol)
    rewards = {state: V[state] / gtV[state] for state in task.get_valid_states()}
    return np.mean([rewards[state] for state in states])

def eval_dataset(policy, tasks, method=expected_reward_tabular_normalized):
    reward = 0.0
    for task in tasks:
        reward += method(policy, task, tol=1e-4)
    return reward / len(tasks)

def merge_line(xs, ys):
    """
    given a list of score lines, merge and average them into one.
    """

