import numpy as np
import numpy.random as npr
from pyrl.tasks.task import Environment, Task
from pyrl.algorithms.valueiter import compute_tabular_value

def reward_tabular(policy, task, tol=1e-4):
    '''
    compute exactly expected rewards averaged over start states.
    '''
    V = np.zeros(task.get_num_states())
    while True:
        # repeatedly perform reward bootstrapping on each state
        # given actions produced from the policy.
        max_diff = 0.

        # TODO: Add priority sweeping for state in xrange(self.num_states):
        for state in task.get_valid_states():
            if not policy.is_tabular():
                state_vector = task.wrap_stateid(state)
                poss_actions = policy.get_action_distribution(state_vector)
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

def eval_dataset(policy, tasks, method=expected_reward_tabular_normalized):
    reward = 0.0
    for task in tasks:
        reward += method(policy, task, tol=1e-4)
    return reward / len(tasks)
