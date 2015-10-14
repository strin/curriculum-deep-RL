import numpy as np
import numpy.random as npr
from pyrl.tasks.task import Environment, Task

def expected_reward_tabular(policy, task, tol=1e-4):
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
            poss_actions = policy.get_action_distribution(state)

            total_val = 0.
            for action, log_prob in poss_actions.items():
                val = 0.
                ns_dist = task.next_state_distribution(state, action)
                for ns, prob in ns_dist:
                    val += prob * (task.get_reward(state, action, ns) +
                                    task.gamma * V[ns])
                total_val += np.exp(log_prob) * val

            diff = abs(V[state] - total_val)
            V[state] = total_val

            if diff > max_diff:
                max_diff = diff

        if max_diff < tol:
            break


