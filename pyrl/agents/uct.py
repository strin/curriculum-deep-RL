# upper-confidence table for the agent to measure uncertainty.
import dill as pickle
from pyrl.agents.agent import StateTable

class UCT(object):
    def __init__(self):
        self.state_table = StateTable()


    def visit(self, state, action):
        curr_action_counts = self.state_table[state]
        if not curr_action_counts:
            curr_action_counts = {}
            self.state_table[state] = curr_action_counts
        if action not in curr_action_counts:
            curr_action_counts[action] = 0.
        curr_action_counts[action] += 1.


    def count_sa(self, state, action):
        curr_action_counts = self.state_table[state]
        if not curr_action_counts:
            return 0.
        if action not in curr_action_counts:
            return 0.
        return curr_action_counts[action]


    def count_s(self, state):
        curr_action_counts = self.state_table[state]
        if not curr_action_counts:
            return 0.
        return sum(curr_action_counts.values())








