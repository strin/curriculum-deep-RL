class TaskSimulator(object):
    def __init__(self, task):
        '''
        learner:
            an abstraction of RL learning agents with
                get_action(curr_states) # get action on current state.
                send_feedback(reward)   # send reward to agent for last action.
        '''
        self.task = task
        self.total_steps = 0


    def run(self, learner, max_steps=None, callback=None):
        task = self.task
        num_steps = 0.
        cum_reward = 0.
        factor = 1.

        task.reset()
        curr_state = task.curr_state
        while True:
            # TODO: Hack!
            # print 'Lying and tell the agent the episode is over!'
            #if self.gamma < 1. and num_steps >= np.log(tol) / np.log(self.gamma):
            #    meta['curr_valid_actions'] = []
            #    self._end_episode(0, meta)
            #    break
            if max_steps is not None and num_steps > max_steps:
                break

            if callback:
                callback()

            if task.is_end():
                break

            action = learner.get_action(curr_state, task.valid_actions)
            reward = task.step(action)
            cum_reward += reward
            try:
                next_state = task.curr_state
                next_valid_actions = task.valid_actions
            except: # session has ended.
                next_state = None
                next_valid_actions = None
            learner.send_feedback(reward, next_state, next_valid_actions, task.is_end())

            curr_state = next_state
            num_steps += 1
        task.reset()
        self.total_steps += num_steps
        return cum_reward
