import pyrl.prob as prob

# common utils for algorithms.
def generate_experience(policy, task, budget_experience, budget_per_episode=None, budget_episodes=None, state_attr='curr_state'):
    '''
    yield experiences by running policy on task.
    '''
    assert(budget_experience > 0)
    num_experience = 0
    num_episodes = 0
    experiences = []
    while num_experience < budget_experience:
        if budget_episodes and num_episodes >= budget_episodes:
            break
        num_step = 0
        task.reset()
        last_state_attr = getattr(task, state_attr)
        last_valid_actions = task.valid_actions
        while not budget_per_episode or num_step < budget_per_episode:
            if task.is_end() or num_experience >= budget_experience:
                break

            action = policy.get_action(task.curr_state, valid_actions=task.valid_actions)
            reward = task.step(action)

            curr_state_attr = getattr(task, state_attr)
            meta = {
                'curr_valid_actions': task.valid_actions,
                'last_valid_actions': last_valid_actions,
                'end': task.is_end()
            }
            experience = (last_state_attr, action, curr_state_attr, reward, meta)
            experiences.append(experience)
            last_state_attr = curr_state_attr
            last_valid_actions = task.valid_actions

            num_step += 1
            num_experience += 1
        num_episodes += 1

    task.reset()
    return experiences


def generate_experience_mt(policy, tasks, budget_experience, budget_per_episode=None, state_attr='curr_state'):
    experiences = []
    while len(experiences) < budget_experience:
        task = prob.choice(tasks, 1)[0]
        experiences.extend(generate_experience(policy, task, budget_experience - len(experiences), budget_per_episode, budget_episodes=1, state_attr=state_attr))
    return experiences


