from pyrl.common import np
from pyrl.tasks.gridworld import GridWorldMultiGoal

def test_gridworld_multigoal():
    grid = np.zeros((2, 2))
    task = GridWorldMultiGoal(start_pos=(1, 0),
                             start_phase=0,
                             grid=grid,
                             action_stoch=0.,
                             goals=[(0, 0), (1, 1)])
    assert task.step(2) == 0.
    assert not task.is_end()
    assert task.phase == 1
    assert task.step(0) == 0.
    assert not task.is_end()
    assert task.phase == 1
    assert task.step(1) == 1.
    assert task.is_end()


