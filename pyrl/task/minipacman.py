import theano
import theano.tensor as T
import numpy as np
import layers
from theano.tensor.signal.downsample import max_pool_2d

from nn import FeedforwardNet

def MiniPacmanCNN(H, W, l2_reg=0.0, epsilon=0.05, batch_size=32, step_size=1e-3,
        F_AGENT_DIM1 = 3,
        F_AGENT_SIZE1 = (1, 1),
        F_AGENT_POOL1 = (2, 2)):
    '''
    dim: dimension of the grid vector (also agent vec, demons vec, goal vec).
    '''
    def arch_func(states):
        '''
        assume states (#data x H x W x #components)
        '''
        params = []
        ## agent.
        H_AGENT_DIM1 = (int((H - F_AGENT_SIZE1[0] + 1) / F_AGENT_POOL1[0]) 
                * int((W - F_AGENT_SIZE1[1] + 1) / F_AGENT_POOL1[1]) 
                * F_AGENT_DIM1)    # concatenated output of filter dimension.
        cnn_agent1 = layers.Conv(1, F_AGENT_DIM1, filter_size=F_AGENT_SIZE1, pool_size=F_AGENT_POOL1, activation='relu')
        params.extend(cnn_agent1.params)
        h_agent1 = T.flatten(cnn_agent1(states[:, 0:1, :, :]), outdim=2)

        ## goal.
        F_GOAL_DIM1 = F_AGENT_DIM1  # thanks to symmetry.
        F_GOAL_SIZE1 = F_AGENT_SIZE1
        F_GOAL_POOL1 = F_AGENT_POOL1
        H_GOAL_DIM1 = H_AGENT_DIM1
        H_GOAL_DIM1 = H_AGENT_DIM1 
        cnn_goal1 = layers.Conv(1, F_GOAL_DIM1, filter_size=F_GOAL_SIZE1, pool_size=F_GOAL_POOL1, activation='relu')
        params.extend(cnn_goal1.params)
        h_goal1 = T.flatten(cnn_goal1(states[:, 2:3, :, :]), outdim=2)

        ## (TODO:) grid.
        ## (TODO:) demons.

        ## combine them all!
        v_joint = T.concatenate([h_agent1, h_goal1], axis=1)

        H_JOINT_DIM1 = 4
        fc_joint1 = layers.FullyConnected(H_AGENT_DIM1+H_GOAL_DIM1, H_JOINT_DIM1, activation='relu')
        params.extend(fc_joint1.params)
        h_joint1 = fc_joint1(v_joint) 

        linear_layer = layers.FullyConnected(H_JOINT_DIM1, 1, activation=None)
        params.extend(linear_layer.params)
        output = linear_layer(h_joint1)

        return (params, output, {
                'fc_agent1': cnn_agent1,
                'fc_goal1': cnn_goal1,
                'fc_joint1': fc_joint1,
                'linear_layer': linear_layer
            })

    return FeedforwardNet(arch_func, l2_reg=l2_reg, epsilon=epsilon, batch_size=batch_size, step_size=step_size, tensor_type=T.tensor4)


def MiniPacmanHCNN(H, W, l2_reg = 0.0, epsilon = 0.05, batch_size = 32, step_size = 1e-3):
    '''
    A hierarchical CNN.

    - downsample the input by factors of 2 to construct scales.
    - for each scale of goal/agent, use CNN to compute features.
    - combine features for goal and gent through joint fully connected layers.
    '''
    (min_H, min_W) = (4, 4)
    H_DIM = 5
    check_pow2 = lambda x: x == int(x)
    assert check_pow2(np.log2(H / min_H))
    assert check_pow2(np.log2(W / min_W))
    def sub_arch_func(states, pool_size=(1,1)):
        params = []
        conv1 = layers.Conv(1, H_DIM, filter_size=(min_H, min_W), pool_size=pool_size, activation='relu')
        params.extend(conv1.params)
        h1 = conv1(states)
        return (params, h1, {
            'conv1': conv1
            })

    def arch_func(states):
        '''
        assume states (#data x H x W x #components)
        '''
        params = []
        agent = states[:, 0:1, :, :]
        goal = states[:, 2:3, :, :]
        ## agent.
        agent00 = max_pool_2d(agent, (2, 2), ignore_border=False) # scale 0, layer 0.
        (par, ha01, _) = sub_arch_func(agent00)  # scale 0, layer 1.
        params.extend(par)
        ha01 = T.flatten(ha01, outdim=2)

        goal00 = max_pool_2d(goal, (2, 2), ignore_border=False)
        (par, hg01, _) = sub_arch_func(goal00)
        params.extend(par)
        hg01 = T.flatten(hg01, outdim=2)

        h01 = T.concatenate((ha01, hg01), axis=1)
        H01_DIM = H_DIM * 2

        agent10 = agent
        (par, ha11, _) = sub_arch_func(agent10, pool_size=(2,2))
        params.extend(par)
        ha11 = T.flatten(ha11, outdim=2)

        goal10 = goal
        (par, hg11, _) = sub_arch_func(goal10, pool_size=(2,2))
        params.extend(par)
        hg11 = T.flatten(hg11, outdim=2)

        h11 = T.concatenate((ha11, hg11), axis=1)
        H11_DIM = 2 * 2 * H_DIM * 2

        ## combine them all!
        h = T.concatenate((h01, h11), axis=1)
        H_JOINT_DIM = H01_DIM + H11_DIM

        linear_layer = layers.FullyConnected(H_JOINT_DIM, 1, activation=None)
        params.extend(linear_layer.params)
        output = linear_layer(h)

        return (params, output, {
            })

    return FeedforwardNet(arch_func, l2_reg=l2_reg, epsilon=epsilon, batch_size=batch_size, step_size=step_size, tensor_type=T.tensor4)


def MiniPacmanHCNN2(H, W, l2_reg = 0.0, epsilon = 0.05, batch_size = 32, step_size = 1e-3):
    '''
    Similar to MiniPacmanHCNN
    difference

    - the agent and goal inputs are merged at the first layer.
    '''
    (min_H, min_W) = (4, 4)
    H_DIM = 10
    check_pow2 = lambda x: x == int(x)
    assert check_pow2(np.log2(H / min_H))
    assert check_pow2(np.log2(W / min_W))
    def sub_arch_func(states, pool_size=(1,1)):
        params = []
        conv1 = layers.Conv(1, H_DIM, filter_size=(min_H, min_W), pool_size=pool_size, activation='relu')
        params.extend(conv1.params)
        h1 = conv1(states)
        return (params, h1, {
            'conv1': conv1
            })

    def arch_func(states):
        '''
        assume states (#data x H x W x #components)
        '''
        params = []
        agent = states[:, 0:1, :, :]
        goal = states[:, 2:3, :, :]
        # ag = T.concatenate((agent, goal), axis=1)
        ag = agent + goal

        ag00 = max_pool_2d(ag, (2, 2), ignore_border=False) # scale 0, layer 0.
        (par, ha01, _) = sub_arch_func(ag00)  # scale 0, layer 1.
        params.extend(par)
        h01 = T.flatten(ha01, outdim=2)

        H01_DIM = H_DIM

        ag10 = ag
        (par, ha11, _) = sub_arch_func(ag10, pool_size=(2,2))
        params.extend(par)
        h11 = T.flatten(ha11, outdim=2)

        H11_DIM = 2 * 2 * H_DIM

        ## combine them all!
        h = T.concatenate((h01, h11), axis=1)
        H_JOINT_DIM = H01_DIM + H11_DIM

        linear_layer = layers.FullyConnected(H_JOINT_DIM, 1, activation=None)
        params.extend(linear_layer.params)
        output = linear_layer(h)

        return (params, output, {
            })

    return FeedforwardNet(arch_func, l2_reg=l2_reg, epsilon=epsilon, batch_size=batch_size, step_size=step_size, tensor_type=T.tensor4)

