# some common architectures of deep networks.
# a architecture function should:
#   take input of (states, **kwargs)
#   and produce (output, model), where model is a dict of layers.
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import layers

def model_params(model):
    ''' get params in model and concatenate them into a list '''
    return sum([layer.params for layer in model.values()], [])

def two_layer(states, input_dim, hidden_dim, output_dim):
    fc1 = layers.FullyConnected(input_dim=input_dim,
                                output_dim=hidden_dim,
                                activation='relu')

    fc2 = layers.FullyConnected(input_dim=hidden_dim,
                                output_dim=hidden_dim,
                                activation='relu')

    linear_layer = layers.FullyConnected(input_dim=hidden_dim,
                                            output_dim=output_dim,
                                            activation=None)

    model = {'fc1': fc1, 'fc2': fc2, 'linear': linear_layer}

    hidden_1 = fc1(states)
    hidden_2 = fc2(hidden_1)
    output = linear_layer(hidden_2)

    return (output, model)

def fully_connected(states, arch_list):
    params = []
    model = []
    for (li, layer) in enumerate(arch_list[1:-1]):
        fc = layers.FullyConnected(arch_list[li], arch_list[li+1], activation='relu')
        params.extend(fc.params)
        model.append(fc)
    linear_layer = layers.FullyConnected(arch_list[-2], arch_list[-1], activation=None)
    model.append(linear_layer)
    params.extend(linear_layer.params)

    # construct computational graph.
    hidden = [model[0](states)]
    for (fi, fc) in enumerate(model[1:-1]):
        hidden.append(model[fi+1](hidden[-1]))
    final_output = model[-1](hidden[-1])
    return (final_output, model)


def GridWorld_5x5_FCN(states, input_dim):
    params = []
    ## agent.
    H_AGENT_DIM1 = 5
    fc_agent1 = layers.FullyConnected(input_dim, H_AGENT_DIM1, activation='relu')
    params.extend(fc_agent1.params)
    h_agent1 = fc_agent1(states[:, :input_dim])
    ## grid.
    H_GRID_DIM1 = 10
    fc_grid1 = layers.FullyConnected(input_dim, H_GRID_DIM1, activation='relu')
    params.extend(fc_grid1.params)
    h_grid1 = fc_grid1(states[:, input_dim:2*input_dim])
    # (TODO:) grid layer 2.
    ## goal.
    H_GOAL_DIM1 = H_AGENT_DIM1 # symmetric
    fc_goal1 = layers.FullyConnected(input_dim, H_GOAL_DIM1, activation='relu')
    params.extend(fc_goal1.params)
    h_goal1 = fc_goal1(states[:, 2*input_dim:3*input_dim])
    ##demons.
    H_DEMONS_DIM1 = 5
    fc_demons1 = layers.FullyConnected(input_dim, H_DEMONS_DIM1, activation='relu')
    params.extend(fc_demons1.params)
    h_demons1 = fc_demons1(states[:, 3*input_dim:4*input_dim])
    ## combine them all!
    v_joint = T.concatenate([h_agent1, h_grid1, h_goal1, h_demons1], axis=1)
    H_JOINT_DIM1 = 5
    fc_joint1 = layers.FullyConnected(H_AGENT_DIM1+H_GRID_DIM1+H_GOAL_DIM1+H_DEMONS_DIM1, H_JOINT_DIM1, activation='relu')
    params.extend(fc_joint1.params)
    h_joint1 = fc_joint1(v_joint)
    linear_layer = layers.FullyConnected(H_JOINT_DIM1, 1, activation=None)
    params.extend(linear_layer.params)
    output = linear_layer(h_joint1)

    return (params, output, {
            'fc_agent1': fc_agent1,
            'fc_grid1': fc_grid1,
            'fc_goal1': fc_goal1,
            'fc_demons1': fc_demons1,
            'fc_joint1': fc_joint1,
            'linear_layer': linear_layer
        })
