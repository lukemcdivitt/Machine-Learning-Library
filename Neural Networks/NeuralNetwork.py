## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import numpy as np
import random as rd

# setup the network
def setup_network(num_inputs, num_outputs, num_hidden):

    # setup the list
    neural_net = list()

    # initialize the hidden layers
    hidden = [{'w':[rd() for idx_inp in range(0,1+num_inputs)]} for idx_hid in range(0,num_hidden)]
    neural_net.append(hidden)

    # initialize the output layers
    output = [{'w':[rd() for idx_hid in range(0,1+num_hidden)]} for idx_out in range(0,num_outputs)]
    neural_net.append(output)

    return neural_net

# create a function to create s
def s(w, inputs):

    # calculate the s value
    s = w[-1]

    # find the sum times the weights
    for idx in range(len(w)-1):
        s += w[idx] * inputs[idx]

    return s

# create the sigmoid function
def sigmoid(s):
    return 1.0 / (1/0 + np.exp(-s))

# pass through the network forward to generate an output
def forward_pass(neural_net, case):

    # setup the inputs
    inputs = case

    # pass through the nuearl network, calculating nodes along the way
    for level in neural_net:
        new_in = []
        for node in level:
            s_node = s(node['w'], inputs)
            node['output'] = sigmoid(s_node)
            new_in.append(node['output'])
            inputs = new_in
    
    return inputs

# get simga(s) prime
def sigmoid_prime(sig):
    return sig*(1.0-sig)

# pass through network backward and store error
def backward_pass(neural_net, e_value):

    for idx in reversed(range(len(neural_net))):

        level = neural_net[idx]
        errs = list()

        if idx != len(neural_net)-1:

            for idx_l in range(0,len(level)):

                err = 0.0

                for node in neural_net[idx + 1]:

                    err += (node['w'][idx_l] * node['del'])
                
                errs.append(err)

        else:

            for idx_l in range(0,len(level)):

                node = level[idx_l]
                errs.append(node['output'] - e_value[idx_l])

            for idx_l in range(0,len(level)):

                node = level[idx_l]
                node['del'] = errs[idx_l] * sigmoid_prime(node['output'])

    return neural_net

# update the weights within the network
def w_update(neural_net, case, rate):

    for idx in range(0,len(neural_net)):

        inputs = case[:-1]

        if idx != 0:

            inputs = [node['output'] for node in neural_net[idx - 1]]

        for node in neural_net[idx]:
            for idx_l in range(0,len(inputs)):

                node['w'][idx_l] -= rate * node['del'] * inputs[idx_l]

            node['w'][-1] -= rate * node['delta']

    return neural_net


# train the network
def train_neural_net(neural_net, training_data, rate, T, num_outputs):

    for epoch in range(T):

        err_sum = 0

        for case in training_data:

            out = forward_pass(neural_net, case)
            e_value = [0 for idx in range(num_outputs)]
            e_value[case[-1]] = 1
            err_sum += sum([(e_value[idx]-out[idx])**2 for idx in range(0,len(e_value))])
            neural_net = backward_pass(neural_net, e_value)
            neural_net = w_update(neural_net,case,rate)

    return neural_net

# create the trained network
