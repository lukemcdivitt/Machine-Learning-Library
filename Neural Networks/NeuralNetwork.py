## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import numpy as np
import random as rd

# setup the network
def setup_network(num_inputs, num_outputs, num_hidden, init):

    # setup the list
    neural_net = list()

    # initialize the hidden layers
    if init == 1:
        hidden = [{'w':[np.random.normal() for idx_inp in range(0,1+num_inputs)]} for idx_hid in range(0,num_hidden)]
        neural_net.append(hidden)

        hidden = [{'w':[np.random.normal() for idx_inp in range(0,1+num_hidden)]} for idx_hid in range(0,num_hidden)]
        neural_net.append(hidden)

        # initialize the output layers
        output = [{'w':[np.random.normal() for idx_hid in range(0,1+num_hidden)]} for idx_out in range(0,num_outputs)]
        neural_net.append(output)
    else:
        hidden = [{'w':[0 for idx_inp in range(0,1+num_inputs)]} for idx_hid in range(0,num_hidden)]
        neural_net.append(hidden)

        hidden = [{'w':[0 for idx_inp in range(0,1+num_hidden)]} for idx_hid in range(0,num_hidden)]
        neural_net.append(hidden)

        # initialize the output layers
        output = [{'w':[0 for idx_hid in range(0,1+num_hidden)]} for idx_out in range(0,num_outputs)]
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
    return 1.0 / (1.0 + np.exp(-s))

# pass through the network forward to generate an output
def forward_pass(neural_net, case):

    # setup the inputs
    inputs = case

    # pass through the nuearl network, calculating nodes along the way
    for level in neural_net[:-1]:
        new_in = []
        for node in level:
            s_node = s(node['w'], inputs)
            node['output'] = sigmoid(s_node)
            new_in.append(node['output'])
        inputs = new_in

    level = neural_net[-1]
    new_in = []
    for node in level:
        s_node = s(node['w'], inputs)
        node['output'] = s_node
        new_in.append(node['output'])
    inputs = new_in
    
    return inputs, neural_net

# get simga(s) prime
def sigmoid_prime(sig):
    return sig*(1.0-sig)

# pass through network backward and store error
def backward_pass(neural_net, e_value, case):

    case = case[:-1]

    for idx in reversed(range(len(neural_net))):

        level = neural_net[idx]
        errs = list()

        if idx != len(neural_net)-1 and idx != 0:

            for idx_l in range(0,len(level)):

                del_n = 0.0
                node = level[idx_l]

                for idx_w in range(len(neural_net[idx+1])):
                    del_n += neural_net[idx+1][idx_w]['del_n'] * neural_net[idx+1][idx_w]['w'][idx_l]

                node['del_n'] = del_n
                del_w = []

                for idx_n in range(0,len(node['w'])-1):

                    del_w.append(node['del_n'] * sigmoid_prime(node['output']) * neural_net[idx-1][idx_n]['output'])

                del_w.append(node['del_n'] * sigmoid_prime(node['output']) * 1)
                node['del_w'] = del_w
        
        elif idx == 0:

            for idx_l in range(0,len(level)):

                del_n = 0.0
                node = level[idx_l]

                for idx_w in range(len(neural_net[idx+1])):
                    del_n += neural_net[idx+1][idx_w]['del_n'] * neural_net[idx+1][idx_w]['w'][idx_l]

                node['del_n'] = del_n
                del_w = []

                for idx_n in range(0,len(node['w'])-1):

                    del_w.append(node['del_n'] * sigmoid_prime(node['output']) * case[idx_n])

                del_w.append(node['del_n'] * sigmoid_prime(node['output']) * 1)
                node['del_w'] = del_w

        else:

            for idx_l in range(0,len(level)):

                node = level[idx_l]
                errs.append(node['output'] - e_value[idx_l])

            for idx_l in range(0,len(level)):

                node = level[idx_l]
                node['del_n'] = errs[idx_l]

                del_w = []

                for idx_n in range(0,len(node['w'])-1):

                    del_w.append(errs[idx_l] * neural_net[idx-1][idx_n]['output'])

                del_w.append(errs[idx_l] * 1)
                node['del_w'] = del_w


    return neural_net

# update the weights within the network
def w_update(neural_net, case, rate):

    for idx in range(0,len(neural_net)):

        inputs = case[:-1]

        if idx != 0:

            inputs = [node['output'] for node in neural_net[idx - 1]]

        for node in neural_net[idx]:
            for idx_l in range(0,len(inputs)):

                node['w'][idx_l] -= rate * node['del_w'][idx_l] ##############################

            node['w'][-1] -= rate * node['del_w'][-1]

    return neural_net


# train the network
def train_neural_net(neural_net, training_data, gamma0, d, T, num_outputs):

    error = []
    ep = []

    for epoch in range(T):

        training_data = rd.sample(training_data,len(training_data))
        rate = rate_schedule(gamma0, d, epoch)
        err_sum = 0.0

        for case in training_data:

            out,neural_net = forward_pass(neural_net, case)
            e_value = [0 for idx in range(num_outputs)]
            e_value[int(case[-1])] = 1
            err_sum += sum([(e_value[i]-out[i])**2 for i in range(len(e_value))])
            neural_net = backward_pass(neural_net, e_value, case)
            neural_net = w_update(neural_net,case,rate)
    
        error.append(err_sum)
        ep.append(epoch)
        # if epoch % 10 == 0:  
        #     print("epoch = %4d   loss = %0.4f" % (epoch, err_sum))

    return neural_net,error,ep

# create the trained network
def create_network(training_data, num_inputs, num_outputs, num_hidden, gamma0, d, init=1, T=10):

    neural_net = setup_network(num_inputs,num_outputs,num_hidden, init)
    trained_nn,error,ep = train_neural_net(neural_net,training_data,gamma0,d,T,num_outputs)

    return trained_nn,error,ep

# make predictions with the NN
def predict(neural_net, case):
    out,neural_net = forward_pass(neural_net, case)
    return out.index(max(out))

# find accuracy
def accuracy(neural_net,test_data):
    incorrect = 0
    for row in test_data:
        pred = predict(neural_net, row[:-1])
        if pred != row[-1]:
            incorrect += 1
    
    return incorrect/len(test_data)

# set training schedule
def rate_schedule(gamma0,d,T):
    return (gamma0 / (1 + (gamma0/d)*T))

# setup the known test case
def TestCase():
    num_inputs = 2
    num_hidden = 2
    num_outputs = 1
    neural_net = setup_network(num_inputs, num_outputs, num_hidden,1)

# setup a test network (from written portion)
    w3 = [2, -1.5, -1]
    w21 = [-2, -3, -1]
    w22 = [2, 3, 1]
    w11 = [-2, -3, -1]
    w12 = [2, 3, 1]
    neural_net[2][0]['w'] = w3
    neural_net[1][0]['w'] = w21
    neural_net[1][1]['w'] = w22
    neural_net[0][0]['w'] = w11
    neural_net[0][1]['w'] = w12

    # complete a forward pass
    value,neural_net = forward_pass(neural_net, [1,1])

    # complete the backward pass
    e_value = [1]
    neural_net = backward_pass(neural_net, e_value, [1,1,1])

    return value, neural_net