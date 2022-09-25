# this was written by u1200682 for CS 6350

# imports
from re import T
import pandas as pd
import numpy as np
import Functions
import pprint

# This section is for the car data ***********************************************
# import training data
training_data = pd.read_csv("train_car.csv")

#import test data
test_data = pd.read_csv("test_car.csv")

# create single tree using manual inputs
gain = 'IG'
max_depth = 6
# tree_car = Functions.learn_decision_tree(training_data, gain=gain, max_depth=max_depth)
# pprint.pprint(tree_car)

# this was separated into a function so it does not run every time
def show_accuracy(training_data, test_data, max_depth):
    # test the tree using different infomration gains methods and varying levels of depth
    test_gain = ['IG', 'ME', 'GI']
    test_max_depth = np.arange(1,max_depth+1,1)
    test_accuracy = np.zeros(len(test_gain) * len(test_max_depth))
    train_accuracy = np.zeros(len(test_gain) * len(test_max_depth))
    iters = 0

    for gindex in range(0, 3):
        for dindex in range(0,len(test_max_depth)):
            test_tree = Functions.learn_decision_tree(training_data, gain=test_gain[gindex], max_depth=test_max_depth[dindex])
            test_accuracy[iters] = Functions.evaluate(test_tree, test_data, test_data.keys()[-1])
            train_accuracy[iters] = Functions.evaluate(test_tree, training_data, training_data.keys()[-1])
            print('Test:')
            print(test_accuracy[iters])
            print('Train:')
            print(train_accuracy[iters])
            iters += 1

    test_accuracy.reshape((int(len(test_accuracy)/len(test_max_depth)),len(test_max_depth)))
    train_accuracy.reshape((int(len(test_accuracy)/len(test_max_depth)),len(test_max_depth)))
    return test_accuracy, train_accuracy

# uncomment this to run and show prediction accuracy for
# show_accuracy(training_data, test_data, max_depth=max_depth)

# This section is for the bank data ***********************************************
# import training data
training_data = pd.read_csv("train_bank.csv")

#import test data
test_data = pd.read_csv("test_bank.csv")

# declare numeric input variables
numeric_keys = ['Age', 'Balance', 'Day', 'Duration', 'Campaign', 'Pdays', 'Previous']

# set whether top replace uknown or not
replace = 0

# convert pandas dataframe with these values to integers
for key in range(0,len(numeric_keys)):
    pd.to_numeric(training_data[numeric_keys[key]])
    pd.to_numeric(test_data[numeric_keys[key]])

# count the frequecy of elements in poutcome
label = training_data.keys()
counts = {}
for element in test_data[label[15]]:
    if element in counts:
        counts[element] += 1
    else:
        counts[element] = 1

# convert integers to binary
for key in range(0,len(label)):
    if isinstance(training_data[label[key]][0], str):
        pass
    else:
        training_data[label[key]] = Functions.convert_to_binary(training_data[label[key]])
        test_data[label[key]] = Functions.convert_to_binary(test_data[label[key]])

    if replace == 1:
        test_uni, test_pos = np.unique(test_data[label[key]],return_inverse=True)
        train_uni, train_pos = np.unique(training_data[label[key]],return_inverse=True)

        test_count = np.bincount(test_pos)
        train_count = np.bincount(train_pos)

        test_maxpos = test_count.argmax()
        train_maxpos = train_count.argmax()

        if train_uni[train_maxpos] == 'unknown':
            train_uni = np.delete(train_uni, train_maxpos)
            train_count = np.delete(train_count, train_maxpos)
            train_maxpos = train_count.argmax()

        if test_uni[test_maxpos] == 'unknown':
            test_uni = np.delete(test_uni, test_maxpos)
            test_count = np.delete(test_count, test_maxpos)
            test_maxpos = test_count.argmax()

        for idx in range(0,len(test_data[label[key]])):
            if training_data[label[key]][idx] == 'unknown':
                training_data[label[key]][idx] = train_uni[train_maxpos]
        
        
            if test_data[label[key]][idx] == 'unknown':
                test_data[label[key]][idx] = train_uni[train_maxpos]
    
    print(np.unique(test_data[label[key]], return_inverse=True))
    print(np.unique(training_data[label[key]], return_inverse=True))


# create single tree using manual inputs
gain = 'IG'
max_depth = 16
tree_bank = Functions.learn_decision_tree(training_data, gain=gain, max_depth=max_depth)
pprint.pprint(tree_bank)

# uncomment this to show the accuracy for the bank data
# [test_accuracy, train_accuracy] = show_accuracy(training_data, test_data, max_depth=max_depth)
def print_accuracy(test_accuracy, train_accuracy):
    print(test_accuracy)
    print(train_accuracy)
    print(np.mean(train_accuracy[0:15]))
    print(np.mean(train_accuracy[16:31]))
    print(np.mean(train_accuracy[32:47]))
    print(np.mean(test_accuracy[0:15]))
    print(np.mean(test_accuracy[16:31]))
    print(np.mean(test_accuracy[32:47]))

# print_accuracy(test_accuracy, train_accuracy)