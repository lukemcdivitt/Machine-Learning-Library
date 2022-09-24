# this was written by u1200682 for CS 6350

# imports
import pandas as pd
import numpy as np
import Functions as Functions
import pprint
import os

# import training data
training_data = pd.read_csv("Data\train.csv")

#import test data
test_data = pd.read_csv("Data/test.csv")

# create single tree using manual inputs
gain = 'IG'
max_depth = 6
tree = Functions.learn_decision_tree(training_data, gain=gain, max_depth=max_depth)
pprint.pprint(tree)

# this was separated into a function so it does not run every time
def show_accuracy(training_data, test_data):
    # test the tree using different infomration gains methods and varying levels of depth
    test_gain = ['IG', 'ME', 'GI']
    test_max_depth = [1, 2, 3, 4, 5 ,6]
    all_data = [training_data, test_data]
    accuracy = np.zeros(len(test_gain) * len(test_max_depth) * len(all_data))
    iters = 0

    for tindex in range(0,2):
        for gindex in range(0, 3):
            for dindex in range(0,6):
                test_tree = Functions.learn_decision_tree(training_data, gain=test_gain[gindex], max_depth=test_max_depth[dindex])
                accuracy[iters] = Functions.evaluate(test_tree, all_data[tindex], 'Values')
                iters += 1

    accuracy.reshape((6,6))
    print(1-accuracy)

# uncomment this to run and show prediction accuracy
show_accuracy(training_data, test_data)
