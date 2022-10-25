# this was written by u1200682 for CS 6350

# imports
from re import T
import pandas as pd
import numpy as np
import DT_Functions
import pprint

# This section is for the bank data ***********************************************
# import training data
training_data = pd.read_csv("bank_train.csv")

#import test data
test_data = pd.read_csv("bank_test.csv")

# declare numeric input variables
numeric_keys = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# set whether top replace uknown or not
replace = 0

# convert numerical features to binary
training_data, test_data = DT_Functions.num_to_bin(training_data, test_data, numeric_keys, replace)

# create stump
gain = 'IG'
max_depth = 1
tree_bank = DT_Functions.learn_decision_tree(training_data, gain=gain, max_depth=max_depth)
pprint.pprint(tree_bank)