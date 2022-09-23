# this was written by u1200682 for CS 6350

# imports
import pandas as pd
import numpy as np
import Functions
import pprint

# import training data
training_data = pd.read_csv("test.csv") 

#import test data
test_data = pd.read_csv("test.csv")

# build the tree using the training data
gain = 'IG'
max_depth =6
tree = Functions.learn_tree(training_data, gain=gain, max_depth=max_depth)
pprint.pprint(tree)
accuracy = Functions.evaluate(tree, test_data, 'Values')
print(accuracy)