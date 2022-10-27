# this was written by u1200682 for CS 6350

# imports
from re import T
import pandas as pd
import numpy as np
import DT_Functions
import AB_Functions as ab
from matplotlib import pyplot as plt
import BG_Functions as bg

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

# clarify which gain
gain = 'IG'

# define max features and number of trees
# max_features = [2, 4, 6]
num_trees = 500
max_depth = 4

# run algorithm
acc_test_2, acc_train_2 = bg.bagging_random(training_data, test_data, sample_size=0.2, T=num_trees, max_features=2, max_depth=max_depth)
acc_test_4, acc_train_4 = bg.bagging_random(training_data, test_data, sample_size=0.2, T=num_trees, max_features=4, max_depth=max_depth)
acc_test_6, acc_train_6 = bg.bagging_random(training_data, test_data, sample_size=0.2, T=num_trees, max_features=6, max_depth=max_depth)

t = np.arange(1,11)
plt.plot(t,acc_test_2,'r--',t,acc_train_2,'k--',t,acc_test_4,'r',t,acc_train_4,'k',t,acc_test_6,'r:',t,acc_train_6,'k:')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.title('Error v Iterations with Different Max Split Features')
plt.legend(['Test Error 2', 'Training Error 2', 'Test Error 4', 'Training Error 4', 'Test Error 6', 'Training Error 6'])
plt.savefig('Figure4_Errors.png')
plt.show()


