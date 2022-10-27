# this was written by u1200682 for CS 6350

# imports
from re import T
import pandas as pd
import numpy as np
import DT_Functions
import AB_Functions as ab
from matplotlib import pyplot as plt

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

# this is to generate the figures
iters=10
test_error = np.ones(iters)
training_error = np.ones(iters)
stump_training_error = np.ones(iters)
stump_test_error = np.ones(iters)

for idx in range(0,iters):
    classifiers = ab.Adaboost(idx+1)
    classifiers.build(training_data, gain)
    test_predictions = classifiers.predict(test_data)
    train_predictions = classifiers.predict(training_data)
    test_acc = ab.accuracy(test_data, test_predictions)
    train_acc = ab.accuracy(training_data, train_predictions)
    test_error[idx] = 1-test_acc
    training_error[idx] = 1-train_acc
    print(test_error)


t = np.arange(1,iters+1)
plt.plot(t,test_error,'r--',t,training_error,'k')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.title('Error v Iterations')
plt.legend(['Test Error', 'Training Error'])
plt.savefig('Figure1_Errors.png')
plt.show()
