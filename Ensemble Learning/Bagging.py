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

# implemnt bagging
iters = 5
acc_test,acc_train = bg.bagging(training_data, test_data, T=iters, sample_size=0.2)
print(acc_test)
print(acc_train)

t = np.arange(1,iters+1)
plt.plot(t,acc_test,'r--',t,acc_train,'k')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.title('Error v Iterations')
plt.legend(['Test Error', 'Training Error'])
#plt.savefig('Figure3_Errors.png')
plt.show()
