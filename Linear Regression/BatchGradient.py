# this was written by u1200682 for CS 6350

# imports
from re import T
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import BG_Functions as bg

# This section is for the concrete data ***********************************************
# import training data
training_data = pd.read_csv("concrete_train.csv")

#import test data
test_data = pd.read_csv("concrete_test.csv")

#learning rate
r = 0.1
ep = 0.0000001

# perform algoithm
b,w,J,e,cost = bg.batch_descent(r, training_data, ep, max_iter=10000)
print(np.mean(b))
print(w)

holder = test_data.to_numpy()
x = holder[:, :-1]
y = holder[:, -1]
m = len(y)
J = 1/2*sum([(b[1] + w*x[idx] - y[idx])**2 for idx in range(m)])
print(w)


iters = np.arange(1,len(cost)+1)
plt.plot(iters,cost,'r--')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.title('Cost v Iterations')
# plt.legend(['Test Error', 'Training Error'])
plt.savefig('Cost.png')
plt.show()

