# this was written by u1200682 for CS 6350

# imports
from re import T
import pandas as pd
import numpy as np
from numpy.linalg import norm

# gradient desc
def batch_descent(a, dataframe, ep=0.0001, max_iter=10000):
    conv = False
    iter = 0
    keys = dataframe.keys()
    m = len(dataframe[keys[0]]) # number of samples

    holder = dataframe.to_numpy()
    x = holder[:, :-1]
    y = holder[:, -1]
    cost_holder = []

    # initial b and w
    b = 0
    w = np.zeros(len(keys)-1)

    # error
    J = sum([(b + w*x[idx] - y[idx])**2 for idx in range(m)])

    # perform loop
    while not conv:
        # compute gradient
        gradient_b = 1.0/m * sum([(b + w*x[idx] - y[idx]) for idx in range(m)]) 
        gradient_h = 1.0/m * sum([(b + w*x[idx] - y[idx])*x[idx] for idx in range(m)])

        # update the the values
        bt = b - a * gradient_b
        wt = w - a * gradient_h
        b = bt
        w = wt

        # msse
        e = sum( [ (b + w*x[idx] - y[idx]) for idx in range(m)]) 
        gradient_h = 1.0/m * sum([(b + w*x[idx] - y[idx])**2 for idx in range(m)] ) 

        if (norm(J - e)) <= ep:
            print("Converged")
            conv = True
    
        J[:] = e   
        iter += 1
        cost_holder.append(np.mean(J))
    
        if iter == max_iter:
            print('Did not Coinverge')
            conv = True

    return b,w,J,e,cost_holder