# this was written by u1200682 for CS 6350

# includes
import numpy as np
import pandas as pd
import SVM_Functions as svm

# import the dataset
training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# convert the classifiactions to 1 and -1
training_data = svm.convert_classifcation(training_data)
test_data = svm.convert_classifcation(test_data)
training_matrix = training_data.to_numpy()
x = training_matrix[:,:-1]
y = training_matrix[:,-1]
test_matrix = test_data.to_numpy()
x_test = test_matrix[:,:-1]
y_test = test_matrix[:,-1]

# set the parameters fo the algorithm
T = 1000
gamma0 = 0.001
a = 0.0001
C = [100.0/873.0, 500.0/873.0, 700.0/873.0]
schedule = 2

# run the subgradient descent
for idx in range(len(C)):
    w1 = svm.subgradient_descent(training_data, gamma0, a, T, C[idx], 1)
    w2 = svm.subgradient_descent(training_data, gamma0, a, T, C[idx], 2)
    error_train = svm.evaluate(training_data,w1)
    error_test = svm.evaluate(test_data,w1)
    error_train2 = svm.evaluate(training_data,w2)
    error_test2 = svm.evaluate(test_data,w2)
    print((w1))
    print(w2)
    print(error_test - error_test2)
    print(error_train - error_train2)

# minimize the dual form
# w,b = svm.get_weights(x,y,C[2])
# print(w)
# print(b)
# preds = svm.classify(x_test,w,b)
# err = svm.error(y_test,preds)
# print(err)

# # minimize with kernel
# alpha = svm.find_alphas_kernel(x, y, C[0], gamma0)
# b = svm.get_b_kernel(alpha,x,y,C[0],gamma0)
# print(b)
# preds = svm.classify_kernel(x_test,alpha,b,y,x,gamma0,C[0])
# preds2 = svm.classify_kernel(x,alpha,b,y,x,gamma0,C[0])
# err = svm.error(y_test,preds)
# print(err)
# err = svm.error(y,preds2)
# print(err)
