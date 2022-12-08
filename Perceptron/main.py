# this was written by u1200682 for CS 6350

# includes
import numpy as np
import pandas as pd
import Perceptron as pe

# import the dataset
training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# set the parameters fo the algorithm
T = 10
learning_rate = 0.01
iterations = 10

# run the algorithm for voted perceptron
acc_voted = []
for blank in range(iterations):
    predictions, weights, c = pe.voted_evaluate(training_data, test_data, learning_rate, T)
    acc, counts = pe.accuracy(test_data, predictions)
    acc_voted.append(acc)
    print(blank)
print(np.mean(acc_voted))
submission = pd.DataFrame({'Weight Vectors':weights,'Counts':c})
pd.DataFrame(submission).to_csv("weigths.csv", index=False)

# run the algorithm for standard perceptron
acc_basic = []
for blank in range(iterations):
    predictions, weights = pe.evaluate(training_data, test_data, learning_rate, T)
    acc, counts = pe.accuracy(test_data, predictions)
    acc_basic.append(acc)
    print(blank)
print(np.mean(acc_basic))
print(weights)

# run the algorithm for the average perceptron
acc_averaged = []
for blank in range(iterations):
    predictions, weights = pe.averaged_evaluate(training_data, test_data, learning_rate, T)
    acc, counts = pe.accuracy(test_data, predictions)
    acc_averaged.append(acc)
    print(blank)
print(np.mean(acc_averaged))
print(weights)

