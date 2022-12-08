# this was written by u1200682 for CS 6350

# includes
from http.client import PRECONDITION_FAILED
from matplotlib import test
from matplotlib.offsetbox import AuxTransformBox
import numpy as np
import pandas as pd

# create a function to create the weights
def weights(training_data, learning_rate, T):
    keys = training_data.keys()
    w = [0.0 for idx in range(len(training_data.iloc[0]))]
    for T_epoch in range(T):
        training_data = training_data.sample(frac=1).reset_index(drop=True)
        for instance in range(len(training_data[keys[0]])):
            pred = predict(training_data.iloc[instance], w)
            if pred != training_data.iloc[instance][-1]:
                w[0] = w[0] + learning_rate * (training_data.iloc[instance][-1] - pred)
                for idx in range(len(training_data.iloc[instance])-1):
                    w[idx+1] = w[idx+1] + learning_rate * (training_data.iloc[instance][-1] - pred) * training_data.iloc[instance][idx]
    return w

# create a function to create the voted weights
def voted_weights(training_data, learning_rate, T):
    keys = training_data.keys()
    w = [0.0 for idx in range(len(training_data.iloc[0]))]
    c_all = []
    c_all.append(1.0)
    w_all = []
    w_all.append(w)
    for T_epoch in range(T):
        training_data = training_data.sample(frac=1).reset_index(drop=True)
        for instance in range(len(training_data[keys[0]])):
            pred = predict(training_data.iloc[instance], w)
            if pred != training_data.iloc[instance][-1]:
                w[0] = w[0] + learning_rate * (training_data.iloc[instance][-1] - pred)
                for idx in range(len(training_data.iloc[instance])-1):
                    w[idx+1] = w[idx+1] + learning_rate * (training_data.iloc[instance][-1] - pred) * training_data.iloc[instance][idx]
                w_all.append(w[:])
                c_all.append(1.0)
            else:
                c_all[-1] += 1    
    return w_all,c_all

# create a function to create the weights for the average perceptron
def averaged_weights(training_data, learning_rate, T):
    keys = training_data.keys()
    w = [0.0 for idx in range(len(training_data.iloc[0]))]
    a = np.zeros(len(w))
    for T_epoch in range(T):
        training_data = training_data.sample(frac=1).reset_index(drop=True)
        for instance in range(len(training_data[keys[0]])):
            pred = predict(training_data.iloc[instance], a)
            w[0] = w[0] + learning_rate * (training_data.iloc[instance][-1] - pred)
            for idx in range(len(training_data.iloc[instance])-1):
                w[idx+1] = w[idx+1] + learning_rate * (training_data.iloc[instance][-1] - pred) * training_data.iloc[instance][idx] 
            a += w
        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (T, learning_rate, error_sum))
    return a

# create a fucntion to predict using the model
def predict(instance, w):
    start = w[0]
    for idx in range(len(instance)-1):
        start += w[idx + 1] * instance[idx]
    if start >= 0.0:
        return 1.0
    else: 
        return 0.0

# create a fucntion to predict using the model with votes
def voted_predict(instance, w, c):
    track = []
    for i in range(len(c)):
        if len(c) == 1:
            start = w[0]
        else:
            start = w[i][0]
        for idx in range(len(instance)-1):
            start += w[i][idx + 1] * instance[idx]
        end = np.sign(start) * c[i]
        track.append(end)
    if np.sum(track) >= 0.0:
        return 1.0
    else: 
        return 0.0

# function to evaluate usng the voted weights
def voted_evaluate(training_data, test_data, learning_rate, T):
    preds = list()
    w,c = voted_weights(training_data, learning_rate, T)
    keys = test_data.keys()
    for instance in range(len(test_data[keys[0]])):
        pred = voted_predict(test_data.iloc[instance],w,c[1:])
        preds.append(pred)
    return preds, w, c

# fucntion to evaluate the dataset using the averaged weights
def averaged_evaluate(training_data, test_data, learning_rate, T):
    preds = list()
    w = averaged_weights(training_data, learning_rate, T)
    keys = test_data.keys()
    for instance in range(len(test_data[keys[0]])):
        pred = predict(test_data.iloc[instance],w)
        preds.append(pred)
    return preds, w

# create a function to evaluate the dataset
def evaluate(training_data, test_data, learning_rate, T):
    preds = list()
    w = weights(training_data, learning_rate, T)
    keys = test_data.keys()
    for instance in range(len(test_data[keys[0]])):
        pred = predict(test_data.iloc[instance],w)
        preds.append(pred)
    return preds, w

# create a function to find accuracy
def accuracy(test_data, predictions):
    keys = test_data.keys()
    actual = test_data[keys[-1]]
    correct = 0.0
    for idx in range(len(actual)):
        if actual[idx] == predictions[idx]:
            correct += 1.0
    acc = correct / len(predictions)
    counts = len(predictions) - correct
    return acc, counts