# this was was written by u1200682 for cs 6350

# imports
import random
import numpy as np
import DT_Functions as dt
import pandas as pd

# create a random subsample with replacement
def subsamp(dataframe, size):
    new = dataframe.sample(frac=size)
    # keys = dataframe.keys()
    # samples = round(len(dataframe[keys[1]])*size)
    # sample_list = np.arange(0,samples)
    # rand = random.choices(sample_list, k=samples)
    # buffer = dataframe
    # if size != 1:
    #     for key in keys:
    #         for r_idx in range(0,len(rand)):
    #             dataframe[key][r_idx] = buffer[key][rand[r_idx]]

    return new

# make predictions with bagging
def bag_prediction(trees, row):
    preds = [dt.predict(row, tree) for tree in trees]
    return max(set(preds), key=preds.count)

# bagging
def bagging(training_data, test_data, max_depth=20, sample_size=1, T=10):
    trees = []
    accuracy_test = []
    accuracy_train = []
    for idx in range(T):
        dataframe = subsamp(training_data, sample_size)
        tree = dt.learn_decision_tree(dataframe, max_depth=max_depth)
        trees.append(tree)
        preds = [bag_prediction(trees, test_data.iloc[row]) for row in range(0,len(test_data['age']))]
        acc = accuracy(test_data, preds)
        accuracy_test.append(1-acc)
        preds = [bag_prediction(trees, training_data.iloc[row]) for row in range(0,len(test_data['age']))]
        acc = accuracy(training_data, preds)
        accuracy_train.append(1-acc)
    return accuracy_test,accuracy_train

def accuracy(test_data, predictions):
    num = 0
    for idx in range(0,len(predictions)):
        if (test_data['y'][idx] == predictions[idx]):
            num = num + 1
    accuracy = num / len(predictions)
    return accuracy

    # bagging
def bagging_random(training_data, test_data, max_depth=20, sample_size=1, T=10, max_features=6):
    trees = []
    accuracy_test = []
    accuracy_train = []
    for idx in range(T):
        dataframe = subsamp(training_data, sample_size)
        tree = dt.learn_random_tree(dataframe.reset_index(drop=True), max_depth=max_depth, max_features=max_features)
        trees.append(tree)
        if idx % 50 == 0:
            preds = [bag_prediction(trees, test_data.iloc[row]) for row in range(0,len(test_data['age']))]
            acc = accuracy(test_data, preds)
            accuracy_test.append(1-acc)
            preds = [bag_prediction(trees, training_data.iloc[row]) for row in range(0,len(test_data['age']))]
            acc = accuracy(training_data, preds)
            accuracy_train.append(1-acc)
            print(accuracy_test)
            print(accuracy_train)
        print(idx)
    return accuracy_test,accuracy_train
    