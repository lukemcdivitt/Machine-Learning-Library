# this was written by u1200682 for CS 6350

# imports
import string
import pandas as pd
import numpy as np

# value
eps = np.finfo(float).eps

# create function to calculate purity of overarching
def find_purity(dataframe, gain='IG'):

    label = dataframe.keys()[-1]
    entropy = 0
    majorityerror = 0
    majorityerror_buffer = []
    giniindex = 1

    values = dataframe[label].unique()
    for value in values:
        probability = dataframe[label].value_counts()[value] / len(dataframe[label])
        if gain == 'IG':
            entropy += -probability * np.log2(probability)
        elif gain == 'ME':
            majorityerror_buffer.append(probability)
        elif gain == 'GI':
            giniindex += -probability**2
        else:
            print("Incorrect gain method entered. Using Information gain.")
            gain = 'IG'
            entropy += -probability * np.log2(probability)

    if gain == 'IG':
        return entropy
    elif gain == 'ME':
        if len(majorityerror_buffer) > 1:
            me_max_index = np.argmax(majorityerror_buffer)
        #if len(me_max_index) > 1:
            #me_max_index = min(me_max_index)
        majorityerror_buffer = np.delete(majorityerror_buffer, me_max_index)
        majorityerror = sum(majorityerror_buffer)
        return majorityerror
    elif gain == 'GI':
        return giniindex

def find_purity_attribute(dataframe, attribute, gain='IG'):

    label = dataframe.keys()[-1]
    target_values = dataframe[label].unique()
    values = dataframe[attribute].unique()
    
    entropy_high = 0
    majorityerror_high = 0
    giniindex_high = 0

    for value in values:
        entropy = 0
        majorityerror_buffer = []
        giniindex = 1
        for target_value in target_values:
            num = len(dataframe[attribute][dataframe[attribute]==value][dataframe[label]==target_value])
            den = len(dataframe[attribute][dataframe[attribute]==value])
            probability = num / (den+eps)
            if gain == 'IG':
                entropy += -probability * np.log2(probability+eps)
            elif gain == 'ME':
                majorityerror_buffer.append(probability)
            elif gain == 'GI':
                giniindex += -probability**2
            else:
                print("Incorrect gain method entered. Using Information gain.")
                gain = 'IG'
                entropy += -probability * np.log2(probability+eps)

        probability_high = den/len(dataframe)
        if gain == 'IG':
            entropy_high += -probability_high*entropy
        elif gain == 'ME':
            if len(majorityerror_buffer) > 1:
                me_max_index = np.argmax(majorityerror_buffer)
            #if len(me_max_index) > 1:
                #me_max_index = min(me_max_index)
            majorityerror_buffer = np.delete(majorityerror_buffer, me_max_index)
            majorityerror_high += -probability_high*(sum(majorityerror_buffer))
        elif gain == 'GI':
            giniindex_high += -probability_high*giniindex


    if gain == 'IG':
        return abs(entropy_high)
    elif gain == 'ME':
        return abs(majorityerror_high)
    elif gain == 'GI':
        return abs(giniindex_high)

# find the maximum gain
def determine_max_gain(dataframe, gain='IG'):
    purity_att = []
    info_gain = []
    for key in dataframe.keys()[:-1]:
        info_gain.append(find_purity(dataframe,gain) - find_purity_attribute(dataframe, key, gain))
    return dataframe.keys()[:-1][np.argmax(info_gain)]

# create a smaller tabel for after branching
def create_branch_table(dataframe, node, value):
    return dataframe[dataframe[node] == value].reset_index(drop=True)

# create function to build the tree
def learn_decision_tree(dataframe, decision_tree=None, gain='IG', max_depth=6, level=0):

    level = level+1
    label = dataframe.keys()

    node = determine_max_gain(dataframe, gain)
    attribute_val = np.unique(dataframe[node])
    
    if decision_tree is None:

        decision_tree = {}
        decision_tree[node] = {}

    for value in attribute_val:

        branch_table = create_branch_table(dataframe, node, value)
        clValue, counts = np.unique(branch_table[label[-1]], return_counts=True)

        if level == max_depth:
            decision_tree[node][value] = clValue[np.argmax(counts)]
        else:
            if len(counts) == 1:
                decision_tree[node][value] = clValue[0]
            else:
                decision_tree[node][value] = learn_decision_tree(branch_table, max_depth=max_depth, level=level)

    return decision_tree

# create a set of functions to test the tree
# this fucntion is passed an instance and goest through the tree testing the prediction
def predict(instance, tree, default=None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return predict(instance,result)
        else:
            return result
    else:
        return default
    
# this function finds the accuracy of the predictions
def evaluate(tree, test_data, label):
    c_predictions = 0
    w_preditctions = 0
    for idx, row in test_data.iterrows():
        prediction = predict(test_data.iloc[idx], tree)
        if prediction == test_data[label].iloc[idx]:
            c_predictions += 1
        else:
            w_preditctions += 1
    accuracy = c_predictions / (c_predictions + w_preditctions)
    return accuracy

# convert numerical values to a binary over or under the median
def convert_to_binary(data):
    bar = np.median(data)
    for idx in range(0, len(data)):
        if data[idx] > bar:
            data[idx] = 'Over'
        else:
            data[idx] = 'Under'
    return data

# print accuarcy 
def print_accuracy(test_accuracy, train_accuracy):
    print(test_accuracy)
    print(train_accuracy)
    print(np.mean(train_accuracy[0:15]))
    print(np.mean(train_accuracy[16:31]))
    print(np.mean(train_accuracy[32:47]))
    print(np.mean(test_accuracy[0:15]))
    print(np.mean(test_accuracy[16:31]))
    print(np.mean(test_accuracy[32:47]))

# convert integers to binary
def int_to_bin(training_data, test_data, label, replace=1):
    for key in range(0,len(label)):
        if isinstance(training_data[label[key]][0], str):
            pass
        else:
            training_data[label[key]] = convert_to_binary(training_data[label[key]])
            test_data[label[key]] = convert_to_binary(test_data[label[key]])

        if replace == 1:
            test_uni, test_pos = np.unique(test_data[label[key]],return_inverse=True)
            train_uni, train_pos = np.unique(training_data[label[key]],return_inverse=True)

            test_count = np.bincount(test_pos)
            train_count = np.bincount(train_pos)

            test_maxpos = test_count.argmax()
            train_maxpos = train_count.argmax()

            if train_uni[train_maxpos] == 'unknown':
                train_uni = np.delete(train_uni, train_maxpos)
                train_count = np.delete(train_count, train_maxpos)
                train_maxpos = train_count.argmax()

            if test_uni[test_maxpos] == 'unknown':
                test_uni = np.delete(test_uni, test_maxpos)
                test_count = np.delete(test_count, test_maxpos)
                test_maxpos = test_count.argmax()

            for idx in range(0,len(test_data[label[key]])):
                if training_data[label[key]][idx] == 'unknown':
                    training_data[label[key]][idx] = train_uni[train_maxpos]
        
        
                if test_data[label[key]][idx] == 'unknown':
                    test_data[label[key]][idx] = train_uni[train_maxpos]
    
    print(np.unique(test_data[label[key]], return_inverse=True))
    print(np.unique(training_data[label[key]], return_inverse=True))
    return training_data, test_data

# count frequncy of elements
def freq_count(training_data, test_data):
    label = training_data.keys()
    counts = {}
    for element in test_data[label[15]]:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    return label

# convert pd dataframe values to int
def convert_to_int(training_data, test_data, numeric_keys):
    for key in range(0,len(numeric_keys)):
        pd.to_numeric(training_data[numeric_keys[key]])
        pd.to_numeric(test_data[numeric_keys[key]])
    return training_data, test_data

# convert numerical features to binary
def num_to_bin(training_data, test_data, numeric_keys, replace=1):
    training_data, test_data = convert_to_int(training_data, test_data, numeric_keys)
    label = freq_count(training_data, test_data)
    training_data, test_data = int_to_bin(training_data, test_data, label, replace)
    return training_data, test_data