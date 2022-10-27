# this was written by u1200682 for CS 6350

# imports
import numpy as np
import DT_Functions as dt
import random

# create a class to hold the stump
class Stump:

    def __init__(self):
        self.label = 1
        self.thresh = None
        self.a = None
        self.index = None

    def predict(self, dataframe):
        keys = dataframe.keys()
        samples = len(dataframe[keys[1]])
        value_col = dataframe[self.index][:]
        pred = ['yes'] * samples
        if self.label == 1:
            for t_idx in range(0,len(value_col)):
                    if (value_col[t_idx] != self.thresh):
                        pred[t_idx] = 'no'
        else:
            for t_idx in range(0,len(value_col)):
                    if (value_col[t_idx] != self.thresh):
                        pred[t_idx] = 'no'

        return pred

    def predict2(self, dataframe):
        keys = dataframe.keys()
        samples = len(dataframe[keys[1]])
        value_col = dataframe[self.index][:]
        pred = np.ones(samples)
        if self.label == 1:
            for t_idx in range(0,len(value_col)):
                    if (value_col[t_idx] != self.thresh):
                        pred[t_idx] = -1
        else:
            for t_idx in range(0,len(value_col)):
                    if (value_col[t_idx] != self.thresh):
                        pred[t_idx] = -1

        return pred

# class to implement the adaboost technique
class Adaboost:

    def __init__(self, T=100):
        self.T = T
        self.classifiers = []

    def build(self, dataframe, gain):
        keys = dataframe.keys()
        samples = len(dataframe[keys[1]])
        sample_list = np.arange(0,samples)
        labels = dataframe[keys[-1]][:]

        w = np.full(samples, (1/samples))

        self.classifiers = []

        for blank in range(self.T):
            classifier = Stump()
            err_min = float('inf')

            feat = dt.determine_max_gain(dataframe, w, gain)

            value_col = dataframe[feat]
            threshs = np.unique(value_col)

            for thresh in threshs:
                l = 1
                pred = ['yes'] * samples
                for t_idx in range(0,len(value_col)):
                    if (value_col[t_idx] != thresh):
                        pred[t_idx] = 'no'

                misclass = w[labels != pred]
                err = sum(misclass)

                if err > 0.5:
                    err = 1 - err
                    l = -1

                if err < err_min:
                    classifier.label = l
                    classifier.thresh = thresh
                    classifier.index = feat
                    err_min = err

            e = 1e-10
            classifier.a = 0.5 * np.log((1.0 - err_min + e) / (err_min + e))

            pred = classifier.predict(dataframe)
            l_holder = np.zeros(samples)

            for l_idx in range(0,len(labels)):
                if labels[l_idx] == 'yes':
                    l_holder[l_idx] = 1
                else:
                    l_holder[l_idx] = -1

            for l_idx in range(0,len(pred)):
                if pred[l_idx] == 'yes':
                    pred[l_idx] = 1
                else:
                    pred[l_idx] = -1

            w *= np.exp(-classifier.a * l_holder * pred)
            w /= np.sum(w)

            # rand = random.choices(sample_list, w, k=samples)
            # buffer = dataframe
            # for key in keys:
            #     for r_idx in range(0,len(rand)):
            #         dataframe[key][r_idx] = buffer[key][rand[r_idx]]

            self.classifiers.append(classifier)

    def predict(self, dataframe):
        classifier_predictions = [classifier.a * classifier.predict2(dataframe) for classifier in self.classifiers]
        label_prediction = np.sum(classifier_predictions, axis=0)
        label_prediction = np.sign(label_prediction)

        return label_prediction

def accuracy(test_data, predictions):
    holder = ['yes'] * len(predictions)
    for idx in range(0,len(predictions)):
        if predictions[idx] == 1:
            holder[idx] = 'yes'
        else:
            holder[idx] = 'no'
    num = 0.0
    for idx in range(0,len(predictions)):
        if (test_data['y'][idx] == holder[idx]):
            num = num + 1
    accuracy = num / len(predictions)
    return accuracy