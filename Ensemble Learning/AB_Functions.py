# this was written by u1200682 for CS 6350

# imports
import numpy as np

# create a class to hold the stump
class Stump:

    def __init__(self):
        self.label = 1
        self.thresh = None
        self.a = None
        self.index = None

    def predict(self, value):
        samples = value.shape[0]
        value_col = value[:, self.index]
        pred = np.ones(samples)
        if self.label == 1:
            pred[value_col < self.thresh] = -1
        else:
            pred[value_col > self.thresh] = -1

        return pred

# class to implement the adaboost technique
class Adaboost:

    def __init__(self, T=100):
        self.T = T
        self.classifiers = []

    def build(self, value, labels):
        samples, features = value.shape

        w = np.full(samples, (1/samples))

        self.classifiers = []

        for blank in range(self.T):
            classifier = Stump()