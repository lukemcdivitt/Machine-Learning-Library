## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import numpy as np
import random as rd

# define the gradient function (MAP)
def gradient(w, x, y, sigma_square):
    x = np.append(x,1)
    g = (-y*x) / (1+np.exp(y*np.dot(w,x))) + (1/sigma_square)*w
    return g

# this is the MLE Gradient
def gradient2(w, x, y, sigma_square):
    x = np.append(x,1)
    g = (-y*x) / (1+np.exp(y*np.dot(w,x)))
    return g

# define the sigmoid
def sigmoid(s):
    return (1 / (1 + np.exp(-s)))

# set learning schedule
def rate_schedule(gamma0, d, T):
    rate = gamma0 / (1 + (gamma0/d)*T)
    return rate

# perform the algorithm
def fit(training_data, gamma0, epochs, sigma_square):

    train = np.array(training_data)
    w = np.zeros([len(training_data[0])])
    w2 = np.zeros([len(training_data[0])])

    for idx in range(len(train)):
        if train[idx,-1] == 0:
            train[idx,-1] = -1

    for epoch in range(0,epochs):

        learn_rate = rate_schedule(gamma0, d=0.001, T=epoch)

        np.random.shuffle(train)
        x_data = train[:,:-1]
        y_data = train[:,-1]

        for idx in range(len(x_data)):

            pred = predict(w, x_data[idx])
            pred2 = predict(w2,x_data[idx])

            if pred != y_data[idx]:
                w -= learn_rate * (gradient(w, x_data[idx], y_data[idx], sigma_square))

            if pred2 != y_data[idx]:
                w2 -= learn_rate * (gradient2(w2, x_data[idx], y_data[idx], sigma_square))

            if any(np.isnan(w)):
                print('hello')

    return w,w2

# use the weights to predict
def predict(w, case):

    y_pred = w[-1]
    for idx in range(len(case)-1):
        y_pred += w[idx] * case[idx]

    if sigmoid(y_pred) > 0.5:
        return 1
    else:
        return -1


# test the accuracy
def accuracy(w, test_data):

    x = np.array(test_data)[:,:-1]
    y = np.array(test_data)[:,-1]
    predictions = []

    for idx in range(len(y)):
        if y[idx] == 0:
            y[idx] = -1

    for idx in range(len(test_data)):
        predictions.append(predict(w, x[idx]))

    incorrect = 0

    for idx in range(0, len(predictions)):

        if predictions[idx] != y[idx]:
            incorrect += 1
    
    return incorrect/len(y)



