# this was written by u1200682 for CS 6350

# includes
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds, BFGS

# convert the data to 1 and -1
def convert_classifcation(dataframe):
    keys = dataframe.keys()
    for idx in range(len(dataframe[keys[-1]])):
        if dataframe[keys[-1]][idx] == 0:
            dataframe[keys[-1]][idx] = -1
    return dataframe

# create schedules
def s(gamma0, a, epoch, schedule):
    if schedule == 1:
        rate = (gamma0 / (1 + (gamma0 / a)*epoch))
        return rate
    elif schedule == 2:
        rate = gamma0 / (1+epoch)
        return rate

# implement SVM in the primal domain with stoch sub-grad descent
def subgradient_descent(training_data, gamma0, a, epochs, C, schedule):

    keys = training_data.keys()
    w0 = np.zeros((len(keys)-1))
    w = np.zeros((len(keys)))
    x = np.ones((len(keys)))

    for epoch in range(0,epochs):

        buffer = training_data.sample(frac=1).reset_index(drop=True).to_numpy()
        learn_rate = s(gamma0, a, epoch, schedule)

        for row in range(0,len(buffer)):
            
            x[:-1] = buffer[row,:-1]

            if (buffer[row,-1] * np.dot(w,x)) <= 1:

                w0 = w[:-1]
                d = np.append(w0,0)
                w = w - learn_rate*d + learn_rate * C * buffer[row,-1] * x

            else:

                w[:-1] = (1 - learn_rate) * w[:-1]

    return w

# create a function to test the solution
def evaluate(test_data,w):

    buffer = test_data.sample(frac=1).reset_index(drop=True).to_numpy()
    wrong = 0

    for idx in range(0,len(buffer)):

        pred = np.sign(np.dot(w[:-1],buffer[idx,:-1]) + w[0])

        if pred != buffer[idx,-1]:
            wrong += 1
        
    return wrong / len(buffer)

# dual equation
def dual_equation(alpha, x, y):
    z = y.reshape((x.shape[0],1))
    answer = 0.5 * alpha.T @ np.multiply(z@z.T,x@x.T) @ alpha - sum(alpha)
    print(answer)
    return answer

# find that alpha values
def find_alphas(x, y, C):
    # get data info
    samples,params = x.shape

    # initialize alphas
    alpha_ic = np.zeros((len(y),1)) * 1.0
    # alpha_ic.reshape((x.shape[0],1))

    # set the constraints
    linear_constraint = LinearConstraint(y, [0], [0])
    bounds_alpha = Bounds(np.zeros(samples), np.full(samples, C))

    # run the minimization
    solution = minimize(dual_equation, alpha_ic, args=(x,y), method='SLSQP', constraints=[linear_constraint], bounds=bounds_alpha, tol=1e-6)

    # collect the alphas
    alphas = solution.x

    return alphas

# calculate the w
def calc_w(alpha, x, y):
    samples = len(x)
    w = np.zeros(x.shape[1])
    for idx in range(samples):
        w = w + alpha[idx]*y[idx]*x[idx,:]
    return w

def get_b(alpha, x, y, w, C):
    zero = 1e-6
    C_num = C - zero
    sv_index = np.where((alpha > zero)&(alpha<C_num))[0]
    b = 0.0
    for idx in sv_index:
        b = b + y[idx] - np.dot(x[idx,:],w)

    b = b / len(sv_index)
    return b

def classify(test_data, w, b):
    preds = np.sum(test_data*w, axis=1) + b
    preds = np.sign(preds)
    preds[preds==0] = 1
    return preds

def error(test_data, preds):
    error = sum(test_data != preds) / len(test_data)
    return error

def get_weights(x,y,C):

    alphas = find_alphas(x,y,C)
    w = calc_w(alphas,x,y)
    b = get_b(alphas,x,y,w,C)

    return w,b

###################################################################
# # altered for the kernal
# def g_kernel(xi,xj,gamma):
#     norm = np.linalg.norm(xi - xj) ** 2
#     return np.exp(-(norm / gamma))

def g_kernel(x, z, gamma):
    samples = x.shape[0]
    num = z.shape[0]
    xx = np.dot(np.sum(np.power(x, 2), 1).reshape(samples, 1), np.ones((1, num)))
    zz = np.dot(np.sum(np.power(z, 2), 1).reshape(num, 1), np.ones((1, samples)))   
    return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / (gamma))

# dual equation
def dual_equation(alpha, H):
    answer = 0.5 * alpha.T @ H @ alpha - sum(alpha)
    print(answer)
    return answer

# find that alpha values
def find_alphas_kernel(x, y, C, gamma):
    # get data info
    samples,params = x.shape

    # initialize alphas
    alpha_ic = np.zeros((len(y),1)) * 1.0
    # alpha_ic.reshape((x.shape[0],1))

    # get k
    K = g_kernel(x,x,gamma)
    # K = np.zeros((samples,samples))
    # for idx in range(samples):
    #     for idx2 in range(samples):
    #         K[idx,idx2] = g_kernel(x[idx],x[idx2],gamma)

    # get H
    H = np.zeros((samples,samples))
    for i in range(samples):
        for j in range(samples):
            H[i,j] = y[i] * y[j] * K[i,j]

    # set the constraints
    linear_constraint = LinearConstraint(y, [0], [0])
    bounds_alpha = Bounds(np.zeros(samples), np.full(samples, C))

    # run the minimization
    solution = minimize(dual_equation, alpha_ic, args=(H), method='SLSQP', constraints=[linear_constraint], bounds=bounds_alpha, tol=1e-6)

    # collect the alphas
    alphas = solution.x

    return alphas

# calc b with kernel
def get_b_kernel(alpha, x, y, C,gamma):
    zero = 1e-6
    C_num = C - zero
    sv_index = np.where((alpha > zero)&(alpha<C_num))[0]
    b = y[sv_index] - (alpha * y).dot(g_kernel(x,x[sv_index],gamma))
    b = np.sum(b)/b.size
    return b

# with kenerl
def classify_kernel(test_data, alpha, b, y,x,gamma,C):
    zero = 1e-6
    C_num = C - zero
    sv_index = np.where((alpha > zero)&(alpha<C_num))[0]
    pred =  np.sum((alpha[sv_index] * y[sv_index]).dot(g_kernel(test_data,x[sv_index] ,gamma).T), axis=0) + b
    return np.sign(pred)