## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import csv
import LogisticRegression as LR

# read in the data
file = open('train.csv','r')
training_data = list(csv.reader(file))
file.close()
file = open('test.csv','r')
test_data = list(csv.reader(file))
file.close()

# convert all of the values to floats
for idx in range(0,len(training_data)):
    for idx_2 in range(0,len(training_data[idx])):
        training_data[idx][idx_2] = float(training_data[idx][idx_2])
for idx in range(0,len(test_data)):
    for idx_2 in range(0,len(test_data[idx])):
        test_data[idx][idx_2] = float(test_data[idx][idx_2])

# set the vlaues
learn_rate = 0.001
epochs = 1000
v = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

# test with the different varainces
MAP_test = []
MAP_train = []
MLE_test =[]
MLE_train = []
for var in v:
    w,w2 = LR.fit(training_data, learn_rate, epochs, var)
    error = LR.accuracy(w, test_data)
    error1 = LR.accuracy(w, training_data)
    error2 = LR.accuracy(w2, test_data)
    error3 = LR.accuracy(w2, training_data)
    MAP_test.append(error)
    MAP_train.append(error1)
    MLE_test.append(error2)
    MLE_train.append(error3)

print('Done')
