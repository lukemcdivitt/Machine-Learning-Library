## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import csv

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

    