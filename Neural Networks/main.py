## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import NeuralNetwork as NN
import csv
import matplotlib.pyplot as plt

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

# run the test case
value, neural_net = NN.TestCase()

# train a network
num_inputs = len(training_data[1])-1
num_outputs = 2
num_hidden = [5,10,25,50] # this is where you can adjust the width
gamma0 = 0.01
d = 0.01
init = 1
T = 100

# run through different widths
for idx in range(len(num_hidden)):
    trained_network,error,ep = NN.create_network(training_data, num_inputs, num_outputs, num_hidden[idx], gamma0, d, init=init, T=T)
    # plt.plot(ep,error)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Convergence Check')
    # plt.show()
    accuracy = NN.accuracy(trained_network, test_data)
    accuracy2 = NN.accuracy(trained_network, training_data)
    print(accuracy)
    print(accuracy2)


