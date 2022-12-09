## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import NeuralNetwork as NN
import csv

# read in the data
file = open('train.csv','r')
training_data = list(csv.reader(file))
file.close()
file = open('test.csv','r')
test_data = list(csv.reader(file))
file.close()

# initialize the neural network
num_inputs = 3
num_hidden = 2
num_outputs = 1
neural_net = NN.setup_network(num_inputs, num_outputs, num_hidden)