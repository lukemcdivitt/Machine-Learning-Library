## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import PyTorch_Functions as PTF
import torch as T

# set device
device = T.device('cpu')

# set initial learn rate
gamma0 = 0.001

# set number of epochs
epochs = 10

# set the training file
train_file = 'train.csv'

# train the network
net = PTF.TrainNetwork(train_file,gamma0,epochs,device)

# set the test file
test_file = 'test.csv'

# test the network
accuracy = PTF.TestNetwork(net, test_file, device)
accuracy2 = PTF.TestNetwork(net, train_file, device)
print(accuracy)
print(accuracy2)