## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import torch as T
import numpy as np
import PyTorch_Classes as PC

# calculate the acccuracy of the model
def accuracy(model, ds):
  # ds is a PyTorch Dataset
  # assumes model = model.eval()
  n_correct = 0; n_wrong = 0

  for i in range(len(ds)):
    inpts = ds[i]['predictors'] 
    target = ds[i]['target']
    with T.no_grad():
      oupt = model(inpts)

    if target < 0.5 and oupt < 0.5:
      n_correct += 1

    elif target >= 0.5 and oupt >= 0.5:
      n_correct += 1

    else:
      n_wrong += 1



  return (n_correct * 1.0) / (n_correct + n_wrong)

# train the network
def TrainNetwork(training_file, gamma0, epochs, device):
    T.manual_seed(1)
    np.random.seed(1)
    train_ds = PC.Dataset(training_file,device)  # all rows

    bat_size = 10
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True)

    net = PC.Net().to(device)
    net = net.train()  # set training mode

    lrn_rate = gamma0
    loss_obj = T.nn.BCELoss()  # binary cross entropy
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)

    for epoch in range(0, epochs):

        epoch_loss = 0.0  # sum of avg loss/item/batch

        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors']  # [10,4]  inputs
            Y = batch['target']      # [10,1]  targets
            optimizer.zero_grad()
            oupt = net(X)            # [10,1]  computeds 

            loss_val = loss_obj(oupt, Y)   # a tensor
            epoch_loss += loss_val.item()  # accumulate
            loss_val.backward()  # compute all gradients
            optimizer.step()     # update all wts, biases

        if epoch % 10 == 0:  
            print("epoch = %4d   error = %0.4f" % (epoch, epoch_loss))

    print("Done Training")

    return net

# test the network
def TestNetwork(net, test_file, device):
    test_ds = PC.Dataset(test_file,device)
    net = net.eval()
    acc = accuracy(net,test_ds)
    print("\nAccuracy = %0.4f" % acc)
    return acc