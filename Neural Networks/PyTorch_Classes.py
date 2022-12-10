## this was written by u1200682 for CS 6350
# Luke McDivitt

# imports
import torch as T
import numpy as np

# create class for the training dataset
class Dataset(T.utils.data.Dataset):

    def __init__(self, src_file, device, num_rows=None):

        all_data = np.loadtxt(src_file, max_rows=num_rows, delimiter=',', skiprows=0, dtype=np.float32)
        self.x_data = T.tensor(all_data[:,0:4], dtype=T.float32).to(device)
        self.y_data = T.tensor(all_data[:,4], dtype=T.float32).reshape(-1,1).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx,:]  # idx rows, all 4 cols
        lbl = self.y_data[idx,:]    # idx rows, the 1 col
        sample = { 'predictors' : preds, 'target' : lbl }
        return sample



# set the network architecture
class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    hidden = 50
    self.hid1 = T.nn.Linear(4, hidden)  # 4-(5-5)-1
    self.hid2 = T.nn.Linear(hidden, hidden)
    self.hid3 = T.nn.Linear(hidden, hidden)
    self.hid4 = T.nn.Linear(hidden, hidden)
    # self.hid5 = T.nn.Linear(hidden, hidden)
    # self.hid6 = T.nn.Linear(hidden, hidden)
    # self.hid7 = T.nn.Linear(hidden, hidden)
    # self.hid8 = T.nn.Linear(hidden, hidden)
    self.oupt = T.nn.Linear(hidden, 1)

    T.nn.init.kaiming_uniform_(self.hid1.weight) 
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.kaiming_uniform_(self.hid2.weight) 
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.kaiming_uniform_(self.hid3.weight)  
    T.nn.init.zeros_(self.oupt.bias)
    T.nn.init.kaiming_uniform_(self.hid4.weight)  
    T.nn.init.zeros_(self.oupt.bias)
    # T.nn.init.xavier_uniform_(self.hid5.weight)  
    # T.nn.init.zeros_(self.oupt.bias)
    # T.nn.init.xavier_uniform_(self.hid6.weight)  
    # T.nn.init.zeros_(self.oupt.bias)
    # T.nn.init.xavier_uniform_(self.hid7.weight)  
    # T.nn.init.zeros_(self.oupt.bias)
    # T.nn.init.xavier_uniform_(self.hid8.weight)  
    # T.nn.init.zeros_(self.oupt.bias)
    T.nn.init.kaiming_uniform_(self.oupt.weight)  
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.relu(self.hid1(x)) 
    z = T.relu(self.hid2(z))
    z = T.relu(self.hid3(z))
    z = T.relu(self.hid4(z))
    # z = T.relu(self.hid5(z))
    # z = T.relu(self.hid6(z))
    # z = T.relu(self.hid7(z))
    # z = T.relu(self.hid8(z))
    z = T.sigmoid(self.oupt(z)) ##################
    return z


    
    