import input_data
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

lx=16 # linear size of the lattice
training=400 # size of training set
bsize=200  # batch size
Ntemp=2 # number of temperatures 0, infinity
samples_per_T_test=2500 # samples per each temperature in the test set

numberlabels=2 # number of labels (T=0, infinity)
# Parameters
params = {'batch_size': bsize,
          'shuffle': True,
          'num_workers': 1, 
          'pin_memory': True}

#reading the data in the directory txt 
mnist = input_data.read_data_sets(numberlabels, lx+1, 'txt', one_hot=True)

print("reading sets ok")

nmaps1=64
nmaps2=64
zero_prob = 0.5
class red(nn.Module):
    def __init__(self):
        super().__init__()
        a = torch.ones([nmaps1, 2, 2, 2], dtype = torch.float)/10.0
        self.W_conv1 = nn.Parameter(torch.normal(mean = 0., std = a))
        self.b_conv1 = nn.Parameter(torch.ones(nmaps1, dtype=torch.float)/10.0)
        a = torch.ones([lx*lx*nmaps1, nmaps2], dtype = torch.float)/10.0
        self.W_fc1 = nn.Parameter(torch.normal(mean = 0., std = a))
        self.b_fc1 = nn.Parameter(torch.ones(nmaps2, dtype=torch.float)/10.0)
        a = torch.ones([nmaps2, numberlabels], dtype = torch.float)/10.0
        self.W_fc2 = nn.Parameter(torch.normal(mean = 0.0, std = a))
        self.b_fc2 = nn.Parameter(torch.ones(numberlabels, dtype=torch.float)/10.0)
        self.drop = torch.nn.Dropout(p=zero_prob)

    # forward pass
    def forward(self, xb):
        x_input = xb.view(-1, 2, lx+1, lx+1)
        x_conv = F.relu(F.conv2d(x_input, self.W_conv1, stride=(1,1), bias = self.b_conv1))
        x_conv_flat = x_conv.view(-1, lx*lx*nmaps1)
        x_fc1 = F.relu(x_conv_flat @ self.W_fc1 + self.b_fc1)
        x_drop = self.drop(x_fc1)
        x_fc2 = F.softmax(x_drop @ self.W_fc2 + self.b_fc2, dim=0)
        return x_fc2

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    yb_arg = torch.argmax(yb, dim=1)
    return (preds == yb_arg).float().mean()

#Train and Evaluate the Model
# cost function to minimize

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = red().to(device)
optimizer =  optim.Adam(model.parameters(), 
                lr=1.0e-4)
loss_func = F.binary_cross_entropy
train_dl = DataLoader(mnist.train, **params)
# train_dl = DataLoader(mnist.test, **params)

x_train, y_train = mnist.train[:]
x_test, y_test = mnist.test[:]
x_train, y_train = x_train.float().to(device), y_train.float().to(device)
x_test, y_test = x_test.float().to(device), y_test.float().to(device)

for i in range(training):
  model.train()
  for batch_idx, (data, labels) in enumerate(train_dl):
    data, labels = data.float().to(device), labels.float().to(device)
    pred = model(data)
    loss = loss_func(pred, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  if i%100 == 0:
    model.eval()
    preds = model(x_train)
    print("step {}, training accuracy {}".format(i, accuracy(preds, y_train)))
    preds = model(x_test)
    print("step {}, test accuracy {}".format(i, accuracy(preds, y_test)))

model.eval()
preds = model(x_train)
print("Final training accuracy {}".format(accuracy(preds, y_train)))
preds = model(x_test)
print("Final test accuracy {}".format(accuracy(preds, y_test)))

#producing data to get the plots we like

# f = open('nnout.dat', 'w')

# #output of neural net
# ii=0
# for i in range(Ntemp):
#   av=0.0
#   for j in range(samples_per_T_test):
#         batch=(mnist.test.images[ii,:].reshape((1,2*(lx+1)*(lx+1))),mnist.test.labels[ii,:].reshape((1,numberlabels)))
#         res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
#         av=av+res
#         #print ii, res
#         ii=ii+1
#   av=av/samples_per_T_test
#   f.write(str(i)+' '+str(av[0,0])+' '+str(av[0,1])+"\n") 
# f.close() 

# f = open('acc.dat', 'w')

# # accuracy vs temperature
# for ii in range(Ntemp):
#   batch=(mnist.test.images[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape(samples_per_T_test,2*(lx+1)*(lx+1)), mnist.test.labels[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape((samples_per_T_test,numberlabels)) )
#   train_accuracy = sess.run(accuracy,feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
#   f.write(str(ii)+' '+str(train_accuracy)+"\n")
# f.close()
