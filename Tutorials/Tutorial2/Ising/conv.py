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

#"linear size of the system "
lx=30

#parameters of the neural network and cost function
numberlabels=2 # Number of phases under consideration (2 for the Ising model on the square lattice) 
hiddenunits1=100 # number of hidden unites in the hidden layer
lamb=0.001 # regularization parameter 
beta=1.0 #``inverse temperature'' of the sigmoid neuron

#Parameters of the optimization
#batch size for the gradient descent 
bsize=1500
# number of iterations
niter=1000

# temperature list at which the training/test sets were generated
tlist=[1.0000000000000000,1.0634592657106510,1.1269185314213019,1.1903777971319529,1.2538370628426039,1.3172963285532548,1.3807555942639058,1.4442148599745568,1.5076741256852078,1.5711333913958587,1.6345926571065097,1.6980519228171607,1.7615111885278116,1.8249704542384626,1.8884297199491136,1.9518889856597645,2.0153482513704155,2.0788075170810667,2.1422667827917179,2.2057260485023691,2.3326445799236715,2.3961038456343227,2.4595631113449739,2.5230223770556250,2.5864816427662762,2.6499409084769274,2.7134001741875786,2.7768594398982298,2.8403187056088810,2.9037779713195322,2.9672372370301834,3.0306965027408346,3.0941557684514858,3.1576150341621370,3.2210742998727881,3.2845335655834393,3.3479928312940905,3.4114520970047417,3.4749113627153929,3.5383706284260401]

# Description of the input data 
Ntemp=40 # number of different temperatures used in the training and testing data
samples_per_T=250 # number of samples per temperature value in the testing set
Nord=20 # number of temperatures in the ordered phase

# Parameters
params = {'batch_size': bsize,
          'shuffle': True,
          'num_workers': 1, 
          'pin_memory': True}

#reading the data in the directory txt 
mnist = input_data.read_data_sets(numberlabels, lx, 'txt')

print("reading sets ok")

class red(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(lx*lx, hiddenunits1, 
                                dtype=torch.float64) / math.sqrt(lx*lx))
        self.b1 = nn.Parameter(torch.zeros(hiddenunits1, dtype=torch.float64))
        self.W2 = nn.Parameter(torch.randn(hiddenunits1, numberlabels, 
                                dtype=torch.float64) / math.sqrt(hiddenunits1))
        self.b2 = nn.Parameter(torch.zeros(numberlabels, dtype=torch.float64))
    
    # forward pass
    def forward(self, xb):
        z1 = beta*(xb @ self.W1) + self.b1
        a1 = torch.sigmoid(z1)
        z2 = beta*(a1 @ self.W2) + self.b2
        return torch.sigmoid(z2)

    def hlayer(self, xb):
        z1 = (xb @ self.W1) + self.b1
        return z1

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = red().to(device)
optimizer =  optim.Adam(model.parameters(), 
                lr=0.0005, weight_decay=0.5*lamb)
loss_func = F.cross_entropy
train_dl = DataLoader(mnist.train, **params)
# train_dl = DataLoader(mnist.test, **params)

x_train, y_train = mnist.train[:]
x_test, y_test = mnist.test[:]
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

# training
for i in range(niter):
  for batch_idx, (data, labels) in enumerate(train_dl):
    data, labels = data.to(device), labels.to(device)
    pred = model(data)
    loss = loss_func(pred, labels)
    # l2 = 0
    # for p in model.parameters():
    #   l2 += p.norm(2)
    # loss = loss + 0.5*lamb*l2 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  if i%100 == 0:
    preds = model(x_train)
    print("step {}, training accuracy {}".format(i, accuracy(preds, y_train)))
    preds = model(x_test)
    print("step {}, test accuracy {}".format(i, accuracy(preds, y_test)))
preds = model(x_train)
print("Final training accuracy {}".format(i, accuracy(preds, y_train)))
preds = model(x_test)
print("Final test accuracy {}".format(i, accuracy(preds, y_test)))

# f = open('hlout.dat', 'w')
# for ii in range(Ntemp*samples_per_T):
#   preds = model.hlayer(x_test[ii])
#   mag_x = torch.mean(x_test[ii]).cpu()
#   mag_x = mag_x-(1-mag_x)
#   f.write(str(mag_x.item())+' '+str(preds[0].item())+
#           ' '+str(preds[1].item())+' '+str(preds[2].item())+"\n") #
# f.close()

# #producing plots of the results
# f = open('nnout.dat', 'w')
# # Average output of neural net over the test set
# ii=0
# for i in range(Ntemp):
#   av=0.0
#   for j in range(samples_per_T):
#         res = model(x_test[ii])
#         av=av+res 
#         ii=ii+1 
#   av_np=av.detach().data.cpu().numpy()/samples_per_T
#   f.write(str(i)+' '+str(tlist[i])+' '+str(av_np[0])+' '+str(av_np[1])+"\n")  
# f.close()

# # Average accuracy vs temperature over the test set
# f = open('acc.dat', 'w')
# for ii in range(Ntemp):
#   preds = model(x_test[ii*samples_per_T:ii*samples_per_T+samples_per_T])
#   train_accuracy = accuracy(preds, y_test[ii*samples_per_T:ii*samples_per_T+samples_per_T])
#   f.write(str(ii)+' '+str(tlist[ii])+' '+str(train_accuracy.item())+"\n") #
# f.close()
