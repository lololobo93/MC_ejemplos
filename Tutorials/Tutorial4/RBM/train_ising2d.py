########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla, Lauren Hayward Sierens and Giacomo Torlai
### Tutorial 4: Training a Restricted Boltzmann Machine (RBM)
#####################################################################################

from __future__ import print_function
import torch 
import itertools as it
from rbm import RBM
import numpy as np
import time
import math
import os

# Input parameters:
L  = 4     #linear size of the system
T_list = [1.0,1.254,1.508,1.762,2.016,2.269,2.524,2.778,3.032,3.286,3.540] #a temperature for which there are MC configurations stored in data_ising2d/MC_results_solutions 
num_visible         = L*L      #number of visible nodes
num_hidden          = 4        #number of hidden nodes
nsteps              = 100000   #number of training steps
learning_rate = 1e-3     #the learning rate will start at this value and decay exponentially
bsize               = 100      #batch size
num_gibbs           = 10       #number of Gibbs iterations (steps of contrastive divergence)
num_samples         = 10       #number of chains in PCD

### Function to save weights and biases to a parameter file ###
def save_parameters(rbm):
    
    parameter_dir = 'data_ising2d/RBM_parameters'
    if not(os.path.isdir(parameter_dir)):
      os.mkdir(parameter_dir)
    parameter_file_path =  '%s/parameters_nH%d_L%d' %(parameter_dir,num_hidden,L)
    parameter_file_path += '_T' + str(T) +  '.pt'
    torch.save(rbm.state_dict(), parameter_file_path)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

for T in T_list:
	
	# Load the MC configuration training data:
	trainFileName = 'data_ising2d/MC_results_solutions/ising2d_L'+str(L)+'_T'+str(T)+'_train.txt'
	xtrain        = np.loadtxt(trainFileName)
	ept           = np.random.permutation(xtrain) # random permutation of training data
	iterations_per_epoch = xtrain.shape[0] / bsize  

	# Initialize the RBM class
	rbm = RBM(num_visible=num_visible, num_hidden=num_hidden).to(device)
	
	optimizer = torch.optim.Adam(rbm.parameters(),lr=learning_rate, eps=1e-2)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1.0)

	bcount      = 0  #counter
	epochs_done = 1  #epochs counter
	loss_list = [] #loss
	for ii in range(nsteps):
		if bcount*bsize+ bsize>=xtrain.shape[0]:
			# print('epoch %d, Mean "Loss" = %.4f'%(epochs_done, np.mean(loss_list)))
			loss_list = [] #loss
			bcount = 0
			ept = np.random.permutation(xtrain)
			save_parameters(rbm)
			epochs_done += 1

		batch = ept[ bcount*bsize: bcount*bsize+ bsize,:]
		data = torch.Tensor(batch).cuda()
		bcount += 1

		v1 = rbm.contrastive_divergence(data, k=num_gibbs)
		loss = rbm.free_energy(data) - rbm.free_energy(v1)
		loss_list.append(loss.item())

		optimizer.zero_grad()
		loss.backward()
		scheduler.step()
		optimizer.step()

	print('Temperature: %.4f finished'%(T))