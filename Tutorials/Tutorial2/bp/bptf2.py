# A bit of setup
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# figure setup
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# activation function
def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def ReLU(x):
  return np.maximum(x, 0)

np.random.seed(0)

# this is a little data set of spirals with 3 branches
N = 50 # number of points per branch
D = 2 # dimensionality of the vectors to be learned
K = 3 # number of branches
X = np.zeros((N*K,D)) # matrix containing the dataset
y = np.zeros(N*K, dtype='int64') # labels
# data generation
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

X_tensor, y_tensor = map(torch.tensor, (X, y))


# plotting dataset
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
fig.savefig('spiral_raw.png')

h = 100 # size of hidden layer

class red(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(D, h, dtype=torch.float64) / math.sqrt(D))
        self.b1 = nn.Parameter(torch.zeros(h, dtype=torch.float64))
        self.W2 = nn.Parameter(torch.randn(h, K, dtype=torch.float64) / math.sqrt(D))
        self.b2 = nn.Parameter(torch.zeros(K, dtype=torch.float64))
    
    # forward pass
    def forward(self, xb):
        z1 = xb @ self.W1 + self.b1
        a1 = F.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return torch.sigmoid(z2)

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# some hyperparameters
step_size = 1.0 #e-0
lmbda = 0.00005
bs = 10
# gradient descent loop
num_examples = X.shape[0]
r=torch.randperm(num_examples)
X_tensor = X_tensor[r]
y_tensor = y_tensor[r]

model = red()
# optimizer =  optim.SGD(model.parameters(), 
#                 lr=step_size, weight_decay=0.5*lmbda)
loss_func = F.cross_entropy

def fit():
    for i in range(20000):
      for j in range((num_examples-1)//bs + 1):
        start_i = j*bs
        end_i = start_i + bs - 1
        xb = X_tensor[start_i:end_i]
        yb = y_tensor[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        l2 = 0
        for p in model.parameters():
          l2 += p.norm(2)
        loss = loss + 0.5*lmbda*l2 
        loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * step_size
                model.zero_grad()
        
      if i % 1000 == 0:
        preds = model(X_tensor)
        print("iteration %d: accuracy %f" % (i, accuracy(preds, y_tensor)))

fit()

 
preds = model(X_tensor)    
print('Final training accuracy: {}'.format(accuracy(preds, y_tensor)))
    
## plot the resulting classifier
# plot the resulting classifier

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(ReLU( np.dot(np.c_[xx.ravel(), yy.ravel()], 
                        model.W1.detach().numpy()) + model.b1.detach().numpy()), 
                        model.W2.detach().numpy()) + model.b2.detach().numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_net_results.png')
plt.show()
