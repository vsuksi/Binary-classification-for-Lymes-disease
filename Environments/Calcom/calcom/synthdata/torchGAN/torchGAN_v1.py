# Implementing GAN using Torch.
#
# Generator is a neural net with
# *parameter* number of input nodes;
# output nodes are same dimension as
# in original space.
#
# Discriminator is a neural net with
# input nodes dimension of the original
# space, and a *single* output node
# indicating some kind of probability
# that the input is real or fake.
#
# For the moment let's look only at the
# discriminator. Generate data from two
# normal distributions in ten dimensions,
# and ask the discriminator to learn real
# from fake data using a simple cross-validation.
#
# I'm looking at the following example:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim
#

import torch
import numpy as np

nreal = 100
nfake = 100
d = 10
hl = 5  # Size of hidden layer in discriminator.

n = nreal + nfake

x = torch.randn(n,d)
x[nreal:] = torch.randn(nfake,d)*0.5 + 1  # Fake data.
y = torch.ones(n,1)
y[nreal:] = torch.zeros(nfake,1)

discr = torch.nn.Sequential(
    torch.nn.Linear(d, hl),
    torch.nn.Tanh(),
    torch.nn.Linear(hl, 1),
)

loss_fn = torch.nn.MSELoss(size_average=False)
dt = 1e-2
optimizer = torch.optim.Adam(discr.parameters(), lr=dt)

perm = np.random.permutation(n)
tr = perm[:n//2]
te = perm[n//2:]

x_tr = x[tr]
y_tr = y[tr]
for i in range(500):
    y_pred = discr(x_tr)
    sse = loss_fn(y_pred, y_tr)
    print(i,sse.item())
    optimizer.zero_grad()   # Apparently always necessary.
    sse.backward()
    optimizer.step()
#

# TODO: insert evaluation of network. confusion matrix,
# scatter plot.
