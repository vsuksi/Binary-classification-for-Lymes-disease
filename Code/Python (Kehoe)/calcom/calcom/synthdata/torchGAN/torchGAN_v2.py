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
# In v2 we'll try to set up the full thing.
#
# I'm looking at the following example:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim
#

import torch
import numpy as np

nreal = 100
nfake = 100
d = 10
hl = 5      # Size of hidden layer in discriminator.
d_noise = 3 # Dimension of noisy data; input layer for generator.
m = 20  # batch size i guess
k_inner = 5 # 5 iterations for discriminator for each one of generator.


#############################################
# Set up neural nets, optimization function.

generator = torch.nn.Sequential(
    torch.nn.Linear(d_noise,hl),  # input layer somehow has to do with inherent dimensionality.
    torch.nn.Tanh(),
    torch.nn.Linear(hl,d),
)

discriminator = torch.nn.Sequential(
    torch.nn.Linear(d, hl),
    torch.nn.Tanh(),
    torch.nn.Linear(hl, 1),
)

# loss_fn = torch.nn.MSELoss(size_average=False)
dt = 1e-2
# d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=dt)

niter = 500

def noise_prior(nsamp):
    return 0.1*np.random.randn(nsamp,d)
#

def data_distribution(nsamp):
    return 2 + np.random.randn(nsamp,d_noise)
#

#########################################
# Generate data.
n = nreal + nfake

x = torch.randn(n,d)
x[nreal:] = torch.randn(nfake,d)*0.5 + 1  # Fake data.
y = torch.ones(n,1)
y[nreal:] = torch.zeros(nfake,1)

perm = np.random.permutation(n)
tr = perm[:n//2]
te = perm[n//2:]

x_tr = x[tr]
y_tr = y[tr]

for i in range(niter):

    # k_inner iterations training the discriminator
    # on the current state of the generator.
    for k in range(k_inner):
        # Get m samples of noise prior for generator
        z_samp = noise_prior(m)
        x_fake = generator(z_samp)
        # Get m samples of **data generating distribution**
        x_real = data_distribution(m)

        # Update discriminator based on *minimization* of
        # -sum(log(D(x)) + log(1-D(G(z)))), that is
        # -sum(log(D(x_real)) + log(1-D(x_fake)))

    #

    # one iteration training the generator
    # based on the curent state of the discriminator.

    # Get m samples of noise prior for generator
    z_samp_2 = noise_prior(m)
    # Update the generator with gradient *descent*
    # on log(1-D(G(z))) or *ascent* on log(D(G(z))) -
    # note in the paper they suggest the second for early in training.

    #
#
