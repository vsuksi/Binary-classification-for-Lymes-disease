# Adapted from
# https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_pytorch.py
#

try:
    import torch
except ImportError:
    torch = None
if torch:
    import torch.nn.functional as nn
    import torch.autograd as autograd
    import torch.optim as optim
    from torch.autograd import Variable
#

import numpy as np
# from scipy import stats
# from matplotlib import pyplot


d = 3      # Dimensionality of the data
fig,ax = pyplot.subplots(d,d, sharex=True, sharey=True)
pyplot.ion()
pyplot.show(block=False)

hl = 3      # Size of hidden layer in discriminator.
d_noise = 3 # Dimension of noisy data; input layer for generator.
m = 400  # batch size i guess
k_inner = 5 # iterations for discriminator for each one of generator.
niter = 100000
dt = 1e-2

def noise_prior(nsamp):
    return torch.rand(nsamp,d_noise)
#

def data_distribution(nsamp):
    return 2 + 0.5*torch.randn(nsamp,d)
#


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

Wzh = xavier_init(size=[d_noise, hl])
bzh = Variable(torch.zeros(hl), requires_grad=True)

Whx = xavier_init(size=[hl, d])
# bhx = Variable(torch.zeros(d), requires_grad=True)
bhx = Variable(1*torch.ones(d), requires_grad=True) # push to mean to start.


def G(z):
    h = nn.tanh(z @ Wzh + bzh.repeat(z.size(0), 1))
    # X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    X = nn.tanhshrink(h @ Whx + bhx.repeat(h.size(0), 1))

    return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[d, hl])
bxh = Variable(torch.zeros(hl), requires_grad=True)

Why = xavier_init(size=[hl, 1])
bhy = Variable(torch.zeros(1), requires_grad=True)


def D(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    y = nn.sigmoid(h @ Why + bhy.repeat(h.size(0), 1))
    return y


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
params = G_params + D_params

"""============= Visualization during training ============="""
def update_plot():
    for i in range(d):
        for j in range(i+1,d):
            ax[i,j].cla()
            ax[j,i].cla()
    #

    ns = 400
    real_sample = data_distribution(ns)
    fake_sample = G(noise_prior(ns)).detach().numpy()

    for i in range(d):
        for j in range(i+1,d):
            ax[i,j].scatter(real_sample[:,i], real_sample[:,j], c='k', s=3, alpha=0.5)
            ax[j,i].scatter(real_sample[:,i], real_sample[:,j], c='k', s=3, alpha=0.5)

            ax[i,j].scatter(fake_sample[:,i], fake_sample[:,j], c='r', s=3, alpha=0.5)
            ax[j,i].scatter(fake_sample[:,i], fake_sample[:,j], c='r', s=3, alpha=0.5)
    #
    return
#

""" ===================== TRAINING ======================== """


def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


G_solver = optim.SGD(G_params, lr=dt)
D_solver = optim.SGD(D_params, lr=dt)

##########
# For using LBFGS
#
# G_solver = optim.LBFGS(G_params, lr=dt)
# D_solver = optim.LBFGS(D_params, lr=dt)
#
# global ones_label
# global zeros_label
# global D_real
# global D_fake
# def eval_loss_D():
#     D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
#     D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
#     D_loss = D_loss_real + D_loss_fake
#     return D_loss
# #
# def eval_loss_G():
#     G_loss = nn.binary_cross_entropy(D_fake, ones_label)
#     return G_loss
# #

ones_label = Variable(torch.ones(m, 1))
zeros_label = Variable(torch.zeros(m, 1))


lrr = float(dt)
for it in range(niter):
    if it%10000==0 and it>0:
        m = int(1.2*m)  # Try increasing the sample size over training
                        # to force generator/discriminator to learn the variance (does this work?)
        ones_label = Variable(torch.ones(m, 1))
        zeros_label = Variable(torch.zeros(m, 1))
        # k_inner += 2
    #
    for k in range(k_inner):
        # Sample data
        # z = Variable( torch.randn(mb_size, Z_dim))
        z = Variable( noise_prior(m) )
        # X, _ = mnist.train.next_batch(mb_size)
        # X = Variable(torch.from_numpy(X))
        X = data_distribution(m)

        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_loss.backward()
        D_solver.step()
        # D_solver.step(eval_loss_D)    # Needed for LBFGS

        # Housekeeping - reset gradient
        reset_grad()
    #

    # Generator forward-loss-backward-update
    z = Variable( noise_prior(m) )
    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss = -nn.binary_cross_entropy(1-D_fake, ones_label)

    G_loss.backward()
    G_solver.step()
    # G_solver.step(eval_loss_G)    # Neededfor LBFGS

    # Housekeeping - reset gradient
    reset_grad()

    if it%100==0:
        X_np = X.detach().numpy()
        Gs_np = G_sample.detach().numpy()
        ss_real = stats.describe(X_np)
        ss_fake = stats.describe(Gs_np)

        diff_of_means = np.linalg.norm(ss_real.mean - ss_fake.mean)/np.linalg.norm(ss_real.mean)
        diff_of_covars = np.linalg.norm(np.cov(X_np.T) - np.cov(Gs_np.T))/np.linalg.norm(np.cov(X_np))

        print('Iteration %i.\n======================'%it)
        print('Relative normed difference of means: %.3e'%diff_of_means)
        # print('Relative matrix normed difference of covariances: %.3e'%diff_of_covars)

        if it%1000==0:
            print('Singular values:')
            _,sr,_ = np.linalg.svd(X_np, full_matrices=False)
            _,sf,_ = np.linalg.svd(Gs_np, full_matrices=False)
            print(sr)
            print(sf)
        #
        print("")

        # # Let's try this out.
        # lrr = 0.99*lrr
        # G_solver = optim.Adam(G_params, lr=lrr)
        # D_solver = optim.Adam(D_params, lr=lrr)
    #
    if it%5000==0:
        update_plot()
        pyplot.pause(0.1)
    #
#


ns = 400
real_sample = data_distribution(ns)
fake_sample = G(noise_prior(ns)).detach().numpy()

for i in range(d):
    for j in range(i+1,d):
        ax[i,j].scatter(real_sample[:,i], real_sample[:,j], c='k', s=10)
        ax[j,i].scatter(real_sample[:,i], real_sample[:,j], c='k', s=10)

        ax[i,j].scatter(fake_sample[:,i], fake_sample[:,j], c='r', s=10)
        ax[j,i].scatter(fake_sample[:,i], fake_sample[:,j], c='r', s=10)
#

pyplot.show(block=False)
