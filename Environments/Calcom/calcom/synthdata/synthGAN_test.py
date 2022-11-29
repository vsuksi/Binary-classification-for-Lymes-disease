import calcom
import numpy as np
from matplotlib import pyplot

ns = 400
d = 3

def data_distribution(nsamp):
    return 2 + 0.5*np.random.randn(nsamp,d)
#
data = data_distribution(ns)
labels = np.zeros(ns, dtype=int)

gan = calcom.synthdata.GANGenerator()
gan.params['n_iter'] = 100000
gan.params['verbosity'] = 2
gan.params['lr'] = 1e-3
gan.params['k_inner'] = 5
gan.params['minibatch_size'] = 100

gan.fit(data,labels)
fake_sample = gan.generate(labels)

fig,ax = pyplot.subplots(d,d, sharex=True, sharey=True, figsize=(8,8))

for i in range(d):
    for j in range(i+1,d):
        ax[i,j].scatter(data[:,i], data[:,j], c='k', s=10)
        ax[j,i].scatter(data[:,i], data[:,j], c='k', s=10)

        ax[i,j].scatter(fake_sample[:,i], fake_sample[:,j], c='r', s=10)
        ax[j,i].scatter(fake_sample[:,i], fake_sample[:,j], c='r', s=10)
#

pyplot.show(block=False)
