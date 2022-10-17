import numpy as np
from numpy.linalg import norm,qr
from matplotlib import pyplot

import calcom

# Function to generate a random data matrix.
def randomranktwo(n,m,eps=1e-2,sigmas=np.array([4.,1.])):
    U = np.zeros( (n,2) )
    for i in range(n//2):
        U[i,:] = np.array([1,0])
        U[i+n//2,:] = np.array([0,1])
    #

    U /= np.sqrt(n/2.)

    V,_ = qr(np.random.randn(m,2))

    # Some extra ugliness is needed to construct A using a sum of 
    # rank-one matrices.
    noise = eps*min(sigmas)*np.random.randn(n,m)

    data = noise
    for i in range(2):
        ui = U[:,i]
        vi = V[:,i]
        ui.shape = (n,1)
        vi.shape = (m,1)
        data += sigmas[i] * np.dot(ui,vi.T)

    return data
#

############################
#
# Set parameters.
#

n = 40
m = 15
eps = 1e-2

# Create a random tall and skinny matrix which is rank-two plus noise.
data = randomranktwo(n,m,eps)
label = np.vstack((np.zeros((int(n/2),1)),np.ones((int(n/2),1))))
readable_label_map = {0:'class 1',1:'class 2'}

ltsavis = calcom.visualizers.LTSAVisualizer()

nn = 30
dim = 3
coords = ltsavis.project(data,label,readable_label_map,nn,dim)
ltsavis.visualize(coords)

