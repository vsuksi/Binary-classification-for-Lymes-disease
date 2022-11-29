import numpy as np
from calcom.utils import animals
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot
import numpy as np

import matplotlib_style

# let's start with a generic 3d scatter plot.
n_classes = 3
d = 3

class_sizes = [np.random.choice(np.arange(40,121)) for _ in range(n_classes)]
n = sum(class_sizes)

class_names = np.random.choice(animals.animals, n_classes)
labels = np.concatenate( [ np.repeat(cn,cs) for cn,cs in zip(class_names, class_sizes) ] )

data = np.random.randn( n,d )

# Plot by class.
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

# this will be the "cleaned up" plot.
fig2 = pyplot.figure()
ax2 = fig2.add_subplot(111, projection='3d')

ec = {l:np.where(labels==l)[0] for l in np.unique(labels)}
for k,v in ec.items():
    ax.scatter(data[v,0], data[v,1], data[v,2], label=k)
    ax2.scatter(data[v,0], data[v,1], data[v,2], label=k)
#
ax.legend()
ax2.legend()

fig3,ax3 = pyplot.subplots(1,1)

for k,v in ec.items():
    ax3.scatter(data[v,0], data[v,1])
#

# apply the cleaning
matplotlib_style.clean_scatter(fig2,ax2)


fig.suptitle('Original', fontsize=24)
fig2.suptitle('Cleaned', fontsize=24)


#    fig.savefig('before.png', dpi=120, bbox_inches='tight')
#    fig2.savefig('after.png', dpi=120, bbox_inches='tight')

fig.show()
fig2.show()
fig3.show()

pyplot.ion()
