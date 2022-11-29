from matplotlib import pyplot
import numpy as np
# from classtrees2 import Forest
from calcom.classifiers import RFClassifier
import artificial_distributions as ad
import matplotlib.patheffects as pe


# Initialize the random forest and set some parameters
rf = RFClassifier()
rf.params['subdim_prop'] = 1
rf.params['return_prob'] = True
rf.params['ntrees'] = 200                    # Number of trees to train
# rf.params['nprocs'] = 4                      # Number of *processes* to use

# Create the training set and a grid of points containing it.
ad.reset_seed(12344411)
npoints = 2400                   # Number of points in the training dataset
ng = 41                         # Number of grid points in one direction for the right panel
mus = [[0,0],[-4,-4],[4,4]]      # (x,y) coordinates of the second cluster (first cluster is at (0,0))

tr_data,tr_labels = ad.point_clouds_2d_triple(npoints,mus, balanced=True)

# tr_labels = np.array([lab+3 for lab in tr_labels])
# tr_labels = np.array(["a" if lab else "b" for lab in tr_labels])

bounds = [tr_data[:,0].min(), tr_data[:,0].max(), tr_data[:,1].min(), tr_data[:,1].max()]
xg,yg,te_data = ad.grid_2d(bounds,ng)

#
print("Training...")
rf.fit(tr_data,tr_labels)

# Get the treeleaf distribution too.
sc,cc = rf.forest.get_treeleaf_distribution()

print("Classifying...")
te_pred = rf.predict(te_data)

##############
#
# Work is done at this point; make a pretty plot.
#

print("Plotting...")
fig,ax = pyplot.subplots(1,3, figsize=(15,5))

cs = ax[0].contour(xg,yg, te_pred.reshape(ng,ng), levels=[0.4,0.5,0.6],colors=[[1,0,0],[0.8,0,0.8],[0,0,1]], linewidths=[2,2,2])

pyplot.setp(cs.collections, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])

unique_labels = np.unique(tr_labels)
nl, = np.where(tr_labels==unique_labels[0])
pl, = np.where(tr_labels==unique_labels[1])

ax[0].scatter(tr_data[nl,0], tr_data[nl,1], c=[1,0.3,0], label='Label 0')
ax[0].scatter(tr_data[pl,0], tr_data[pl,1], c=[0,0.3,1], label='Label 1')
ax[0].legend(loc='upper left')

ax[0].set_title('Training data and decision boundaries',fontsize=14)

# Look at the classification rates in a rectangular region containing the data.
pcm = ax[2].pcolormesh(xg,yg,te_pred.reshape(ng,ng), cmap=pyplot.cm.Spectral, vmin=0,vmax=1)

cs2 = ax[2].contour(xg,yg, te_pred.reshape(ng,ng), levels=[0.4,0.5,0.6],colors=[[1,0,0],[0.8,0,0.8],[0,0,1]], linewidths=[2,2,2])
pyplot.setp(cs2.collections, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])

cs2.collections[0].set_label(r'$40\%$ decision boundary')
cs2.collections[1].set_label(r'$50\%$ decision boundary')
cs2.collections[2].set_label(r'$60\%$ decision boundary')

cbar = fig.colorbar(pcm, ticks=np.linspace(0,1,11))
ax[2].legend(loc='upper left')
ax[2].set_title('Fraction vote for label 1',fontsize=14)

ax[1].bar(sc,cc)
ax[1].set_title('Distribution of tree leaves')
ax[1].set_xlabel('Number of leaves')
ax[1].set_ylabel('Count')

fig.suptitle('Random forests; %i data points, %i trees'%(npoints,rf.params['ntrees']),fontsize=18)

pyplot.tight_layout()
fig.subplots_adjust(top=0.85)

pyplot.show(block=True)
