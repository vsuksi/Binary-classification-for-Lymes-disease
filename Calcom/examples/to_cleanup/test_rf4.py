# Another random forests test.
# Pick out two dimensions out of 100 to have variation on.
# Remaining dimensions all have iid white noise plus a shift.
# See if the RF code can pick out the dimensions.

import numpy as np
from matplotlib import pyplot
import calcom
from calcom.metrics import ConfusionMatrix
# from calcom import Experiment


d = 100     # Number of dimensions
r = 2       # Number of dimensions with variation
n = 200    # Number of samples.

labels = np.zeros(n)
labels[n//2:] = 1

shuffling = np.random.permutation(n)
labels = labels[shuffling]

data = np.zeros( (n,d) )
selected_dims = np.random.permutation(d)[:2]

shifts = 50*np.random.randn(d)


if False:
    for i in range(d):
        data[:,i] = 0.1*np.random.randn(n) + shifts[i]
        if i in selected_dims:
            data[n//2:,i] = 0.1*np.random.randn(n//2) - shifts[i]
        #
    #
else:
    for i in range(d):
        data[:,i] = 0.1*np.random.randn(n) + shifts[i]
    #
    # Make the data slightly non-separable, and co-dependent on the two dimensions.
    data[:n//2,selected_dims[0]] = 2 + np.random.randn(n//2)
    data[n//2:,selected_dims[0]] = -2 + np.random.randn(n//2)

    data[:n//2,selected_dims[1]] = -2 + np.random.randn(n//2)
    data[n//2:,selected_dims[1]] = 2 + np.random.randn(n//2)

#


data = data[shuffling,:]

myrf = calcom.classifiers.RFClassifier()
myrf.params['ntrees'] = 100
myrf.params['subset_prop'] = 1.
myrf.params['subdim_prop'] = 0.2

myclflist = [myrf]

bsr = ConfusionMatrix('bsr')

expObj = calcom.Experiment(
    data = data,
    labels = labels,
    classifier_list = myclflist,
    cross_validation = "stratified_k-fold",
    folds = 5,
    # cross_validation = "stratified_k-fold",
    evaluation_metric = bsr,
    verbosity = 1
)

best_classification_models = expObj.run()

bestrf = best_classification_models['RFClassifier']
ts,torder,ds,dorder = bestrf.isolate_subsets(data,labels)

def count_split_dims(k):
    # Get splitting dimensions of top 10 trees.
    splitdimct = np.zeros(d)
    for i in range(k):
        subidxs = bestrf.forest.tree_subspaces[torder[i]]
        tree = bestrf.forest.trees[torder[i]]
        for node in tree.tree:
            if len(node.decision):  # Leaves don't add extra information
                splitdimct[subidxs[node.decision[0]]] += 1
            #
        #
    #
    topksplitdim = np.argsort(-splitdimct)
    return splitdimct,topksplitdim
#


print("Best individual trees:")
for i in range(10):
    print("%5.3f, %5i" % (ts[torder[i]], torder[i]))
#
# print(torder[:10])
# print(ts[torder[:10]])
print("Best aggregate dimension scores:")
for i in range(10):
    print("Dimension %5i : %5.3f" % (dorder[i],ds[dorder[i]]))
#

spldim10,top10spl = count_split_dims(10)
print("Occurrences of dimension when splitting, amongst the top 10 trees:")
for i in range(sum(spldim10>0)):
    print("Dimension %5i : Occurrences: %5i" % (top10spl[i],spldim10[top10spl[i]]) )
#

spldim100,top100spl = count_split_dims(100)
print("Occurrences of dimension when splitting, amongst the top 100 trees:")
for i in range(sum(spldim100>0)):
    print("Dimension %5i : Occurrences: %5i" % (top100spl[i],spldim100[top100spl[i]]) )
#

# print(dorder[:5])
# print(ds[dorder][:5])
fig,ax = pyplot.subplots(1,2)
ax[0].imshow(data)
ax[1].scatter(data[:,selected_dims[0]],data[:,selected_dims[1]],c=labels)
ax[1].set_xlabel("Dimension "+str(selected_dims[0]))
ax[1].set_ylabel("Dimension "+str(selected_dims[1]))


pyplot.show(block=False)
