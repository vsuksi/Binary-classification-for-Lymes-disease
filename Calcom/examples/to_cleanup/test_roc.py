import calcom
import numpy as np
from matplotlib import pyplot

ccom = calcom.Calcom()
roc = calcom.metrics.ROCcurve()
rf = calcom.classifiers.RFClassifier()

rf.params['return_prob'] = True
rf.params['ntrees'] = 1000
rf.params['subdim_prop'] = 1.


# data,labels = ccom.load_data('./data/CS29h.csv',shuffle=True)

nsamp,nfeat = data.shape

ntr = int(nsamp*0.5)
nte = nsamp - ntr


# rf.fit(data[:ntr], labels[:ntr])
# labels_pred = rf.predict(data[ntr:])
# labels_true = labels[ntr:]

# This is from the LASSO on experiment 1 with alpha=0.1. Getting AUC=0.
# Anti-predicting every label. What's going on?
labels_true = [ 0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.]
labels_pred = [ 0.56, 0.52, 0.50154116, 0.56047978, 0.56, 0.52, 0.56, 0.52, 0.52, 0.56, 0.52, 0.56, 0.56, 0.56, 0.56, 0.52, 0.56, 0.56, 0.52, 0.56, 0.49969808, 0.52, 0.52, 0.48391488, 0.52, 0.52]

labels_true = np.array(labels_true)
labels_pred = np.array(labels_pred)

fprs,tprs = roc.evaluate(labels_true,labels_pred)
auc = roc.results['auc']

fig,ax = pyplot.subplots(1,1)

ax.plot([0,1],[0,1], lw=2, ls='--')
ax.plot(fprs,tprs, lw=2)
ax.set_xticks(np.linspace(0,1,11))
ax.set_yticks(np.linspace(0,1,11))
ax.grid(True)
ax.set_xlim([0,1])
ax.set_ylim([0,1])

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

props = dict(facecolor=[0.95,0.95,0.95], alpha=0.7)
ax.text(0.9,0.6,'AUC= %.2f'%auc, fontsize=14, ha='right', transform=ax.transAxes, bbox=props)

pyplot.show(block=False)
