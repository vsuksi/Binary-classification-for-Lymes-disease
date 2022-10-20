from sklearn import svm
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in train data:
dftrain = pd.read_csv('train_data.csv', sep=',', header=None)
dftrain = dftrain.to_numpy()
ytrain = np.copy(dftrain[1:, 0])
xtrain = np.copy(dftrain[1:, 1:])

# Read in test data
dftest = pd.read_csv('test_data.csv', sep=',', header=None)
dftest = dftest.to_numpy()
ytest = np.copy(dftest[1:, 0])
xtest = np.copy(dftest[1:, 1:])

# Fit model
svm_model = svm.SVC()
svm_model.fit(xtrain, ytrain)

# Predict test cases
prediction = svm_model.predict(xtest)

# Metrics
#fpr, tpr, thresholds = metrics.roc_curve(ytest, prediction, pos_label="lyme")
#roc_auc = metrics.auc(fpr, tpr)
accuracy = metrics.accuracy_score(ytest, prediction)
print(accuracy)

#display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, #estimator_name='SVM', pos_label="lyme")
#print(prediction)

#roc_auc = metrics.auc(fpr, tpr)



#how do we make it a string method?
