import calcom
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Read in data.
## Training data
df_train = pandas.read_csv("train_data.csv")
train_labels = df_train[".outcome"]
train_features = df_train.drop(".outcome", axis=1)
## Test data
df_test = pandas.read_csv("test_data.csv")
test_labels = df_test["outcome"]
test_features = df_test.drop("outcome", axis=1)
"""
# Initialize metrics.
mean = []
std = []

# Choose classifier and specify range of regularization parameter values to be plotted against classification accuracy.
classifier=calcom.classifiers.SSVMClassifier()
C = np.arange(0, 0.5 , .005)

# Train five-fold cross-validated models and save metrics for each value of the regularization parameter.
for c in C:
    classifier.params['C'] = c

    experiment = calcom.Experiment(train_features, train_labels, classifier_list=[classifier], cross_validation='stratified_k-fold', evaluation_metric=calcom.metrics.ConfusionMatrix('acc'), folds=5)

    experiment.run()

    mean.append(experiment.classifier_results['SSVMClassifier_0']['mean'])
    std.append(experiment.classifier_results['SSVMClassifier_0']['std'])

mean = np.array(mean)
std = np.array(std)

# Plot regularization parameter values against classification accuracy.
plt.figure(figsize=(20, 10))
plt.plot(C, mean, 'k', color='#1B2ACC')
plt.fill_between(C, mean - std, mean + std, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=4, linestyle='solid', antialiased=True)

plt.yticks(np.arange(0, 1.05, .05))
plt.xticks(C[0::10])

plt.ylabel('k-fold ' + 'Accuracy' + '( k = 5)')
plt.xlabel('Regularization Parameter (Low Reg --> High Reg)')
plt.title('SSVM Classification on lyme_train.csv' + '\nby ' + 'disease state', fontsize=15)
plt.savefig(fname='Regularization parameter.png')
plt.show()
"""
# Choose classifier and regularization parameter value.
classifier=calcom.classifiers.SSVMClassifier()
## Change this if the value does not point to the high-accuracy plateau in the graph.
classifier.params['C'] = 0.1

# Train five-fold cross-validated model ten times to average an accuracy
experiment = calcom.Experiment(train_features, train_labels, classifier_list=[classifier], cross_validation='stratified_k-fold', evaluation_metric=calcom.metrics.ConfusionMatrix('acc'), folds=5)
    experiment.run()

    # Classify test samples.
ssvm_classifier = experiment.best_classifiers['SSVMClassifier']
predicted_labels = np.array(ssvm_classifier.predict(test_features))

    # Make confusion matrix and calculate metrics.
C = confusion_matrix(test_labels, predicted_labels, labels=['healthy', 'lyme'])
acc = (C[0, 0] + C[1, 1]) / np.sum(C)  # accuracy score
bsr = .5 * ((C[0, 0] / (C[0, 0] + C[0, 1])) + (C[1, 1] / (C[1, 1] + C[1, 0])))



# make a setup part where the k in k-fold and other parameters are specified
# generate the graph and choose a value for the regularization parameter before proceeding
