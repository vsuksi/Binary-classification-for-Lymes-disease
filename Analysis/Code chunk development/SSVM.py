import calcom
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

dftrain = pandas.read_csv("train_data.csv")
train_labels = dftrain[".outcome"]
train_features = dftrain.drop(".outcome", axis=1)

mean = []
std = []
min_vals = []
max_vals = []

C = np.arange(1, 10, .2)
# Chance reg parameter to ndarray if int
if (type(C) == float) or (type(C) == int):
          C = [C]

classifier=calcom.classifiers.SSVMClassifier()
C = np.arange(0, 1 , .01)

#repeat 100 times?
for c in C:
    classifier.params['C'] = c

    experiment = calcom.Experiment(train_features, train_labels, classifier_list=[classifier], cross_validation='stratified_k-fold', evaluation_metric=calcom.metrics.ConfusionMatrix('acc'), folds=2)

    experiment.run()

    mean.append(experiment.classifier_results['SSVMClassifier_0']['mean'])
    std.append(experiment.classifier_results['SSVMClassifier_0']['std'])
    min_vals.append(experiment.classifier_results['SSVMClassifier_0']['min'])
    max_vals.append(experiment.classifier_results['SSVMClassifier_0']['max'])

mean = np.array(mean)
std = np.array(std)
min_vals = np.array(min_vals)
max_vals = np.array(max_vals)

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

classifier.params['C'] = c

experiment = calcom.Experiment(train_features, train_labels, classifier_list=[classifier], cross_validation='stratified_k-fold', evaluation_metric=calcom.metrics.ConfusionMatrix('acc'), folds=2)

experiment.run()



# make a setup part where the k in k-fold and other parameters are specified
# generate the graph and choose a value for the regularization parameter before proceeding

# how to save the graph in such
