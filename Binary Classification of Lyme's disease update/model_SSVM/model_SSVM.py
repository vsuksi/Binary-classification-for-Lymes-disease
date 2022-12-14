import calcom
import os
import pandas
import numpy as np

# Read in data.
df = pandas.read_csv("lyme_data.csv")

# Shuffle the outcome column if modeling for baseline accuracy.
np.random.shuffle(df["outcome"])

# Designate outcome column.
labels = df["outcome"]

# Designate features.
pdfeatures = df.drop("outcome", axis=1)
features = pdfeatures.values

# Initialize metrics.
mean_acc = []
std_acc = []
min_vals_acc = []
max_vals_acc = []

# Set classifier and specify penalty parameter ['C']
classifier=calcom.classifiers.SSVMClassifier()
classifier.params['C'] = 1.0

# Run ML-pipeline in a similar manner to Dorde's run_ml(); five-fold cross validation is performed 100 times. Save metrics in lists.
for i in range(100):
    SSVM_model = calcom.Experiment(features, labels, classifier_list=[classifier], cross_validation='stratified_k-fold', evaluation_metric=calcom.metrics.ConfusionMatrix('acc'), folds=5)

    SSVM_model.run()

    mean_acc.append(SSVM_model.classifier_results['SSVMClassifier_0']['mean'])
    std_acc.append(SSVM_model.classifier_results['SSVMClassifier_0']['std'])
    min_vals_acc.append(SSVM_model.classifier_results['SSVMClassifier_0']['min'])
    max_vals_acc.append(SSVM_model.classifier_results['SSVMClassifier_0']['max'])

# Write metrics to file.
with open("model_SSVM_results", "w") as file:
    for i in mean_acc:
        file.write(f"{i}, ")
    file.write("\n")
    for i in std_acc:
        file.write(f"{i}, ")
    file.write("\n")
    for i in min_vals_acc:
        file.write(f"{i}, ")
    file.write("\n")
    for i in max_vals_acc:
        file.write(f"{i}, ")
