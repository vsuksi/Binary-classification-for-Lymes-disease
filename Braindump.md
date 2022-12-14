# Braindump

**The braindump is a repository for the fleeting thoughts that further the project at hand but are better picked up at a later stage.**

Use C=0.1

Include the regularization parameter plot in the Quarto document with an accompanying text explaining that it if the value of the regularization parameter in the code is below the plateau in the graph, it can be adjusted manually in the .qmd file. In order to make the comparison between the regularization parameter value and the graph, make sure to include the used regularization parameter value in the Quarto document. For example: "The regularization parameter may be adjusted if the regularization parameter value does not point to the high-accuracy plateau in the graph".

Fuck it, they Kehoe's regularization parameter plot doesn't tell what is in question. Thus, we could either make a plot, which would make it include variable user input. This is not cool for the report since it is not possible to have variable user input in a Quarto document. Thus, better choose a value of C that works for the data processing at hand, that is log10 transformation. Try generating a graph and choose a value of C slightly higher than necessary to account for the possibility variability in optimal C between datasets. The fact that the data is from two batches helps a bit. You could also check the value for the regularization parameter by plotting the new data that Leo has.

Further, you could do a separate graph from the new batch that Leo has.

Although k-fold feature selection was used by Kehoe, the regularization parameter lambda was used on all 4851 features. Moreover, Kehoe also used log ranking.

Choose the smallest value for the regularization hyperparameter C which doesn't see a decline.
But should I repeat the cross-validation 100 times the SSVM classifier? I think so. Save the acc and bsr stats to separate lists.

ssvm_classifier = experiment.best_classifiers['SSVMClassifier']

####
import calcom
import os
import pandas
import numpy as np

mean = []
std = []
min_vals = []
max_vals = []

C = np.arange(1, 10, .2)
      # Chance reg parameter to ndarray if int
if (type(C) == float) or (type(C) == int):
          C = [C]

classifier=calcom.classifiers.SSVMClassifier()
for c in C:
    # Run experiment
    classifier.params['C'] = c

    experiment = calcom.Experiment(features, labels, classifier_list=[classifier], cross_validation='stratified_k-fold', evaluation_metric=calcom.metrics.ConfusionMatrix('acc'), folds=5)

    experiment.run()

    # Store accuracy scores
    mean.append(experiment.classifier_results['SSVMClassifier_0']['mean'])
    std.append(experiment.classifier_results['SSVMClassifier_0']['std'])
    min_vals.append(experiment.classifier_results['SSVMClassifier_0']['min'])
    max_vals.append(experiment.classifier_results['SSVMClassifier_0']['max'])

mean = np.array(mean)
std = np.array(std)
min_vals = np.array(min_vals)
max_vals = np.array(max_vals)

    ssvm_classifier = experiment.best_classifiers['SSVMClassifier']




features_train = dftrain.drop(".outcome", axis=1)
features_test = pdfeatures.values

mean_acc = []
std_acc = []
min_vals_acc = []
max_vals_acc = []

classifier=calcom.classifiers.SSVMClassifier()
classifier.params['C'] = 1.0

experiment = calcom.Experiment(features, labels, classifier_list=[classifier], cross_validation='stratified_k-fold', evaluation_metric=calcom.metrics.ConfusionMatrix('acc'), folds=5)

experiment.run()

ssvm_classifier = experiment.best_classifiers['SSVMClassifier']

df_test = pandas.read_csv("test_data.csv")

labels_true = df_test["outcome"]

features_test = df_test.drop("outcome", axis=1)
features_test = pdfeatures.values

labels_predicted = np.array(ssvm_classifier.predict(features_test))

C = confusion_matrix(labels_true, labels_predicted, labels=['healthy', 'lyme'])
acc = (C[0, 0] + C[1, 1]) / np.sum(C)  # accuracy score
bsr = .5 * ((C[0, 0] / (C[0, 0] + C[0, 1])) + (C[1, 1] / (C[1, 1] + C[1, 0])))

####

SSVM_model.predict()

Use SSVM_model.predict()

experiment = calcom.Experiment(features, labels, classifier_list=[classifier], cross_validation='stratified_k-fold', evaluation_metric=calcom.metrics.ConfusionMatrix('acc'), folds=5)


SSVMClassifier from Calcom is based on scikit, right?

Make separate files, extended and non-extended

I believe Dorde's pipeline stops at validation since there is no sequestering of test data. The logical follow-up here is, as per Leo's "Analysis" in GitLab, to use this to get the hyperparameters for the dataset for training on the entirety of the training data. Dorde's report also includes metric estimates for classifier performance which can inform the decision of model.

Maybe make a separate chunk for defining the method, cv_times, training fraction, kfold and specify default values

Fix the training fraction: 0.98 (r training_frac)

What do you need to do in order to make the feature number plot?
- limit the number of features from wilcox; use the same functions as Dorde to specify the number of important features involved
- make a barplot in the same way as the feature importance?

4 clusters in analysis.rmd plot is for visualizing batch effect?

lyme_data.csv includes all features, you can select 10, 50 100 from there

Rmember to use inline code in Quarto, for example `r 2 * pi`

Metric: _accuracy, roc_curve.py, _confusionmatrix.py

Try to pass the working .csv directly from r as per the reticulate website

The Dorde identifiers are ID's. Using them you can check whether the most important features have been validated by Kehoe, and overall do a comparison between the most important features across Dorde and Kehoe.

10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10,000 piirrettä in glmnet.

- is it soft margin or hard margin ssvm? Check the code.
can't access the information since I don't have the calcom library.
