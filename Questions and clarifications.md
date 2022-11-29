# Questions and clarifications

**This is a repository for critical information regarding the project at hand. Formulating it into a question allows for a specific viewpoint to be addressed or an implication to be hinted at, for example. Include synonyms when applicable.**

**At what point is kNN-imputation done in Leo's analysis pipeline? Is there a separate script for it?**
There is not a separate script for it. Preprocessing of the data, which includes kNN-imputation, can probably vary bit depending on the source. As such, the preprocessing is not included in the pipeline.

**Should I always report the baseline accuracy? Couldn't one say that this is unimportant since it should always be 0.5 in binary classification?**
Alternatively, one could argue that showing the model has a baseline of around 0.5 is a checkpoint of sorts to show that the model works.

**Concepts can take a while to crystallize in a seamless formulation; meanwhile, try to formulate the matter as it often means progress.**

**How are you going to communicate the calcom library and the environments.yml? Can calcom be installed by using the yml in Conda?**

**Is the test data that Leo used in GitLab the same as test data from seed=2019 in mikropml?**

**Is the Turku data completely different from the data I've been working with for the last month? Is this how batch effects are taken into consideration?**

**What is the TreeSummarizedExperiment and why is it used by Leo in GitLab/pipeline-dorde/analysis/report.rmd?**
It is an extension of SummarizedExperiment from Bioconductor, which is based on the R matrix datatype. Not sure if there is a good reason for it to be used by Leo for the dataset I've been working with; instead it is probably used for the practical application when data comes raw from patients with multiplicate assays. Even then the tree function shouldn't be needed, and indeed Leo does

**How are the feature importance results presented?**
The feature importances are presented as a mean difference between test and permuted AUROC for each feature; that is, each feature is permuted 100 times and the mean accuracy for models based on the permuted feature is compared against the accuracy for the unpermuted model.

**What is UMAP?**
Uniform Manifold Approximation and Projection,UMAP, can show clustering of very high-dimensional data.

**What preprocessing does Dorde use?**
He uses the log10 transformation as an argument in the preproc() function from the Metabolomics Data Analysis Toolbox library.

**What is the difference between Test.R and Train models.R?**
In Train models.R, the training dataset is split into three training

**What is order and what is feature in the PeakTable from Kehoe sent to Skyline?**
Not sure what order is but it is not important. However, the targeted feature

**What does the Wilcoxon p-value mean in the context of this project?**
Given a null hypothesis, what is the probability that a given feature's rank sum intensity differs from that of the population rank sum intensity by chance?

**What disparities in feature importance are to be expected due to the different feature selection methodology across Dorde and Kehoe?**
Since Dorde used statistical inference for feature selection, features weren't chosen for their predictive importance. Instead features were chosen with regards to the distribution of features intensities. This may lead to very different features being chosen.
Statistical inference is faster

**How does the testing AUC relate to training AUC in a successful model?**
The testing AUC ought to be similar to the cross-validated training AUC.

**What is the lambda hyperparameter in logistic regression?**
The lambda hyperparameter is a measure of regularization, where
**Are feature importances from Dorde are already ranked across the whole dataset since he used cv**

**The XG-boost model slightly better, why not use it instead?**

**Does Dorde include features unique to the site collection (batch effect)? They are removed in Kehoe.**

**Which validation parameter was given most attention in Dorde?**

**Why is the data log-transformed before feature selection by Wilcoxon in Dorde?**

**What test is used, exactly, and why?**
Wilcoxon rank-sum test (Mann-Whitney U test). It is tolerant against outliers which are abundant in clinical data, which is problematic, and being non-parametric, it works better for data that isn't normally distributed. This is often the case in biomedical data.

**How to make a plot with the number of features**
Number of features on x-axis, accuracy on y-axis

**How does L1 (Lasso) and L2-regularized (Ridge) regression differ? Which one performed better in Dorde?** Lasso regression adds the absolute weight coefficient as a penalty term to the cost function, while Ridge regression adds the square of the weight coefficient to the cost function.
L2-regularized logistic regression was used in glmnet.

**What exactly is the automated pipeline?**
Inputs: A dataset with feature vectors being in the columns, outcome variable, training method, parameters.
1. Data organization
2. Missing value imputation using k-nearest neighbors
3. Feature selection and model fitting
4. Model evaluation: comparison, visualization
5. Return dataset following kNN imputation, feature ranking from best to worst,
UMAP charts, PCA plot, Cluster plot, Probability ellipse plot, and model per-
formance.

**How does Dorde deal with the site batch effect?**

**What are the pros and cons of the approaches to feature quality between Dorde and Kehoe?**
The targeted Kehoe features are quite interpretable in that they correspond to 42 known metabolites thus can be used as biomarkers. Given sufficient amounts of data from different batches (both site and instrument effects) would also mitigate batch effects, and Dorde would indeed

**How many features did Dorde end up using? Is it possible that the PCA ended up identifying many of the same metabolites as Kehoe?**
1023 features from 4800 features in the original dataset.


**Is the quality of the features important for the diagnostic application?**
Quality of features is important as if the features are not targeted, batch effects from measurement devices can be pronounced. Measurement of targeted features can also provide insight into symptoms experienced by the patient.

**Does the Dorde pipeline produce quality targeted features in the same way as the Kehoe pipeline. That is, does it take into account the quality of the features?**
I believe Dorde's pipeline works on raw data

**Is the method about decreasing the number of features the same as iterative feature removal?**
No. For the sake of computational cost, Leo is interested in the amount of features needed to still be able to produce results on par with the full feature set.

**What does it mean that the IFR removes batch-discriminatory features?**
It simply means that between the two healthy control groups, features unique to the groups are removed.

**Is it possible to adjust the number of features used in logistical regression classifier glmnet?**
This is unclear, since

**Are iterative feature removal (IFR) and the method suggested by Leo the same?**
No, it is not the same thing, since iterative feature removal works by removing the most important features in each iteration.

**Does Dorde use k-fold feature selection?**
No, he uses k-fold cross validation for choosing optimal values for hyperparameters.

**How does K-fold feature selection work (kFFS) in Kehoe et al. and how does it differ from Dorde's feature selection?**
The training data is fitted k-fold using the SSVM classifier, and  a vote for the most important features is obtained by repeating the fitting with random subsets for each fol. The most highly voted features across imputation methods were then sent for targeting in Skyline.
The average of the folds' accuracy is an estimate of the model's accuracy. Random subsets of data in the cross-validation scheme can result in small differences in obtained feature sets.

**How does combining features sets work in Kehoe et al.?**
The max of the following features is chosen: mono-isotopic vs isotopic ion, intact vs insource fragment ion, and ion intensity. This is done not for the accuracy of the model, but for the interpretability of the results.

**How did targeted features compare to non-targeted features (Kehoe et al.)?**
Targeted features did better, at a sensitivity of

**What is five-fold cross validation repeated one hundred times?**
K-fold cross validation is used to estimate how well a model performs on new data and is good at preventing bias.

**What is Skyline?**
 A freely available, open-source software tool for targeted quantitative mass spectrometry method development and data processing.

**Which scripts require variable user input?**
feature_selection.py, iterative_feature_removal.py, test_targeted_features.py, test_targeted_log_features.py**

**What is the cost function in the case of feature_selection.py (SSVM, calcom.classifiers.ssvm)**
I can't be sure without accessing the the calcom library, but in general, the cost function for SVM is

**What is the variable hyperparameter use case in feature_selection.py?**
Cost parameter, although I don't know how it was determined.

**What is the variable hyperparameter use case in iterative_feature_removal.py?**
Probably what type of  normalization and imputation is chosen.

**What is the use case in test_targeted_features.py?**
What type of normalization.

**What is the use case in test_targeted_log_features.py?**

**Do you need to check that the classifier implementations works identically in R and Python?**

**How does the identification level relate to the accuracy of the model?**
The accuracy of the predictions should, on average, be better with improved quality of data, since there is a smaller risk of metabolite misidentification and a clearer signal.**

**How will the training data be split?**
The training data is split to 80 % training, 20 % validation, and that's it; use the exact same sets as Dorde.

**What is the training-test split?**
80 % training, 20% split?

**How does iterative feature removal work?**
In iterative feature removal, a model is fit to data repeatedly but after each iteration, the most important features of the earlier iterations are left out. This is motivated by criticism of the validity of parsimony in biological data.

**Is calcom the library that wasn't available in R? It is not a data cleaning library as I thought, but perhaps feature engineering?**
Indeed, calcom is not availabe in R.

**What libraries and packages are used in the scripts?**
metabolomics(calcom, copy from deepcopy, missingpy) confusion_matrix from sklearn.metrics, numpy, pandas, datetime, os, re + pickle (in the optional test_targeted_log_features.py)

**What is the difference between features and targeted features?**
Targeted features are a subset of features which build the best possible model while avoiding overfitting.

**Who is Djordje?**
Djordje is the guy who did the Lyme's disease work in R, but he used other classifiers in addition to.

**What classifiers did Djordje use?**
Djordje used  logistic regression, decision trees, random forests, support vector machines with radial basis kernel, gradient boosted trees and k-nearest neighbor.

**What does is mean that a split is stratified in Random Forest Classifier?**
A stratified split splits the data with respect to something, for example so that the validation set target classes are in proportion to that of the proportions of . This is especially important for small datasets, where random sampling is more likely to result in an uneven proportions of

**Which hyperparameters should be tuned nd what are they exactly?**
Cost hyperparameter: L2-regularized logistic regression and L1- and L2-regularized SVM with linear and radial basis function kernels.

**What is the sigma hyperparameter in SVM with radial basis function kernel?**
The sigma hyperparameter is tuned for SVM with radial basis function kernel and controls the range of a single training instance. For a large value of sigma, the SVM decision boundary will rely on the points that are closest to the decision border**

 **How are the hyperparameters tuned?**
 The hyperparameters are tuned using a k-fold cross-validation on the training data, where the greatest hyperparameter value for average CVAUROC (from 100 repetitions) for each hyperparameter is used for running the test set.
