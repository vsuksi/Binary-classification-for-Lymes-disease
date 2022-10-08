# Questions and clarifications

**What is Skyline?**
 A freely available, open-source software tool for targeted quantitative mass spectrometry method development and data processing with a 10 year history supporting six major instrument vendors

**Which scripts require variable user input?**
feature_selection.py, iterative_feature_removal.py, test_targeted_features.py, test_targeted_log_features.py**

**What is the cost function in the case of feature_selection.py (SSVM, calcom.classifiers.ssvm)**
I can't be sure without accessing the the calcom library, but in general, the cost function for SVM is

**What is the variable hyperparameter use case in feature_selection.py?**
Cost parameter, although I don't know how it was determined.

**What is the variable hyperparameter use case in iterative_feature_removal.py?**
Probably what type of  normalization and imputation is chosen**

**What is the use case in test_targeted_features.py?**
What type of normalization.

**What is the use case in test_targeted_log_features.py?**
Do you need to check that the classifier implementations works identically in R and Python?

**How does the identification level relate to the accuracy of the model?**
The accuracy of the predictions should, on average, be better with improved quality of data, since there is a smaller risk of metabolite misidentification and a clearer signal.**

**How will the training data be split?**
The training data is split to 80 % training, 20 % validation, and that's it; use the exact same sets as Dorde.

**What is the training-test split?**
80 % training, 20% split?

**How does iterative feature removal work?**
In iterative feature removal, a model is fit to data repeatedly but after each iteration, the most important features of the earlier iterations are left out. This is motivated by dubious claims of parsimony in biological 

**Is calcom the library that wasn't available in R? It is not a data cleaning library as I thought, but perhaps feature engineering?**
Indeed, calcom is not availabe in R.

**What libraries and packages are used in the scripts?**
metabolomics(calcom, copy from deepcopy, missingpy) confusion_matrix from sklearn.metrics, numpy, pandas, datetime, os, re + pickle (in the optional test_targeted_log_features.py)

What is the difference between features and targeted features?
>targeted features are a subset of features which build the best possible model while avoiding overfitting

Who is Djordje?
>**Djordje is the guy who did the Lyme's disease work in R, but he used other classifiers in addition to**

What classifiers did Djordje use?
>Djordje used - logistic regression, decision trees, random forest, support vector machines with radial basis kernel, gradient boosted trees and k-nearest neighbors

What does is mean that a split is stratified in Random Forest Classifier?
> A stratified split splits the data with respect to something, for example so that the validation set has. This is especially important for small datasets, where random sampling is more likely to result in an uneven proportions of  Indeed,

 Which hyperparameters should be tuned nd what are they exactly?
 >- Cost hyperparameter: L2-regularized logistic regression and L1- and L2-regularized SVM with linear and radial basis function kernels**
**- The sigma hyperparameter is tuned for SVM with radial basis function kernel and controls the range of a single training instance. For a large value of sigma, the SVM decision boundary will rely on the points that are closest to the decision border**



 How are the hyperparameters tuned?
 > The hyperparameters are tuned using a k-fold cross-validation on the training data, where the average CVAUROC for each hyperparameter is used for running the test set.
