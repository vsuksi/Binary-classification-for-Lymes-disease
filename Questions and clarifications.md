# Questions and clarifications

**This is a repository for critical information regarding the project at hand. Formulating it into a question allows for a specific viewpoint to be addressed or an implication to be hinted at, for example. Include synonyms when applicable.**

**What are the pros and cons of running the regularization parameter loop repeatedly in order to generate an graph averaged over repeated runs instead of simply running the graph several times and checking by eye?**
If included in the final pipeline

**Why can't you simply choose the regularization parameter value which gives the best accuracy**
Because of the plateau; there is a range of regularization parameter values which perform similarly. Thus, choosing the one that gives the best accuracy could be one that doesn't incur penalty in an optimal fashion. This also prevents over-fitting to the training data, giving you the best bet at predicting classes in the test data.

**Why can't you use the regularization parameter value from Kehoe directly?**
If Kehoe had used log10-transformation and kNN, this would probably work, especially if valued somewhat higher just to be safe. Thing is, Kehoe used log2 transformation and the exact value of the regularization parameter is not mentioned in Kehoe.

**How is the hyperparameter value C chosen in Kehoe et al.?**
The hyperparameter value C between 1 and 10 with intervals of 0.2 is chosen from a plot generated using "classify()", a function in "metabolomics.py".

**Is the SSVM classifier, as implemented through Calcom experiment class in SSVM_for_qmd.py, the same as Dorde's run_ml()?**
The strata in cross-validation are random anyways, so the experiment class works out similarly to run_ml() but without the validation partition. The 100 times repeating five-fold cross validation is for choosing the best lambda value to predict the labels in the test data. Repeated k-fold cross-validation also serves as an estimate of accuracy.

UMAP was used to evaluate the feature selection by kFFS using different imputation methods.

**How can you get the python code to use the exact same data split?**
There are three options: split the data in a separate file which can be  called upon by the main file and the python file, use reticulate within the same file or do the data split with the same seed in the python file which is Quarto included.

**Is the accuracy/number of features plot best implemented by using the Wilcox sum rank test on the data that is read in Leo's pipeline (has kNN performed)? Couldn't the missing values imputed by kNN result in false ranking of features, where some features get more important somehow?**
It should be ok to use the Wilcoxon method on the imputed data. This is sufficient since Wilcoxon is after kNN in Dorde's thesis! Moreover, since kNN works by taking the average of the five nearest neighbors for a specific variable in feature space, features with imputed values are unlikely to provide better predictive accuracy or perform better in the Wilcoxon sum rank test.

**How to plot the number of features against accuracy?**
Make a grid to run the model with differing amounts of features. Save results after repeated runs for each feature amount.  Number of features on x-axis, accuracy on y-axis.

**How to rank the features? Or is there a file with the features ranked á la Wilcoxon?**
You can use the "Feature selection" file from Dorde with binary_outcome.csv to generate the ranked list of features á la Wilcoxon!

**What are the user inputs in the pipeline?**
Data, method and whether to execute the extended pipeline with feature importance and influence of feature amount. Another good option would be to make the extended version a separate script.

**In your presentation, how are you going to deal with feature importance code chunks taking a long time to run?**
I'll run them beforehand, use cache=true and "include" them in main presentation file.

**Is it worth including labels for code chunks?**
Yes, following in Leo's footsteps.

**How are you going to include the optional feature importance code chunk?**
Navigate to the code chunk and change setting eval:true.

**Is there a good use case code chunk cache?**
Setting cache=true for the SSVM code chunk could be handy since it shouldn't be used in the real-world application, yet it is good to include for validation.

**Should I include the SSVM in the report?**
Probably best to include the SSVM in the report as #not run. Comment why it is not a usable option: the creators of the library noted that it should be used for validation only, as development was halted and they don't have confidence in the package for real-world applications.

**Can you simply hold out the test set for the SSVM in order to get reproducible results?**
Yes, if I can figure out how to use the fitted model to classify test data samples.

**If you make a heatmap of top features, how could you include the results Kehoe achieved**?
Make a heatmap of the top features and highlight the ones that were found to have high feature importance in Kehoe's work

**How does dataset.csv and dataset_after_knnin GitLab differ?**
Dataset_after_KNN is with the original "EDL", "HDL" etc. outcomes, while dataset.csv has "lyme" and "healthy" as outcomes.

**How many combinations of transformation and imputation schemes did Kehoe evaluate?**
Kehoe evaluated 18 different combinations of transformation and imputation schemes. The chosen ones:
KNN imputation with log transformation
Median imputation with log transformation
KNN imputation with median-fold change normalization
KNN imputation with standardization

**How are site and measurement batch effect different from one another?**
Site effects refer to how the metabolic functioning in a population can differ depending on the site, because of environment, culture and more. Measurement batch effect refers to differential results obtained because of instrument setup and use.


**How does Dorde split do 3 random data splits?**
By choosing different seeds. Logistics model with 3 random data splits means training the model 3 times on random 75% of training data and cross validated on the 75% 100 times before testing on the remaining 25%.

**What does intensity and mass/charge ratio mean in the context of LC/MS?**
Intensity is the relative abundance while the mass/charge ratio and retention time describes the identity of the feature.

**What is the clinical data and why is it used?**
The clinical dataset contains information about the patients including whether they have Lyme's disease.

**Are Dorde's and Kehoe's results comparable in any meaningful sense?**
Barely, because there is no batch effect in Dorde's approach as the test set came from the same batch. Kehoe's test set was from another batch.

**Was the Kehoe test set from a different batch altogether?**
Indeed, the test set Kehoe uses is from a different batch, establishing model model performance on samples from different batches


**The justification for training with old and testing with new data has to do with the reality of batch effects, perhaps?**
Probably, since batch effects will be present in the real-life application due to different LC/MS-equipment and other factors.

**How does LC/MS work?**
LC separates metabolites by their retention time, MS by mass

**Which features contributed to the batch effect in Kehoe?**
Features 2198, 206, 682 and 147 were identified to contribute to batch effects from the log/knn, median-fold change/knn, standard/knn and raw/knn methods, respectively. After removal of these features, the batch effect disappeared for the healthy control samples. These features did, however, contribute to the separation of disease state (EDL vs ELL).

**What preprocessing methodology worked best for Kehoe?**
KNN imputation with log transformation.

**Is it problematic that Kehoe did no correlation analysis of the features?**

**Can I simply feed the pipeline (disp_features(result ,1:45))  in order run the pipeline with Leo's suggested amounts of features?**
This should be possible, try it out!

**Is Dorde even using Wilcoxon to limit the number of features?**
Turns out that Dorde isn't limiting the amount of features.

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

**What is order and what is id in the PeakTable from Kehoe sent to Skyline?**
Not sure what order is but it is not important. ID is where it's at for tracking samples.

**What does the Wilcoxon p-value mean in the context of this project?**
Given a null hypothesis, what is the probability that a given feature's rank sum intensity differs from that of the population rank sum intensity by chance?

**What disparities in feature importance are to be expected due to the different feature selection methodology across Dorde and Kehoe?**
Since Dorde used a filter method based on statistical inference for feature selection, features weren't chosen for their predictive importance. Instead features were chosen with regards to the distribution of features' intensities. This may lead to very different features being chosen from Kehoe's embedded selection.

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
 The hyperparameters are tuned using a k-fold cross-validation on the training data, where the best hyperparameter value for average CVAUROC (from 100 repetitions) for each hyperparameter is used for running the test set.
