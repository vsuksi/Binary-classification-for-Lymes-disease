# Braindump

**The braindump is a repository for the fleeting thoughts that further the project at hand but are better picked up at a later stage.**

I believe Dorde's pipeline stops at validation since there is no sequestering of test data. The logical follow-up here is, as per Leo's "Analysis" in GitLab, to use this to get the hyperparameters for the dataset for training on the entirety of the training data. Dorde's report also includes metric estimates for classifier performance which can inform the decision of model.

Maybe make a separate chunk for defining the method, cv_times, training fraction, kfold and specify default values

Fix the training fraction: 0.98 (r training_frac)

For SSVM you don't need any graphics; don't bother installing Cairo to do draw graphics.

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
