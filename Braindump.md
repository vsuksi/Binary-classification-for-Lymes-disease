# Braindump

**The braindump is a repository for the fleeting thoughts that further the project at hand but are better picked up at a later stage.**

change "parameter" to "hyperparameter" wherever it comes up in regards to the regularization HYPERPARAMETER

###############################################################################↓
# Does it make sense to emulate the run_ml()-modeling for the SSVM and try to select an optimal value of the regularization parameter? Is there a rational way to select a model from the k-fold cross-validation scheme when using a value of the regularization parameter chosen from the graph?

Shouldn't the model should be trained on the whole of the training data at this point! Yes, I agree wholeheartedly! Fuck dat cross-validation. This also eliminates the need to take averages anywhere!

It does not make sense, since it is in the nature of SSVM that the classification accuracy starts to plateau at some value of the regularization hyperparameter. There is no rational way to select a "best" model from the cross-validation scheme a value of the regularization hyperparameter since

If we were to do this similarly to mikropml, each value of C would be cross-validated five-fold 100 times and the one with the highest average cv_auroc score would be chosen. Why not do it like this?
= because we don't want to choose a score higher than necessary: it is in the nature of SSVM that the score starts to plateau. So why test for further values of C?

Perhaps check how it is partitioned?

Final question is: how do we choose the final model then, since mikropml used cross-validation to choose the optimal lambda value? Well,

But how to save the best model to predict the test data? Well, the best model on validation data doesn't tell us much about the performance on test data. So there is no way we can select the best model, and we of course we can't select the model based on which one classifies the test data best since we would be overfitting to the test data.

Make cv_times=100 in order to match the methodology used by Dorde?

Q Does it even make sense to train the model
The model got different classification accuracies depending on which model it selected. We want the average across runs, right?
Perhaps it would be better to select the model based on AUROC?
Is there a repeat option available for the, that is the question.
the one with the best lambda value 100 times repeat cv AUROC was used to selec
Perhaps set a seed?
Does experiment.best_classifiers choose the model that performed best out of the five used in k-fold feature selection? Could all these models be saved and tested several times on random partitions of the training data to get a sense of which one is the best?
Is this how cv times functions in mikropml? The partition of test data is so small that it is a bit to choose a model based on that.
As of now, it generates a model for each

What is the fold size for the Calcom experiment?

So I've been using a model which works best on a random partition (training data) which perform best on a random partition (validation data). Could I cross-validate the models on further partitions to validate it further?
= No, since you used that data for training the model.

If no, what would it mean to take the average accuracy across 10 runs of training a model and trying it on the test data? It's the same as mikropml!


Check how much the SSVM graph and especially accuracy varies between runs! Does the variation justify the additional code? Better make an average of a few runs for the final prediction, no? 0.895, 0.937, 0.937, 0.895 0.906
looks rather unacceptable to me. We may indeed end up looping da shit out of it.

T make the latter part (prediction on test data) of SSVM.py loop 10 times, this should give a good estimate. That solution can then be transferred to the regularization parameter graph part

How do I use the reticulate package to following effect? C = regularization parameter in the python chunk.
Regularization parameter value: `r py$classifier.params['C']`

###############################################################################↓
# How should I incorporate the SSVM model along with the graph relating the value of the regularization parameter to classification accuracy, the graph relating the amount of features to classification accuracy, and feature importance? In what order?

Have a separate, extended_report.qmd for the extended trinity. The extended file is a direct copy of the regular one except for the three added parts. The run time will be rather long, but it is more user friendly to have a single extended report as compared to separate reports. Make separate, single chunks for the trinity. Include them at the end, in the following order:

1. Model feature importance
2. Graph relating the amount of features to classification accuracy
3. SSVM model along with the graph relating the value of the regularization parameter to classification accuracy

In Quarto document:

Try to pass the working .csv directly from r as per the reticulate website

"Regularization parameter value: `r C`

"The regularization parameter may be adjusted in the "analysis.qmd" if the chosen regularization parameter value does not point to the high-accuracy plateau in the graph".
###############################################################################↑

D Since the partitioning of folds in cross-validation may impact the accuracy score, the average across five runs might be needed. Kehoe did not take this into account though, and is probably the reason why their graph is rather wonky.
Make a list of lists, where the lists include the accuracy from separate runs which are averaged.

#Fuck it, they Kehoe's regularization parameter plot doesn't tell what is in question. Thus, we could either make a plot, which would make it include variable user input. This is not cool for the report since it is not possible to have variable user input in a Quarto document. Thus, better choose a value of C that works for the data processing at hand, that is log10 transformation. Try generating a graph and choose a value of C slightly higher than necessary to account for the possibility variability in optimal C between datasets. The fact that the data is from two batches helps a bit. You could also check the value for the regularization parameter by plotting the new data that Leo has.

#Although k-fold feature selection was used by Kehoe, the regularization parameter lambda was used on all 4851 features. Moreover, Kehoe also used log ranking.
#Check the plot for the regularization parameter value. If the regularization parameter value is sub-optimal, it can be changed

#Choose the smallest value for the regularization hyperparameter C which doesn't see a decline.
#But should I repeat the cross-validation 100 times the SSVM classifier? I think so.

#Fix the training fraction: 0.98 (r training_frac)

D What do you need to do in order to make the feature number plot?
- limit the number of features from wilcox; use the same functions as Dorde to specify the number of important features involved
- make a barplot in the same way as the feature importance?
- lyme_data.csv includes all features, you can select 10, 50 100 from there


#4 clusters in analysis.rmd plot is for visualizing batch effect?

#Metric: _accuracy, roc_curve.py, _confusionmatrix.py

The Dorde identifiers are ID's. Using them you can check whether the most important features have been validated by Kehoe, and overall do a comparison between the most important features across Dorde and Kehoe.

10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10,000 piirrettä in glmnet.

- is it soft margin or hard margin ssvm? Check the code.
can't access the information since I don't have the calcom library.
