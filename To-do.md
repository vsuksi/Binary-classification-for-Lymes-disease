
# To-do

**Only list specific and actionable things, for example: pick up your package at R-kioski! The slow burn list is not about getting shit done, but more about remembering the things which I want to get done at some point.**

**Enter tasks in a rough order of importance!**

Run the SSVM classifier overnight on repeat and modify the code so that the graph presents the average accuracy for the regularization parameter C across the repeated runs. At least run it a few times and compare the graphs (jot down this met)

Do the rest of the SSVM pipeline, what I mean the part where you use the regularization parameter 0.1 to train (or choose from list?) and predict the classes for the new data.

{r, child=if (!SSVM) 'appendix.Rmd'} add this back to report.rmd when it works otherwise

Rename the GitHub repository accordingly: commit and push, then delete it locally and clone from GitHub after renaming it there

Run feature importance with the same three data splits as Dorde

Check how to plot number of features vs accuracy, choose which method/library to use (same as Leo in analysis?)

Make get_feature_importance() compatible with whatever method the user chooses.

Make a project folder make scripts for the individual code chunks in a "code chunks" -folder

Test rendering the cached test_project.qmd with environment lyme_main

Try using the parallel processing one more time for get_feature_importance() in spite of it not working last time

Run feature importance with folds 2020-2022

Try to specify the train-test split for SSVM or try to use the trained model to predict samples from training data

Do baseline for CoDaCoRe

Make a random baseline for the Python SSVM, overnight! Also use different baseline scores?

Train the logistic regression model using the same seed (2020, 2021 or 2022) and train-test split (0.75) to see if you get the same results as Dorde.


## Slow burn
- compulsory PowerPoint presentation
-
