# Braindump

**The braindump is a repository for the fleeting thoughts that further the project at hand but are better picked up at a later stage.**

Can I simply feed the pipeline in a similar manner to 'below' in order run the pipeline with Leo's suggested amounts of features?
disp _ features (result ,1:45)

Check the wilcoxon rank-sum test implementation and look for clues on how to limit the number of features, perhaps there is a max somewhere?
-Wilcox from Metabolomics Data analysis toolbox

Features were selected by the Wilcox test, so you ought to be able to increase the number of features that way, somehow!

You can advance by serching for "wilcox"

You can advance by searching for "dataset" in Dorde's code.

The dataset is fed into the pipeline (very last thing in appendices) which allows you to

How do I choose the best features for Leo's proposal? Come up with a few suggestions and discuss them with Leo.

In what contexs is the word grid used in data science.



How to pick the feature weights (metabolite importance) from the glmnet?
-

How to make a plot with the number of features?
- Number of features on x-axis, accuracy on y-axis

The two healthy control groups come from different sites, and had somewhat different features. The union of the features was of greater interest, since this work was done for diagnostic purposes.

Check if the features selected by Kehoe (targeted) and Dorde (untargeted) are similar, but how?

In the Quarto presentation, do a comparison of the data cleaning methods across the two approaches. State that they are quite different indeed and present some statistics on how they differ? The differences in data cleaning methods made the results uncomparable. Now, let me present the results as classified by Kehoe's SSVM and Dorde's classifiers on the same dataset, as cleaned by Dorde. and  Present SSVM from Kehoe along with Dordes methods.

10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10,000 piirrettä in glmnet.

Appendix A has nothing to do with Lyme's classification.

Iterative feature removal removes features between healthy control groups so that they have the same features?

What stage is the dataset in which Dorde is referring to? Pretty raw, methinks.

How to have a system explaining, in detail, what the code is doing?

Vibe a jämpti presentation as follows: intent/design, pipeline, comparison, limitations, conclusion

Make the Quarto presentation so that you have specific things in the code that you point out as important.

Test running Dorde's R code in Quarto.
Explain the classifiers in the presentation

Combining feature sets?

XCMS

Will I need iterative_feature_removal at all?

- presentation: simply run the pipeline in Quarto!

- borsta mellan wc kakel med tandborste!

- is it soft margin or hard marginn ssvm? Check the code.
can't access the information since I don't have the calcom library.

- how is Dordes pipeline automated? Because of the hyperparameter tuning done as informed by ROC?

- calcom vs calcom.io?

- is ROC or AUROC mentioned in Kehoe?

- pip install in conda environment? Remember, different packages are available in pip and conda.

- why is data science so interesting?

- the code was indeed in Dordes thesis! Always check the code for answers which Leo may or may not have!

- is variable hyperparameter tuning input usually recorded somewhere for reproducing the results?

-train-validation split same as in Dorda?

- Dordes email or contact on linked in

Are the classifiers that Dorde used detailed in the publication? Or do you need access to the source code?

You might be able to tease apart the hyperparameter tuning steps from the publication!

Classifiers used from publication text directly?
Can I get the R code from Djordjes work directly?
Targeted features vs features

Reproducibility vs replicability:
Reproducibility is the extent to which the exact workflow of a study can be followed to produce identical results. Replicability, on the other hand, focuses on whether a hypothesis can be confirmed across different experimental setups and datasets.

Use the date to access the correct versions of packages (can they be accessed that way?)? Is this a reasonable assumption or should I ask for the exact versions?

The scripts may well work without using exact same versions as in the publication but how would I know whether it's the variable user input or the scripts which cause results differing from those in the publication?
