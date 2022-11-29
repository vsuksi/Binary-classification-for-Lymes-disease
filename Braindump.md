# Braindump

**The braindump is a repository for the fleeting thoughts that further the project at hand but are better picked up at a later stage.**

Make libraries go in alphabetical order?

Maybe make a separate chunk for defining the method, cv_times, training fraction, kfold and specify default values

Fix the training fraction: 0.98 (r training_frac)

For SSVM you don't need any graphics; don't bother installing Cairo to do draw graphics.

What do you need to do in order to make the feature number plot?
- limit the number of features from wilcox; use the same functions as Dorde to specify the number of important features involved
- make a barplot in the same way as the feature importance?
-

4 clusters in analysis.rmd plot is for visualizing batch effect?

lyme_data.csv includes all features, you can select 10, 50 100 etc. from there according to

Dorde isn't actually even using wilcox for feature selection!

Rmember to use inline code in Quarto, for example `r 2 * pi`

winCairo.dll was, for an unknown reason, not included with a conda installation of R, and is needed to draw some of the graphics. It can be included by copying from a regular executable install of the same version of R.

If method=SSVM:
reticulate till SSVM.py

Libraries "mt" (Metabolomics Data Analysis Toolbox), "scater", "mia", "bluster", "biostrings" and "rmarkdown" along with respective dependencies was installed through R

Conver rmd to qmd? (Analysis.rmd to analysis.qmd?)

"For an unknown reason, the R Markdown package has to be installed separately through R."

Perhaps you could make a separate option in main.R, quarto::quarto_render("report_extensive.Rmd") where you include feature importances, limit_features and possibly CoDaCoRe?

Make limit_features so that it works with the data in report.rmd

This is the deal: make a Quarto file in the style of report.rmd. In the same folder, have data.R, funcs.R, main.R and misc.R.

What are the possible user inputs?

Use source function

Beware when installing R through conda; vanilla installs an old version; use conda forge instead.

How to Quarto help()? The web-page is not the best.

Also include what accuracy you got using the most important features,

When a code chunk is time-consuming to run, consider caching it by using cache=true. This solves a problem with freeze, namely that using freeze the whole project is frozen, or alternatively on "auto", if a change is made, the whole project is rendered. Using cache, you can make changes to the documents in a project and render it without re-executing all the code.

If the pipeline is done in Quarto, include instructions on how to Quarto "include" the feature importance. Eval=true?! "Navigate to code_chunks/get_feature_importance"

So, if someone is to run a code chunk

How does this relate to what I'm doing? If someone wants to rerender the whole project, including the cached project

One only needs the virtual environment to execute the code, so cache=true may well work
Text for explaining how to execute

Freeze is not going to work for computation

There seems to be a script for converting a .yml file to a .txt file if needed.

In Quarto, Use eval:false to include code blocks without executing them; write the result instead. Where is this applicable in the project at hand? Or can you simply run it once you are completely certain of it working?

Remember, even when using warnings:false and errors:false in the execution options of a code chunk, you can view errors and warnings in the terminal during render.
- get_feature_importance()

To install calcom: include the calcom directory in the same directory as the environment file and create a virtual environment from the .yml environment file. Or navigate to the file and do pip install -e calcom.

Probably best to include the SSVM in the report as #not run. Comment why it is not a usable option: The creators of the library noted that it should be used for validation only, as development was halted and they don't have confidence in the package for real-world applications.

You can also make webpages with Quarto!

What does it mean when a code block is executed, where do I store the results?

Use the same test data as Leo in GitLab/pipeline-dorde/analysis/report.rmd?

So the point is to

Use  dataset.csv from gitlab with the three seed splits and test with CoDaCoRe

Run feature importance with the same three data splits as dorde?

Creating subdirectory for conda environment.
https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/02-working-with-environments/index.html

Choose the number of features as follows: include the important features as per get_feature_importance() and coda_core(), then

Running mikropml adds "." before the outcome variable

If you make a file called test or some other spurious name, it should be deleted immediately. Otherwise, name it according to your naming conventions. Moreover, name your test files with intent.

Try taking the feature importances from the model object using get_feature_importance()

You weren't getting comparable results due to having kfold=4 after testing with important features

Can you simply hold out the test set for the SSVM in order to get reproducible results?

Should I do this the feature limit thingamadingy for the most important features as per wilcox, permutation feature importance or random?

If you just want to check something quickly as a one off, use any pane and close it afterwards

Navigate the library with your arrows annd keys! How to stop it from scrolling though?

Which file in calcom was like "insert the number of cvfolds, load cc dataset..."?

Metric: _accuracy, roc_curve.py, _confusionmatrix.py

Have the to-do list open by default in the right pane?

Perhaps run the whole pipeline with different amounts of data?

Specify the lambda parameter?

Actually, you should run the scripts with the full(lyme) dataset and check if the test set is the same as dorde used afterward

You should be using the log loss feature importance for ranking feature importance

Heatmap of feature importance vs feature importance?

Make a heatmap of the top features and highlight the ones that were found to have high feature importance in the run_ml()

Feature importance heatmap and

How to collapse code rows in quarto presentation code?

Code fold = true

Try to pass the working .csv directly from r as per the reticulate website

You may have to install do a fresh environment so that you install everything as before except that for installing setup.py, install its dependencies

We don't need feature importances so perhaps take away feature id's from numpy array for SSVM?

Use the same seed for every classifier?

A random seed (or seed state, or just seed) is a number (or vector) used to initialize a pseudorandom number generator.

%>% can be read as "and then"

What is saveRDS?

Wrapper method in pipeline? Perhaps filter methods just are the proverbial shit for biomedical data?

https://www.ds-econ.com/quarto/

Good genera
Kehoe extensively used wrapper methods

Site discriminatory features were removed by IFR, where features were removed until the balanced success rate for classification of site fell under 60%.

KNN-imputation with log transform was used for IFR

Kehoe evaluated 18 different combinations of transformation and imputation schemes. The chosen ones:
KNN imputation with log transformation
Median imputation with log transformation
KNN imputation with median-fold change normalization
KNN imputation with standardization

Note which kind of batch effect; site or measurement.

Features selected to Skyline were at least once in the top 10 of features in kW folds of kffs, features from all transformation/imputation-schemes were included.

Only use the ID, I have no clue what the "order" stands for.

UMAP was used to evaluate the feature selection by kFFS using different imputation methods.

"The accuracy of each SSVM model was assessed by fivefold cross-validation (Table 1), and revealed an accuracy of greater than 92%, regardless of the transformation/imputation scheme"

Perhaps there is a good reason to use those with best

IFR was used to remove batch-discriminatory features

The two files what Dorde used are now on the DeskTop; the LCMS data is in  patient order without regard to the batch. When Dorde matched the old lcms data to the clinical data, the result is a dataset which uses only from one, the old, batch!

Kehoe, on the other hand, built the model on the old data but based on features present in both old and new data. The model was tested on a new batch in order to estimate batch effect.

The clinical data file simply contains clinical data on the patients

How can I figure out what exact file is

What would it take to run the pipeline on the newnew data?

Hope that the newnew data can be run through Dorde's scripts.

Leo did write something about old and new data.

Combine the training and test data into one file so that you can use training_frac and specify the range used to train? In that way, you could use Dorde's code directly methinks.

Dorde: "Logistics model with 3 random data splits" means training the model 3 times on random 75% of training data and cross validated on the 75% 100 times before testing on the remaining 25%.

Random baseline by shuffling.

The results are not comparable between Kehoe and Dorde because there is no batch effect in Dorde's approach as the test set came from the same batch.

Intensity is the relative abundance while the mass/charge ratio and retention time describes the identity of the feature.

Was the Kehoe test set from a different batch altogether? Dorde used a test set from the same batch. Perhaps I could do that?

The old/new data has to do with the reality of batch effects, perhaps?

LC separates metabolites by their retention time, MS by mass

There is a distribution function in Wilcox which allows for the calculation of p-value

The Dorde identifiers are ID's. Using them you can check whether the most important features have been validated by Kehoe, and overall do a comparison between the most important features across Dorde and Kehoe.

This demonstrated that KNN imputation with log transformation on training sam-
ples provided the highest mean fivefold cross-validation accuracy (99.8%, 0.3%) when an SSVM classifier was applied (Kehoe et al., Dorde also did log transformation)

Wilcoxon normalizes the data? Nope, the preprocessing function does!

Mann-Whitney tests features against the dependent variable,

Both Dorde and Kehoe used UMAP

Wilcoxon feature selection is an example of a filter method, while Kehoe's method is a wrapper method.

Wilcoxon basically ranks the features with respect to their relationship to the dependent variable. This

Positive to both Kehoe's statistical inference and Dorde's method is that they preserve the original semantics of the variables, unlike PCA, for example.

Is it problematic that Kehoe did no correlation analysis of features?

Specifically, 2198, 206, 682 and 147 were identified to contribute to batch effects from the log/knn, median-fold change/knn, standard/knn and raw/knn methods, respectively. After removal of these features, the batch effect disappeared for the healthy control samples. These features did, however, contribute to the the separation of disease state (EDL vs ELL)

Perhaps check the file which was fed into skyline for ID's?
- order vs ID?


SSVM chose the features in Kehoe et al.,

What are the Kehoe feature identifiers?

It should be possible to make a occurrence ranking in Kehoe style

Feature importances from Dorde are already ranked across the whole dataset since he used cv?

Perhaps feature importances are different across models?

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
