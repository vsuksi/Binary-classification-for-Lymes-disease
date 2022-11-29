# Binary classification of Lyme's disease using metabolic profiling

## Overview

Turku Data Science Group (TDSG) did an R implementation of work originally done in Python by E. Kehoe et al. (2022). The results of the two implementations are, however, not comparable because of differences in the available data cleaning methods across Python and R. Thus, I will do a Python implementation using the dataset TDSG which has been cleaned in R, so that the results can be compared. Results will be written to a .csv file. When the Python implementation is working as planned, the pipeline will be compiled using Quarto.

At the end of the presentation, compare the accuracy of the SSVM prediction with the original data preprocessing and Dorde's classifiers.


Important links:
Metabolomics data analysis toolbox:
- log10 normalization
https://rdrr.io/github/wanchanglin/mt/man/preproc.html

Mikropml
- run_ml
https://rdrr.io/github/SchlossLab/mikropml/man/run_ml.html

Wilcoxon rank sum test:
https://peterstatistics.com/CrashCourse/3-TwoVarUnpair/BinOrd/BinOrd-2a-MannWhitneyUtest.html
### Methods

#### Experimental methods

Liquid chromatography-mass spectrometry (LCMS) is a technique used to

#### Data science methods

Sparse support vector machines
The data

### Metabolite identification level descriptions
Remember that this is not an order of operations, but a tier list of identification confidence.

##### Level 5: Unique feature

After the data has been narrowed down to a prioritized list of peaks by untargeted analyses, unique features are identified by their measurement accuracy.

##### Level 4: Molecular Formula

The molecular formula is established from isotope abundance distribution, charge state and adduct ion determination.

##### Level 3: Tentative structure

Databases are screened for tentative structures by matching the parent ion data to structure data in databases and the literature.

##### Level 2: Putative identification

Fragmentation data from literature and databases reveals a probable structure.

##### Level 1: Validated identification

Exact identification by reference standard.
