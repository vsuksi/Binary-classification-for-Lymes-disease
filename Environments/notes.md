Libraries "mt" (Metabolomics Data Analysis Toolbox), "scater", "mia", "bluster", "biostrings" and "rmarkdown" along with respective dependencies was installed through R

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("")

winCairo.dll was, for an unknown reason, not included with a conda installation of R, and is needed to draw some of the graphics. It can be included by copying from a regular executable install of the same version of R.
