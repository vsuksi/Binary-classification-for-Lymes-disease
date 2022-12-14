library(magrittr)
library(dplyr)
library(mikropml)
library(data.table)

# Read in data.
dfbinary <- read.csv(file = "train_data.csv")
dfbinary$.outcome <- as.factor(dfbinary$.outcome)

result_randomforest <- run_ml(dfbinary, 'svmRadial', outcome_colname = '.outcome', kfold = 5, cv_times = 100, training_frac = 0.8, seed = 2019)
