library(magrittr)
library(dplyr)
library(mikropml)
library(data.table)

dfbinary <- read.csv(file = "train_data.csv")
dff <- transpose(dfbinary)
dfbinary$.outcome <- as.factor(dfbinary$.outcome)

write.csv(dff, "transposed_train_data.csv")


#write.table(dfbinary, file="dfbinary.txt", )

#random_glmnet <- transform(dfbinary, outcome = sample(.outcome))

#result_glmnet_random <- run_ml(random_glmnet, 'glmnet', outcome_colname = #'outcome', kfold = 5, cv_times = 100, training_frac = 0.8, seed = 2019)

#write.csv(result_glmnet_random$test_data, "baseline_results.csv", #row.names=FALSE)
