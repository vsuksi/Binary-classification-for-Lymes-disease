library(magrittr)
library(dplyr)
library(mikropml)
library(data.table)

dfbinary <- read.csv(file = "train_data.csv")
dfbinary$.outcome <- as.factor(dfbinary$.outcome)

random_glmnet <- transform(dfbinary, outcome = sample(.outcome))

result_glmnet_random <- run_ml(random_glmnet, 'glmnet', outcome_colname = 'outcome', kfold = 5, cv_times = 100, training_frac = 0.8, seed = 2019)

write.csv(result_glmnet_random$performance%>%select(Accuracy), "logistic_baseline_accuracy.csv", row.names=FALSE)
