library(magrittr)
library(dplyr)
library(mikropml)
library(data.table)

dfbinary <- read.csv(file = "train_data.csv")
dfbinary$.outcome <- as.factor(dfbinary$.outcome)

result_decisiontree <- run_ml(dfbinary, 'rpart2', outcome_colname = '.outcome', kfold = 5, cv_times = 100, training_frac = 0.8, seed = 2019)

write.csv(result_decisiontree$performance%>%select(Accuracy), "decisiontree_accuracy.csv", row.names=FALSE)

write.csv(result_glmnet_random)
