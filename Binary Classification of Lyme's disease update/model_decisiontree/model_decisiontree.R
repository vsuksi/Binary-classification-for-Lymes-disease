library(magrittr)
library(dplyr)
library(mikropml)
library(data.table)

# Read in data.
df <- read.csv(file = "lyme_data.csv")

# Shuffle the outcome column if modeling for baseline accuracy.
#df <- transform(df, outcome = sample(outcome))

model_decision_tree <- run_ml(df, 'rpart2', outcome_colname = 'outcome', kfold = 5, cv_times = 100, training_frac = 0.8, seed = 2019)

write.csv(result_decisiontree$performance%>%select(Accuracy), "decision_tree_accuracy.csv", row.names=FALSE)
