library(magrittr)
library(dplyr)
library(mikropml)
library(data.table)

# Read in data.
df <- read.csv(file = "train_data.csv")

# Run the mikropml pipeline.
result_random_forest <- run_ml(df, 'rf', outcome_colname = '.outcome', kfold = 5, cv_times = 100, training_frac = 0.8, seed = 2020)

# Write metrics to file.
write.csv(result_random_forest$performance%>%select(Accuracy), "random_forest_results.csv", row.names=FALSE)
