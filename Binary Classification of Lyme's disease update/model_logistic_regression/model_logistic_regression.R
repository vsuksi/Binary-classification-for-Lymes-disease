library(magrittr)
library(dplyr)
library(mikropml)
library(data.table)

# Read in data.
df <- read.csv(file = "lyme_data.csv")

# Shuffle the outcome column if modeling for baseline accuracy.
#df <- transform(df, outcome = sample(outcome))

# Optionally use only select, important features
# df <- df %>% select("outcome", "V972", "V259", "V3167", "V933")

# Run the mikropml pipeline.
model_logistic_regression <- run_ml(df, 'glmnet', outcome_colname = 'outcome', kfold = 5, cv_times = 100, training_frac = 0.80, seed = 2020)

# Write metrics to file.
write.csv(model_logistic_regression$performance%>%select(Accuracy), "logistic_regression_results.csv", row.names=FALSE)
