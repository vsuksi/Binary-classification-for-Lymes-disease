library(mikropml)
library(dplyr)
library(doFuture)
library(foreach)
library(future)
library(future.apply)

doFuture :: registerDoFuture ()
future :: plan(future::multisession, workers = 2)

dfbinary <- read.csv(file = "lyme_data.csv")
#dfbinary$outcome <- as.factor(dfbinary$outcome)

#method <- glmnet(dmax=100, lambda)

result_glmnet <- run_ml(dfbinary, 'glmnet', outcome_colname = 'outcome', kfold = 5, cv_times = 100, training_frac = 0.8, seed = 2019, hyperparameters = list(alpha = 0) find_feature_importance = TRUE)
#removed seed=2019
#training_data <- result_glmnet$trained_model%>%select(training_data)
#get_feature_importance(result_glmnet$trained_model$, result_glmnet$trainingData, result_glme

a <- result_glmnet$feature_importance

write.csv(a, "feature_importance.csv")

write.csv(result_glmnet$test_data, "test_data.csv", row.names=FALSE)
#write.csv(a, "feature_importance.csv")
#1: UNRELIABLE VALUE: One of the foreach() iterations ('doFuture-1') #unexpectedly generated random numbers without declaring so. There is a risk #that those random numbers are not
# statistically sound and the overall results might be invalid. To fix this, #use '%dorng%' from the 'doRNG' package instead of '%dopar%'. This ensures that #proper, parallel-safe
#random numbers are produced via the L'Ecuyer-CMRG method. To disable this #check, set option 'doFuture.rng.onMisuse' to "ignore".
#2: UNRELIABLE VALUE: One of the foreach() iterations ('doFuture-2') #unexpectedly generated random numbers without declaring so. There is a risk #that those random numbers are not
# statistically sound and the overall results might be invalid. To fix this, #use '%dorng%' from the 'doRNG' package instead of '%dopar%'. This ensures that #proper, parallel-safe
#random numbers are produced via the L'Ecuyer-CMRG method. To disable this #check, set option 'doFuture.rng.onMisuse' to "ignore".
#Error in file(file, ifelse(append, "a", "w")) :
#  cannot open the connection
