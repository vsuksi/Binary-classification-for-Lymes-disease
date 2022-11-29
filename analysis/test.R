knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(mikropml)
library(data.table)
library(umap)
library(ggfortify)
library(VIM)
library(mt)
library(cluster)
library(kableExtra)
library(rmdformats)

library(ggplot2)
library(ggplot2)
library(scater)
library(mia)
library(dplyr)
# Load packages
library(TreeSummarizedExperiment)

source("funcs.R")
set.seed(142)

# Read in the data (assuming standardized format!)
source("data.R")

# Split the full data into training and test sets
# The modeling functions will never see the test set

## Training data
train <- sample(ncol(tse), round(0.8*ncol(tse)))
tse.train <- tse[, train]

## Independent test data
test <- setdiff(seq(ncol(tse)), train)
tse.test  <- tse[, test]

# Add log10 transformation and scaling for the signal assay
pseudocount <- 0
if (min(assay(tse, "signal"))==0) {
  x <- assay(tse, "signal")
  pseudocount <- min(x[x>0])/2
}
tse <- mia::transformSamples(tse, abund_values="signal", method="log10", pseudocount=pseudocount)
tse <- mia::transformFeatures(tse, abund_values="log10",  method="z")

# Run UMAP
tse <- scater::runUMAP(tse, name = "UMAP", exprs_values = "z")

p.umap <- plotReducedDim(tse, "UMAP", colour_by = "Group") + labs(title = "UMAP")

tse <- scater::runPCA(tse, name = "PCA", exprs_values = "z")

p.pca <- plotReducedDim(tse, "PCA", colour_by = "Group") + labs(title = "PCA")

library(patchwork)
print(p.umap + p.pca)

# Define for next chunk
nclust <- 4

library(bluster)
cl <- clusterRows(t(assay(tse, "z")), HclustParam(k=nclust, method="ward.D2"))
plotUMAP(tse, colour_by=I(cl))

method <- "glmnet"
cv_times <- 2
training_frac = 0.98 # Set to this since 1 is not accepted; train with (almost) full training data
kfold <- 5
outcome_variable <- "Group"

library(mikropml)
dataset <- as.data.frame(t(assay(tse.train, "signal")))
dataset$Group=colData(tse.train)$Group
res <- run_ml(dataset,
              method,
              outcome_colname = outcome_variable,
              kfold = kfold,
              cv_times = cv_times,
              training_frac = training_frac,
              seed = 2019)

training_data <- res$trained_model$trainingData %>%
  dplyr::rename(dx = .outcome)
test_data <- res$test_data

hyperparameters <- get_hyperparams_list(dataset, method)

cross_val <- define_cv(training_data,
                       "dx",
                       hyperparameters,
                       perf_metric_function = caret::multiClassSummary,
                       class_probs = TRUE,
                       cv_times = 2)

tune_grid <- get_tuning_grid(hyperparameters, method)

model_formula <- stats::as.formula(paste("dx", "~ ."))
      model <- caret::train(
        form = model_formula,
        data = training_data,
        method = method,
        metric = "AUC",
        trControl = cross_val,
        tuneGrid = tune_grid)

        test_data <- as.data.frame(t(assay(tse.test, "signal")))
        test_data$Group=colData(tse.test)$Group
        pred <- predict(model, newdata = test_data)
        pred <- data.frame(
                  Sample = colData(tse.test)$Sample,
                  Original = colData(tse.test)$Group,
                  Prediction=pred)

        prob <- predict(model, newdata = test_data, type="prob")
        pred <- cbind(pred, prob)

        pred %>% kable(digits=2)

        predicted <- as.factor(pred$Prediction)
        original  <- as.factor(pred$Original)
        confusionMatrix(predicted, original)
