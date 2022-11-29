library(mikropml)
library(dplyr)
library(ggplot2)
library(magrittr)

# Read in data.
df <- read.csv(file = "lyme_data.csv")

# Run the mikropml pipeline.
model_logistic_regression <- run_ml(df, 'glmnet', outcome_colname = '.outcome', training_frac = 0.8, kfold=5, cv_times = 100, hyperparameters = list(alpha = 0, lambda = 0.04), seed=2019)

# Rename the outcome column for the resultant test data; for some reason, the training and test data from run_ml() have different column names.
names(model_logistic_regression$test_data)[1] <- ".outcome"

# Get permutation feature importance.
feature_importance <- get_feature_importance(model_logistic_regression$trained_model, model_logistic_regression$trained_model$trainingData, model_logistic_regression$test_data, outcome_colname = ".outcome", multiClassSummary, "AUC", class_probs=TRUE, method="glmnet")

# Write feature importances to file"
write.csv(feature_importance, "important_features", row.names=FALSE)

# Select top n features.
perm_top <- feature_importance %>%
  group_by(names)%>%
  summarise(median = median(perf_metric_diff), iqr_AUC = IQR(perf_metric_diff), mean = mean(perf_metric_diff), se = sd(perf_metric_diff)/sqrt(n())) %>%
  mutate(sign = case_when(median > 0 ~ "positive", median < 0 ~ "negative")) %>%
  arrange(-median) %>%
  head(n=10) %>%
  select(names, median, iqr_AUC, mean, se)

#ggplot2 bar plot of top features
plot <- ggplot(perm_top, aes(reorder(names, mean), mean)) +
 geom_bar(position = position_dodge(), width = .25, stat="identity", fill="steelblue")  +
 geom_hline(yintercept = 0, color = "black") +
        geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width=0.2) +
 theme(panel.grid.major = element_blank(),
       panel.grid.minor = element_blank()) +
 theme_bw() +
 theme(panel.grid.major = element_blank(),
       panel.grid.minor = element_blank()) +
 xlab("Features") +
 ylab('Mean difference between test and permuted AUROC') +
 coord_flip() +
 theme(axis.text.x = element_text(size = 10,  colour=c("black")),
       axis.text.y = element_text(size = 10, colour=c("black")),
       axis.title.x = element_text(size=12, vjust = 0),
       axis.title.y = element_text(size=12, vjust = 0.5),
       legend.text = element_text(size=13))

ggsave("feature_importance02112022.png", plot=plot)
