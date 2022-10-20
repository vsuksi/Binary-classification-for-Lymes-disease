library(mikropml)
library(dplyr)
library(doFuture)
library(foreach)
library(future)
library(future.apply)

#doFuture :: registerDoFuture ()
#future :: plan(future::multisession, workers = 2)

dfbinary <- read.csv(file = "lyme_data.csv")

result_glmnet <- run_ml(dfbinary, 'glmnet', outcome_colname = 'outcome', kfold = 5, cv_times = 100, training_frac = 0.8, seed = 2019, hyperparameters = list(alpha = 0) find_feature_importance = TRUE)


feature_importance <- result_glmnet$feature_importance

write.csv(feature_importance, "feature_importance.csv")

#1: UNRELIABLE VALUE: One of the foreach() iterations ('doFuture-1') #unexpectedly generated random numbers without declaring so. There is a risk #that those random numbers are not
# statistically sound and the overall results might be invalid. To fix this, #use '%dorng%' from the 'doRNG' package instead of '%dopar%'. This ensures that #proper, parallel-safe
#random numbers are produced via the L'Ecuyer-CMRG method. To disable this #check, set option 'doFuture.rng.onMisuse' to "ignore".
#2: UNRELIABLE VALUE: One of the foreach() iterations ('doFuture-2') #unexpectedly generated random numbers without declaring so. There is a risk #that those random numbers are not
# statistically sound and the overall results might be invalid. To fix this, #use '%dorng%' from the 'doRNG' package instead of '%dopar%'. This ensures that #proper, parallel-safe
#random numbers are produced via the L'Ecuyer-CMRG method. To disable this #check, set option 'doFuture.rng.onMisuse' to "ignore".
#Error in file(file, ifelse(append, "a", "w")) :
#  cannot open the connection

# Data has a names column that has the feature/group of features name.
# Data has the auc_diff column that has real auc - permuted auc for each datasplit

perm_top10 <- data %>%
  group_by(names)%>%
  summarise(median = median(auc_diff), iqr_AUC = IQR(auc_diff), mean = mean(auc_diff), se = sd(auc_diff)/sqrt(n())) %>%
  mutate(sign = case_when(median > 0 ~ "positive", median < 0 ~ "negative")) %>%
  #  Arrange from highest median delta to  descending
  arrange(-median) %>%
  # Grab only the largest delta top 10
  head(n=10) %>%
  select(names, median, iqr_AUC, mean, se)

# ggplot2 bar plot
plot <- ggplot(perm_top10, aes(reorder(names, mean), mean)) +
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
