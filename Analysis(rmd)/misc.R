  # KNN imputation
  # dataset<- kNN(dataset, k = 5)
  # dataset <- knn_res[grep("_imp", colnames(knn_res), invert = TRUE)] #deleting unnecessary produced columns _imp
    
# Running the pipeline with glmnet method

We run the pipeline and show some outputs.

```{r, message=FALSE, warning=FALSE}
result <- pipeline(df, "outcome","glmnet", 2)
```

  # features selection using Wilcox test
  x <- dataset %>%
    select(-c(outcome_variable))
  y <- dataset[,outcome_variable]
  # log transform
  x <- preproc(x, method="log10")
  
  wilcox <- fs.wilcox(x, y)
  wilcox_order <- wilcox$fs.order


Model performance

```{r}
library(knitr)
res$performance %>% kable(digits=2)
```