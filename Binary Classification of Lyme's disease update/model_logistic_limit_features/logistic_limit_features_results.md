# Results
### Checking whether the results are the same between runs.
"train_data.csv"
"glmnet" logistic regression
type.measure = "auc"
family = "binomial"
alpha = 0
dfmax = 100
ncv = 10
The runs are indeed the same. Thus, we need to set different seeds for each run.
s = 0.04 confirmed
First run:
$auc
[1] 0.9800454
attr(,"measure")
[1] "AUC"

Second run:
$auc
[1] 0.9800454
attr(,"measure")
[1] "AUC"
