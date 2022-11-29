library(magrittr)
library(dplyr)
library(glmnet)
library(data.table)
library(ipflasso)

# Read in data.
dftrain <- read.csv(file = "train_data.csv")

# Select data for training; don't
ytrain <- dftrain[-1, 1]
xtrain <- dftrain[-1, -1]

xtrain <- as.matrix(xtrain)
ytrain <- as.matrix(ytrain)

ytrain <- as.factor(ytrain)
grid = seq(100, 1, length=1000)
# standardize = FALSE?
# repeat this pipeline 100 times for each feature amount and take the average?
glmnet_fit <- cv.glmnet(xtrain, ytrain, type.measure="auc", family="binomial", alpha=0, dfmax = 1, pmax = 4851, nfolds=5, nlambda=100)
#specify lamda as well to get rid of the error?
#dftest <-  read.csv(file = "test_data.csv")

#ytest <- dftest[-1, 1]
#xtest <- dftest[-1, -1]

#xtest <- as.matrix(xtrain)
#ytest <- as.matrix(ytrain)

#ytest <- as.factor(ytest)

predict(glmnet_fit$glmnet.fit, newx = xtest)
glmnet_assess <- assess.glmnet(glmnet_fit, xtest, ytest) #s=0.04, from object
write.csv(glmnet_assess, "assesment.csv")

#write.csv(glmnet_fit$coeff, "glmnet_coeff")

#Loading required package: survival
#Warning message:
#NAs introduced by coercion
#Error in y - predmat : non-numeric argument to binary operator
#Calls: assess.glmnet -> do.call -> cv.elnet
#Execution halted
#Try with a regular cv.glmnet object in order to rule out a problem in the extension?
#Try data.matrix() to make dummies of the categorical variables
