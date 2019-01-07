
# Data Import -------------------------------------------------------------

# set working directory
setwd("C:\\Users\\samee\\Documents\\Emory\\Fall 2018 Courses\\Machine Learning_ISOM674\\FINAL_PROJECT")

# install and/or load libraries
if(!require("data.table")) { install.packages("data.table"); require("data.table") } # for primary data manipulation
if(!require("randomForest")) { install.packages("randomForest"); require("randomForest") } # for bagging and random forests

# read in cleaned training, validation 1 and validation 2 data files
load("train_clean.RData")
load("val1_clean.RData")
load("val2_clean.RData")

# get y-values to evaluate
actual_val1_y <- val1_clean$click
actual_val2_y <- val2_clean$click

# ensure click i.e. y is a factor in all files
train_clean[, click := as.factor(click)]
val1_clean[, click := as.factor(click)]
val2_clean[, click := as.factor(click)]

# Log-Loss Function -------------------------------------------------------

# function to calculate log loss, adapted from https://rdrr.io/cran/MLmetrics/src/R/Classification.R
log_loss_fn <- function(predicted, actual, eps=1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  log_loss <- -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
  return(log_loss)
}

# Find best nodesize for bagging -------------------------------------------------------
# first, find the best node-size by looking for minimum performance on the validation1 data

# set up formula for bagging
vars <- colnames(train_clean) # get column names
fm_bagging <- paste(vars[2],"~",paste(vars[3:23],collapse=" + "),sep=" ") # create string of formula with x-variables
fm_bagging <- formula(fm_bagging) # convert string to a formual

# initialize results table for bagging that stores the varying nodesize arguments and the eventual log-loss performance
results_bagging <- data.table( "node_size"=c(1,5,10,30,50,100,250,500,1000,5000), "log_loss"= rep(99999, 10))

# run bagging with varying minimum sizes of terminal nodes
for (i in 1:nrow(results_bagging)) {
  # fit 100 trees with specified terminal node size, full set of features sampled
  param_bagging <- randomForest(fm_bagging, data=train_clean, 
                                nodesize=results_bagging[i, node_size],
                                mtry=length(vars[3:23]), ntree=50)

  # get ensemble predictions on val1 data
  y_pred_bagging <- predict(param_bagging, newdata=val1_clean[, 3:23], type="prob") # get class probabilities
  y_pred_bagging <- y_pred_bagging[,2] # only get probabilities for prediction of 1 i.e. click outcome
  
  # compute and store log loss in results table
  results_bagging[i, log_loss := log_loss_fn(y_pred_bagging, actual_val1_y)]
}

best_nodesize_bagging <- results_bagging[which.min(log_loss), node_size]

# Fit bagging model on full training ----------------------------------------------------
# use the best parameter value from val1, build a new model that combines the training and initial validation data

# get full training and validation data to fit final bagging model
train_full <- rbind(train_clean, val1_clean)

# fit bagging model on full training data
model_bagging <- randomForest(fm_bagging, data=train_full,
                              nodesize=best_nodesize_bagging,
                              mtry=length(vars[3:23]), ntree=50)

# get bagging prediction on validation data
pred_val_bagging <- predict(model_bagging, newdata=val2_clean[, 3:23], type="prob") # get class probabilities
pred_val_bagging <- pred_val_bagging[,2] # only get probabilities for prediction of 1 i.e. click outcome

# get and store log loss performance on validation data
lloss_val_bagging <- log_loss_fn(pred_val_bagging, actual_val2_y)