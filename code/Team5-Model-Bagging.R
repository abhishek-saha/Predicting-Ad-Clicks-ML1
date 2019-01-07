
# Data Import -------------------------------------------------

# load libraries
library("data.table") # for data manipulation
library("randomForest") # for bagging

# read in all training data
load("train_split1.RData")
load("train_split2.RData")
load("train_split3.RData")
load("train_split4.RData")
load("train_split5.RData")
load("train_split6.RData")
load("train_split7.RData")
load("train_split8.RData")
load("train_split9.RData")
load("train_split10.RData")

# read in validation data
load("val_clean.RData")

# store validation y-values for evaluation 
actual_val_y <- val_clean$click # store as integer for computing log-loss
val_clean[, click := as.factor(click)] # store as factor for passing into algorithm

# Functions for Model-Building and Prediction -------------------------------------------------------------

# get universal formula for bagging
vars <- colnames(train_split1) # get column names
fm_bagging <- paste(vars[2],"~",paste(vars[3:23],collapse=" + "),sep=" ") # create string of formula with x-variables
fm_bagging <- formula(fm_bagging) # convert string to a formula

# function that builds and returns bagging model for one data file
get_model_bagging <- function(chunk){
  chunk_bagging <- randomForest(fm_bagging, data=chunk, mtry=21, ntree=100) # build bagging model
  return(chunk_bagging) # return model
}

# function that gets bagging predictions given a model and new data
get_pred_bagging <- function(model, new_data){
  pred_bagging <- predict(model, newdata=new_data, type="prob") # get class probabilities on validation
  pred_bagging <- pred_bagging[,2] # only get probabilities for prediction of 1 i.e. click outcome
  return(pred_bagging)
}

# Get Model and Predictions For Each Chunk -------------------------------------------------------

# get bagging results for each chunk
# ran this manually instead of in a loop to examine and store each model one at a time

# Data1: get model and predictions
model_bagging_1 <- get_model_bagging(train_split1) # get model
pred_val_bag1 <- get_pred_bagging(model_bagging_1, val_clean[,3:23]) # get predictions
results_val_bagging <- data.table("Pred_Val_Bag1"=pred_val_bag1) # store in data table

# Data2: get model and predictions
model_bagging_2 <- get_model_bagging(train_split2) # get model
pred_val_bag2 <- get_pred_bagging(model_bagging_2, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag2 := pred_val_bag2] # store in data table

# Data3: get model and predictions
model_bagging_3 <- get_model_bagging(train_split3)
pred_val_bag3 <- get_pred_bagging(model_bagging_3, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag3 := pred_val_bag3] # store in data table

# Data4: get model and predictions
model_bagging_4 <- get_model_bagging(train_split4) 
pred_val_bag4 <- get_pred_bagging(model_bagging_4, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag4 := pred_val_bag4] # store in data table

# Data5: get model and predictions
model_bagging_5 <- get_model_bagging(train_split5) 
pred_val_bag5 <- get_pred_bagging(model_bagging_5, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag5 := pred_val_bag5] # store in data table

# Data6: get model and predictions
model_bagging_6 <- get_model_bagging(train_split6) 
pred_val_bag6 <- get_pred_bagging(model_bagging_6, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag6 := pred_val_bag6] # store in data table

# Data7: get model and predictions
model_bagging_7 <- get_model_bagging(train_split7) 
pred_val_bag7 <- get_pred_bagging(model_bagging_7, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag7 := pred_val_bag7] # store in data table

# Data8: get model and predictions
model_bagging_8 <- get_model_bagging(train_split8) 
pred_val_bag8 <- get_pred_bagging(model_bagging_8, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag8 := pred_val_bag8] # store in data table

# Data9: get model and predictions
model_bagging_9 <- get_model_bagging(train_split9) 
pred_val_bag9 <- get_pred_bagging(model_bagging_9, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag9 := pred_val_bag9] # store in data table

# Data10: get model and predictions
model_bagging_10 <- get_model_bagging(train_split10) 
pred_val_bag10 <- get_pred_bagging(model_bagging_10, val_clean[,3:23]) # get predictions
results_val_bagging[, Pred_Val_Bag10 := pred_val_bag10] # store in data table

# Get Log Loss On Validation -------------------------------------------------------

# function to calculate log loss, adapted from https://rdrr.io/cran/MLmetrics/src/R/Classification.R
log_loss_fn <- function(predicted, actual, eps=1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  log_loss <- -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
  return(log_loss)
}

# get average predictions across all 10 models (ensemble method)
pred_val_bag_avg <- rowMeans(results_val_bagging)

# get and store log loss performance on validation data
lloss_val_bagging <- log_loss_fn(pred_val_bag_avg, actual_val_y)
