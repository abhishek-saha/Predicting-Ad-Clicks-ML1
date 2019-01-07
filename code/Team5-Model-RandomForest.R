
# Data Import-------------------------------------------------

# load libraries
library("data.table") # for data manipulation
library("randomForest") # for building forest

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

# read in validation and test data
load("val_clean.RData")
#load("test_clean.RData")

# store validation y-values for evaluation 
actual_val_y <- val_clean$click # store as integer for computing log-loss
val_clean[, click := as.factor(click)] # store as factor for passing into algorithm

# Functions for Model-Building and Prediction -------------------------------------------------------------

# get universal formula for random forest
vars <- colnames(train_split1) # get column names
fm_rf <- paste(vars[2],"~",paste(vars[3:23],collapse=" + "),sep=" ") # create string of formula with x-variables
fm_rf <- formula(fm_rf) # convert string to a formula

# function that creates random forest model for one data file and returns model
get_model_rf <- function(chunk){
  
  # originally tried using tuneRF to auto-optimize mtry, but hit size computation limits even with 2M-row chunks
  # moreover, trying on a smaller sample indiccated it was only recommending the default value, so stuck with original
  #chunk_rf <- tuneRF(chunk[,3:23], chunk$click, stepFactor = 1, 
  #                   doBest=TRUE, trace=FALSE, plot=FALSE)
  
  # build and return model
  ## originally completed full run with ntree=100, due to observed variance between models switched to ntree=500
  ## realized larger ntree sizes were leading to vector size errors, so ran final model with ntre=250
  chunk_rf <- randomForest(fm_rf, data=chunk, ntree=250) # build rf model
  return(chunk_rf) # return model
}

# function that returns random forest predictions given a model and new data
get_pred_rf <- function(model, new_data){
  pred_rf <- predict(model, newdata=new_data, type="prob") # get class probabilities on validation
  pred_rf <- pred_rf[,2] # only get probabilities for prediction of 1 i.e. click outcome
  return(pred_rf)
}

# Get Model and Predictions For Each Chunk -------------------------------------------------------

# get random forest results for each chunk
# ran this manually instead of in a loop to examine and store each model one at a time

# Data1: get model and predictions
model_rf_1 <- get_model_rf(train_split1) # get model
pred_val_rf1 <- get_pred_rf(model_rf_1, val_clean[,3:23]) # get predictions
results_val_rf <- data.table("Pred_Val_rf1"=pred_val_rf1) # store in data table

# Data2: get model and predictions
model_rf_2 <- get_model_rf(train_split2) # get model
pred_val_rf2 <- get_pred_rf(model_rf_2, val_clean[,3:23])
results_val_rf[, Pred_Val_rf2 := pred_val_rf2] # store in data table

# Data3: get model and predictions
model_rf_3 <- get_model_rf(train_split3)
pred_val_rf3 <- get_pred_rf(model_rf_3, val_clean[,3:23]) # get predictions
results_val_rf[, Pred_Val_rf3 := pred_val_rf3] # store in data table

# Data4: get model and predictions
model_rf_4 <- get_model_rf(train_split4) 
pred_val_rf4 <- get_pred_rf(model_rf_4, val_clean[,3:23]) # get predictions
results_val_rf[, Pred_Val_rf4 := pred_val_rf4] # store in data table

# Data5: get model and predictions
model_rf_5 <- get_model_rf(train_split5) 
pred_val_rf5 <- get_pred_rf(model_rf_5, val_clean[,3:23]) # get predictions
results_val_rf[, Pred_Val_rf5 := pred_val_rf5] # store in data table

# Data6: get model and predictions
model_rf_6 <- get_model_rf(train_split6) 
pred_val_rf6 <- get_pred_rf(model_rf_6, val_clean[,3:23]) # get predictions
results_val_rf[, Pred_Val_rf6 := pred_val_rf6] # store in data table

# Data7: get model and predictions
model_rf_7 <- get_model_rf(train_split7) 
pred_val_rf7 <- get_pred_rf(model_rf_7, val_clean[,3:23]) # get predictions
results_val_rf[, Pred_Val_rf7 := pred_val_rf7] # store in data table

# Data8: get model and predictions
model_rf_8 <- get_model_rf(train_split8) 
pred_val_rf8 <- get_pred_rf(model_rf_8, val_clean[,3:23]) # get predictions
results_val_rf[, Pred_Val_rf8 := pred_val_rf8] # store in data table

# Data9: get model and predictions
model_rf_9 <- get_model_rf(train_split9) 
pred_val_rf9 <- get_pred_rf(model_rf_9, val_clean[,3:23]) # get predictions
results_val_rf[, Pred_Val_rf9 := pred_val_rf9] # store in data table

# Data10: get model and predictions
model_rf_10 <- get_model_rf(train_split10) 
pred_val_rf10 <- get_pred_rf(model_rf_10, val_clean[,3:23]) # get predictions
results_val_rf[, Pred_Val_rf10 := pred_val_rf10] # store in data table

# Get Log Loss On Validation -------------------------------------------------------

# function to calculate log loss, adapted from https://rdrr.io/cran/MLmetrics/src/R/Classification.R
log_loss_fn <- function(predicted, actual, eps=1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  log_loss <- -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
  return(log_loss)
}

# get average predictions across all 10 models (ensemble method)
pred_val_rf_avg <- rowMeans(results_val_rf)

# get and store log loss performance on validation data
lloss_val_rf <- log_loss_fn(pred_val_rf_avg, actual_val_y)
