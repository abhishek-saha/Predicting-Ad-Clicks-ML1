
# Data Import --------------------------------------------------------------

# load relevant libraries
library("data.table") # for data manipulation
library("rpart") # for trees

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

# get 2 chunks from training data that will be used as a initial validation layer, to optimize tree parameters
val_params <- rbind(train_split9, train_split10) # compine 2 training chunks into validation for parameter values
actual_param_y <- as.integer(val_params$click) # convert to integer to compute log loss

# store validation y-values for evaluation 
actual_val_y <- val_clean$click # store as integer for computing log-loss
val_clean[, click := as.factor(click)] # store as factor for passing into algorithm

# Functions for Model-Building and Prediction -----------------------------------

# Getting formula for tree
vars <- colnames(train_split1) # get column names
fm_tree <- paste(vars[2],"~",paste(vars[3:23],collapse=" + "),sep=" ") # create string of formula with x-variables
fm_tree <- formula(fm_tree) # convert string to a formula

# function to calculate log-loss to enable minsplit parameter optimization
log_loss_fn <- function(predicted, actual, eps=1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  log_loss <- -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
  return(log_loss)
}

# function to get get best min split argument for a chunk
get_best_minsplit <- function(chunk) {
  
  # Initialize data table for min split and associated log loss
  ## arguments for min-split based on initial testing of 17 values on chunk 1, where best minsplit was 6000
  ## based on the first chunk performance on these, we down-sized to 6 arguments with added focus on the range around 6000
  results_tree <- data.table("min_split" = c(20, 500, 2000, 5000, 6000, 7000), 
                             "log_loss" = rep(-10, 6))
  
  # Find the log loss values for each minsplit value to test
  for (i in 1:nrow(results_tree)){
    # Fit the full tree once
    tree_control <- rpart.control(minsplit=results_tree[i, min_split], maxdepth=30, cp=0) # control parameters
    out_tree <- rpart(fm_tree, data=chunk, control=tree_control, 
                      method="class", parms = list(split = 'information')) # tree function call
    
    # Find the cp parameter for the best model (i.e. with lowest error) and prune the tree.
    best_cp <- out_tree$cptable[which.min(out_tree$cptable[,"xerror"]),"CP"]
    
    # Store the "best" model
    out_best_tree <- prune(out_tree, cp=best_cp)
    
    # Make the predictions for the parameter validation data
    pred_param_tree <- predict(out_best_tree, newdata=val_params[, 3:23], type="prob")
    pred_param_tree <- pred_param_tree[,2]
    
    # Get performance on log-loss
    results_tree[i, log_loss := log_loss_fn(pred_param_tree, actual_param_y)]
  }
  
  # Obtaining the parameter with lowest log loss i.e. the best min split
  best_minsplit_tree <- results_tree[which.min(log_loss), min_split]
  
  return(best_minsplit_tree) # return th ebest minsplit
}

# function to build the tree model for a chunk
get_model_tree <- function(chunk){
  
  # Get the best minsplit argument for this chunk
  best_minsplit_tree <- get_best_minsplit(chunk)
  
  # Fit the full tree once using the best minsplit argument
  tree_control <- rpart.control(minsplit=best_minsplit_tree, maxdepth=30, cp=0)
  initial_tree <- rpart(fm_tree, data=chunk, control=tree_control, method="class", 
                        parms = list(split = 'information'))
  
  # Find the cp parameter for the best model
  best_cp_tree <- initial_tree$cptable[which.min(initial_tree$cptable[,"xerror"]),"CP"]
  
  # Storing the "best" model
  model_tree <- prune(initial_tree, cp = best_cp_tree)
  
  return(model_tree) # return the model
}

# function to get predicted values
get_pred_tree <- function(model, new_data) {
  pred_tree <- predict(model, newdata=new_data, type="prob") # get predictions
  pred_tree <- pred_tree[,2] # only get probabilities for prediction of 1 i.e. click outcome
  return(pred_tree)
}

# Get Model and Predictions For Each Chunk -------------------------------------------------------

## Note: only ran for 8 training chunks, since the other 2 were used to tune parameters 
## i.e. served as a type of validation data
## ran this manually instead of in a loop to examine and store each model one at a time

# Data1: get model and predictions
model_tree_1 <- get_model_tree(train_split1) # get model
pred_val_tree1 <- get_pred_tree(model_tree_1, val_clean[,3:23]) # get predictions,
results_val_tree <- data.table("Pred_Val_tree1"=pred_val_tree1) # store in data table

# Data2: get model and predictions
model_tree_2 <- get_model_tree(train_split2) # get model
pred_val_tree2 <- get_pred_tree(model_tree_2, val_clean[,3:23]) # get predictions
results_val_tree[, Pred_Val_tree2 := pred_val_tree2] # store in data table

# Data3: get model and predictions
model_tree_3 <- get_model_tree(train_split3)
pred_val_tree3 <- get_pred_tree(model_tree_3, val_clean[,3:23]) # get predictions
results_val_tree[, Pred_Val_tree3 := pred_val_tree3] # store in data table

# Data4: get model and predictions
model_tree_4 <- get_model_tree(train_split4) 
pred_val_tree4 <- get_pred_tree(model_tree_4, val_clean[,3:23]) # get predictions
results_val_tree[, Pred_Val_tree4 := pred_val_tree4] # store in data table

# Data5: get model and predictions
model_tree_5 <- get_model_tree(train_split5) 
pred_val_tree5 <- get_pred_tree(model_tree_5, val_clean[,3:23]) # get predictions
results_val_tree[, Pred_Val_tree5 := pred_val_tree5] # store in data table

# Data6: get model and predictions
model_tree_6 <- get_model_tree(train_split6) 
pred_val_tree6 <- get_pred_tree(model_tree_6, val_clean[,3:23]) # get predictions
results_val_tree[, Pred_Val_tree6 := pred_val_tree6] # store in data table

# Data7: get model and predictions
model_tree_7 <- get_model_tree(train_split7) 
pred_val_tree7 <- get_pred_tree(model_tree_7, val_clean[,3:23]) # get predictions
results_val_tree[, Pred_Val_tree7 := pred_val_tree7] # store in data table

# Data8: get model and predictions
model_tree_8 <- get_model_tree(train_split8) 
pred_val_tree8 <- get_pred_tree(model_tree_8, val_clean[,3:23]) # get predictions
results_val_tree[, Pred_Val_tree8 := pred_val_tree8] # store in data table

# Get Log Loss On Validation -----------------------------------------------

# get average predictions across all 10 models (ensemble method)
pred_val_tree_avg <- rowMeans(results_val_tree)

# get and store log loss performance on validation data
lloss_val_tree <- log_loss_fn(pred_val_tree_avg, actual_val_y)
