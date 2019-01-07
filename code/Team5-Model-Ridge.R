# Data Import --------------------------------------------------------------

# load relevant libraries
library("data.table") # for data manipuation
library("glmnet") # for lasso logistic regression
library("Matrix") # for creating sparse matrices

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

# Data Prep For Matrices --------------------------------------------------

# To enable faster computation for logistic regression, our approach creates a sparse matrix as input
## Since this involves splitting out each level of a variable into a separate column, to ensure the dimensions
## are the same, we start by combining all the training data into one matrix and then splitting it back out

# get full training data to deal with different level issues
train_full <- rbind(train_split1, train_split2, train_split3, train_split4, train_split5, train_split6,
                    train_split7, train_split8, train_split9, train_split10)

# extract categorical variable names to a list (use for all training/validation data)
list_colnames <- as.data.table(colnames(train_split1))
list_colnames <- list_colnames[!V1%in%c('id','click'),]

# initialize matrix based on first factor
C1 <- factor(train_full$C1)
X_train_matrix <- sparse.model.matrix(~C1)

# Get sparse matrix that combines values for all factors (append to initial matrix)
for (name in list_colnames[[1]][2:21]){
  assign(name, factor(train_full[[name]]))
  matrix_temp <- sparse.model.matrix(~get(name))
  X_train_matrix <- cbind(X_train_matrix, matrix_temp)
}

# get cutoffs for each split that preserve initial order
cutoff1 <- nrow(train_split1)
cutoff2 <- cutoff1 + nrow(train_split2)
cutoff3 <- cutoff2 + nrow(train_split3)
cutoff4 <- cutoff3 + nrow(train_split4)
cutoff5 <- cutoff4 + nrow(train_split5)
cutoff6 <- cutoff5 + nrow(train_split6)
cutoff7 <- cutoff6 + nrow(train_split7)
cutoff8 <- cutoff7 + nrow(train_split8)
cutoff9 <- cutoff8 + nrow(train_split9)

# apply cutoffs to get updated chunks
train_mat1 <- X_train_matrix[1:cutoff1,]
train_mat2 <- X_train_matrix[(cutoff1+1):cutoff2,]
train_mat3 <- X_train_matrix[(cutoff2+1):cutoff3,]
train_mat4 <- X_train_matrix[(cutoff3+1):cutoff4,]
train_mat5 <- X_train_matrix[(cutoff4+1):cutoff5,]
train_mat6 <- X_train_matrix[(cutoff5+1):cutoff6,]
train_mat7 <- X_train_matrix[(cutoff6+1):cutoff7,]
train_mat8 <- X_train_matrix[(cutoff7+1):cutoff8,]
train_mat9 <- X_train_matrix[(cutoff8+1):cutoff9,]
train_mat10 <- X_train_matrix[(cutoff9+1):nrow(X_train_matrix),]

# Functions for Model-Building and Prediction---------------------------------------------------------------

# extract categorical variable names to a list (use for all training/validation data)
list_colnames <- as.data.table(colnames(train_split1))
list_colnames <- list_colnames[!V1%in%c('id','click'),]

# Function to get logistic regression ridge model for each chunk
get_model_ridge <- function(matrix_chunk, y) {
  
  # Fit logistic regression using ridge in glmnet, use cv to get best lambda
  chunk_ridge <- cv.glmnet(matrix_chunk, y, family = "binomial", 
                           alpha = 0, lambda = NULL, nfolds=5)
  
  # Return model
  return(chunk_ridge)
}

# Function to get model results for each chunk
get_pred_ridge <- function(model, new_data){
  # convert new-data to sparse matrix format
  C1 <-factor(new_data$C1)
  X_pred_matrix<- sparse.model.matrix(~C1)
  
  for (name in list_colnames[[1]][2:21]){
    assign(name, factor(new_data[[name]]))
    matrix_temp <- sparse.model.matrix(~get(name))
    X_pred_matrix <- cbind(X_pred_matrix, matrix_temp)
  }
  
  # from the model, get best lambda i.e. that which minimizes error
  best_lambda_ridge <- model$lambda.min
  
  # make the prediction
  pred_ridge <- predict(model, newx=X_pred_matrix, type='response', s=best_lambda_ridge)
  
  # return predictions
  return(pred_ridge)
}

# Get Model and Predictions For Each Chunk -------------------------------------------------------

# get logistic ridge results for each chunk
# ran this manually instead of in a loop to examine and store each model one at a time

# Data1: get model and predictions
model_ridge_1 <- get_model_ridge(train_mat1, train_split1$click) # get model
pred_val_ridge1 <- get_pred_ridge(model_ridge_1, val_clean[,3:23]) # get predictions
results_val_ridge <- data.table("Pred_Val_ridge1"=pred_val_ridge1) # store in data table

# Data2: get model and predictions
model_ridge_2 <- get_model_ridge(train_mat2, train_split2$click) # get model
pred_val_ridge2 <- get_pred_ridge(model_ridge_2, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge2 := pred_val_ridge2] # store in data table

# Data3: get model and predictions
model_ridge_3 <- get_model_ridge(train_mat3, train_split3$click)
pred_val_ridge3 <- get_pred_ridge(model_ridge_3, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge3 := pred_val_ridge3] # store in data table

# Data4: get model and predictions
model_ridge_4 <- get_model_ridge(train_mat4, train_split4$click) 
pred_val_ridge4 <- get_pred_ridge(model_ridge_4, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge4 := pred_val_ridge4] # store in data table

# Data5: get model and predictions
model_ridge_5 <- get_model_ridge(train_mat5, train_split5$click) 
pred_val_ridge5 <- get_pred_ridge(model_ridge_5, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge5 := pred_val_ridge5] # store in data table

# Data6: get model and predictions
model_ridge_6 <- get_model_ridge(train_mat6, train_split6$click) 
pred_val_ridge6 <- get_pred_ridge(model_ridge_6, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge6 := pred_val_ridge6] # store in data table

# Data7: get model and predictions
model_ridge_7 <- get_model_ridge(train_mat7, train_split7$click) 
pred_val_ridge7 <- get_pred_ridge(model_ridge_7, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge7 := pred_val_ridge7] # store in data table

# Data8: get model and predictions
model_ridge_8 <- get_model_ridge(train_mat8, train_split8$click) 
pred_val_ridge8 <- get_pred_ridge(model_ridge_8, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge8 := pred_val_ridge8] # store in data table

# Data9: get model and predictions
model_ridge_9 <- get_model_ridge(train_mat9, train_split9$click) 
pred_val_ridge9 <- get_pred_ridge(model_ridge_9, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge9 := pred_val_ridge9] # store in data table

# Data10: get model and predictions
model_ridge_10 <- get_model_ridge(train_mat10, train_split10$click) 
pred_val_ridge10 <- get_pred_ridge(model_ridge_10, val_clean[,3:23]) # get predictions
results_val_ridge[, Pred_Val_ridge10 := pred_val_ridge10] # store in data table

# Get Log Loss On Validation -------------------------------------------------------

# function to calculate log loss, adapted from https://rdrr.io/cran/MLmetrics/src/R/Classification.R
log_loss_fn <- function(predicted, actual, eps=1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  log_loss <- -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
  return(log_loss)
}

# get average predictions across all 10 models (ensemble method)
pred_val_ridge_avg <- rowMeans(results_val_ridge)

# get and store log loss performance on validation data
lloss_val_ridge <- log_loss_fn(pred_val_ridge_avg, actual_val_y)
