
# Initial data prep -------------------------------------------------

# load libraries
library("data.table")
library("randomForest")

# read in all data
load("train_clean.RData")
load("val1_clean.RData")
load("val2_clean.RData")
load("test_clean.RData")

# combine val1 and val2 data into one validation since opted for an ensemble method with no separate parameter tuning
train_clean <- train_cleaned
val_clean <-  rbind(val1_cleaned, val2_cleaned)
test_clean <- test_cleaned

# remove unnecessary objects from workspace
rm(list = c("test_cleaned", "train_cleaned", "val1_cleaned", "val2_cleaned"))

# ensure click is as.factor for all calculations in training
train_clean[, click:= as.factor(click)] 

# Align categorical variables ---------------------------------------------

# This section drops rows with levels that are missing in any of the three datasets (train/val/test)
# In practice, given earlier cleaning, this should only drop levels that occur in the training data but not in the validation sets
# This is because our earlier cleaning assigned any unseen levels from the validation and test data to the 'other' category

vars <- colnames(train_clean[, 3:23]) # get categorical variables that are relevant

factor_check <- data.table("variable"=character(), "mismatch"=character()) # table to store mismatch results

## for each categorical feature column in data, get values that are missing across one or more tables
for (v in vars){
  # get level values that are not overlapping across each dataset combination
  train_val <- setdiff(levels(train_clean[, get(v)]), levels(val_clean[, get(v)]))
  train_test <- setdiff(levels(train_clean[, get(v)]), levels(test_clean[, get(v)]))
  val_test <- setdiff(levels(val_clean[, get(v)]), levels(test_clean[, get(v)]))
  
  # if the length of the list that stores these 'mismatched' values is larger than 0, align levels 
  if (length(train_val)>0 | length(train_test)>0 | length(val_test) >0) {
    # get temp data table that has variable and values to drop
    temp <- data.table("variable"=v, "mismatch"=paste(unique(c(train_val, train_test, val_test)), collapse=","))
    factor_check <- rbind(factor_check, temp) # store values in table
    values_drop <- unlist(strsplit(factor_check$mismatch, split=",")) # get mismatched values to drop
    
    # convert original data to character to enable dropping
    train_clean[, (v) := as.character(get(v))] # convert original data to character
    val_clean[, (v) := as.character(get(v))] # convert original data to character
    test_clean[, (v) := as.character(get(v))] # convert original data to character
    
    # drop mismatched values
    train_clean <- train_clean[!(get(v) %in% values_drop), ] # drop values in mismatched field if any
    val_clean <- val_clean[!(get(v) %in% values_drop), ] # drop values in mismatched field if any
    test_clean <- test_clean[!(get(v) %in% values_drop), ] # drop values in mismatched field if any
    
    # convert back to factor
    train_clean[, (v) := as.factor(get(v))] # convert data back to factor
    val_clean[, (v) := as.factor(get(v))] # convert data back to factor
    test_clean[, (v) := as.factor(get(v))] # convert data back to factor
    
    # double-check it worked for the most recent variable
    newdiff1 <- setdiff(levels(train_clean[, get(v)]), levels(val_clean[, get(v)])) # check to make sure new diff is empty
    newdiff2 <- setdiff(levels(train_clean[, get(v)]), levels(test_clean[, get(v)])) # check to make sure new diff is empty
    newdiff3 <- setdiff(levels(val_clean[, get(v)]), levels(test_clean[, get(v)])) # check to make sure new diff is empty
  }
}

# overwrite and save files
save(train_clean, file="train_clean.RData")
save(val_clean, file="val_clean.RData")
save(test_clean, file="test_clean.RData")

# Split Data Chunks -------------------------------------------------------

# split full training data into 10 chunks to do build up to 10 models for each technique, and average results for ensemble

# randomly shuffle the dataset
n <- nrow(train_clean) # number of rows in data
train_split <- train_clean[sample(n), ] # shuffle dataset using random sample of all rows

# establish index cutoffs for all 10 chunks
cutoff_1 <- floor(0.1*n)
cutoff_2 <- floor(0.2*n)
cutoff_3 <- floor(0.3*n)
cutoff_4 <- floor(0.4*n)
cutoff_5 <- floor(0.5*n)
cutoff_6 <- floor(0.6*n)
cutoff_7 <- floor(0.7*n)
cutoff_8 <- floor(0.8*n)
cutoff_9 <- floor(0.9*n)

# create 10 non-overlapping chunks of data that are roughly equal
train_split1 <- train_split[1:cutoff_1,]
train_split2 <- train_split[(cutoff_1+1):cutoff_2,]
train_split3 <- train_split[(cutoff_2+1):cutoff_3,]
train_split4 <- train_split[(cutoff_3+1):cutoff_4,]
train_split5 <- train_split[(cutoff_4+1):cutoff_5,]
train_split6 <- train_split[(cutoff_5+1):cutoff_6,]
train_split7 <- train_split[(cutoff_6+1):cutoff_7,]
train_split8 <- train_split[(cutoff_7+1):cutoff_8,]
train_split9 <- train_split[(cutoff_8+1):cutoff_9,]
train_split10 <- train_split[(cutoff_9+1):n]

# save split files 
save(train_split1, file="train_split1.RData")
save(train_split2, file="train_split2.RData")
save(train_split3, file="train_split3.RData")
save(train_split4, file="train_split4.RData")
save(train_split5, file="train_split5.RData")
save(train_split6, file="train_split6.RData")
save(train_split7, file="train_split7.RData")
save(train_split8, file="train_split8.RData")
save(train_split9, file="train_split9.RData")
save(train_split10, file="train_split10.RData")
