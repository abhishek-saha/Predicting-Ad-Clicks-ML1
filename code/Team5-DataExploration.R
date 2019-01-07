
# load libraries
library(data.table)

# read the full data
TrainingData <- fread("ProjectTrainingData.csv")

# check for null values
sapply(TrainingData, function(x) sum(is.na(x)))

# check the classes for y variable
Y_variable <- TrainingData[,.N, by=click]
Y_variable[,proportion := N/sum(N)]

# get names of categorical variables
factor_cols <- colnames(TrainingData[,3:24]) 

# factorise categorical variables
TrainingData[, (factor_cols) := lapply(.SD, as.factor), .SDcols=3:24]

# get number of levels for each categorical variable
num_levels_dt <- TrainingData[, lapply(.SD, function(x) length(levels(x))), .SDcols=3:24] # get table with frequency of occurrence for each variable
num_levels_dt <- melt(num_levels_dt,  measure.vars=c(1:22), variable.name="variable", value.name="frequency") # transpose format to get a frequency table

# get subset list of variables that have too many levels and need to be 'downsized' using pareto principle
fix_levels_dt <- num_levels_dt[frequency > 30,]

# for each variable to fix, get data table with count and proportion of values by each unique level, ordered from highest to lowest 
num_rows <- nrow(factors_data) # get total number of rows in data for proportion calculation
freq_tables_list <- data.table() # initialize an empty list

# store a datatable that has the frequency count for each variable to fix in a nested list
for (n in fix_levels_dt$variable){ # loop through variable names that need fewer levels
  temp <- factors_data[, .(freq=.N, prop=.N/num_rows), by=n][order(-freq)] # get data table containing each variable's data by level
  freq_tables_list <- c(freq_tables_list, list(temp))
}

# split the dataset into 60% for training, 20% for validation 1 and 20% for validation 2

trainIndex <- sample(1:nrow(TrainingData), size = round(0.6 * nrow(TrainingData)),replace = FALSE) # get training index for split
train <- TrainingData[trainIndex,] # create training data
val <- TrainingData[-trainIndex,] # create validation data

# further split the validation data into val1 and val2, to have separate datasets for parameter optimization and model evaluation
valindex <- sample(1:nrow(val), size = round(0.5 * nrow(val)),replace = FALSE)
val1 <- val[valindex,]
val2 <- val[-valindex,]
