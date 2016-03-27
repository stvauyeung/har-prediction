## SET UP WORKING DIRECTORY AND PARTITION DATA
setwd("~/Projects/data specialization/machine learning")
library(caret)

raw_test <- read.csv("pml-testing.csv")
raw_train <- read.csv("pml-training.csv")

inTrain <- createDataPartition(y=raw_train$classe, p=0.7, list=FALSE)
training <- raw_train[inTrain,]
testing <- raw_train[-inTrain,]

# segment training data into smaller partitions
inTrain2 <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
train_2 <- training[inTrain2,]
train_2_validation <- training[-inTrain2,]

# create 1000 row training set to prototype models
rndid <- with(training, ave(X, classe, FUN=function(x) {sample.int(length(x))}))
small_train <- training[rndid<=200,]
small_test <- training[rndid<=25,]

## CREATE DATAFRAMES WITH SEGMENTS OF AVAILABLE PREDICTORS
# column names for variables to be removed from main dataset for exploratory analysis
max_names <- grep("max", names(training), value=T)
min_names <- grep("min", names(training), value=T)
timestamp_names <- grep("timestamp", names(training), value=T)
window_names <- grep("window", names(training), value=T)
kurtosis_names <- grep("kurtosis", names(training), value=T)
skewness_names <- grep("skewness", names(training), value=T)
amplitude_names <- grep("amplitude", names(training), value=T)
var_names <- grep("var_", names(training), value=T)
stddev_names <- grep("stddev", names(training), value=T)
avg_names <- grep("avg_", names(training), value=T)

# sensor only data dataframe creation
non_sensor_columns <- c("X", "user_name",
                    max_names, min_names, 
                    timestamp_names, window_names,
                    kurtosis_names, skewness_names, 
                    amplitude_names, var_names, 
                    stddev_names, avg_names)

subset_sensor_columns <- function(dataframe) {
    sensor_columns <- setdiff(names(dataframe), non_sensor_columns)
    return(dataframe[,sensor_columns])
}

# sensor data with timeseries dataframe creation
timeseries_unused_columns <- c(max_names, min_names, 
                               kurtosis_names, skewness_names, 
                               amplitude_names, var_names, 
                               stddev_names, avg_names)

subset_ts_sensor_columns <- function(dataframe) {
    ts_obs_columns <- setdiff(names(dataframe), timeseries_unused_columns)    
    return(dataframe[,ts_obs_columns])
}

# look at correlation between sensor variables
convert_variables_to_numeric <- function(dataframe) {
    n <- length(dataframe) - 1
    for(i in 1:n) {
        dataframe[,i] <- as.numeric(dataframe[,i])
    }
    return(dataframe)
}

num_sensor_data <- convert_variables_to_numeric(sensor_data[,c(-53)])
M <- abs(cor(num_sensor_data))
diag(M) <- 0
which(M > 0.9,arr.ind=T) # returns 44 highly correlated predictors, suggesting PCA would be useful

# plot different predictors against timeseries


# table with colnames, sd features, var features
mean <- sapply(sensor_data, mean)
var <- sapply(sensor_data, var)
df <- data.frame(mean, var)


## MODEL TRAINING AND PREDICTION

smallT_sensor_columns <- subset_sensor_columns(small_train)
smallT_ts_sensor_columns <- subset_ts_sensor_columns(small_train)
# classification tree
modFitTree1 <- train(classe~., data=smallT_sensor_columns, method="rpart") # small_train very fast; train_2 1 min
predTree1 <- predict(modFitTree1, train_2_validation)
confusionMatrix(predTree1, train_2_validation$classe) # 43% accuracy using small_train, 49% using train_2

modFitTree2 <- train(classe~., data=smallT_ts_sensor_columns, method="rpart") # < 1 min
predTree2 <- predict(modFitTree2, train_2_validation)
confusionMatrix(predTree2, train_2_validation$classe) # 99% accuracy, 66% using train_2

# random forest
modFitRF1 <- train(classe~., data=smallT_sensor_columns, method="rf") # start 3:01 end 3:07
modFitRF2 <- train(classe~., data=smallT_sensor_columns, method="rf", preProcess="pca") # using small_train 12:24 start 12:27 end
modFitRF3 <- train(classe~., data=smallT_ts_sensor_columns, method="rf")

predRF1 <- predict(modFitRF1, newdata=train_2_validation)
confusionMatrix(predRF1, train_2_validation$classe) # results in 90.8% accuracy

predRF2 <- predict(modFitRF2, newdata=train_2_validation)
confusionMatrix(predRF2, train_2_validation$classe) # results in 81.0% accuracy

predRF3 <- predict(modFitRF3, newdata=train_2_validation)
confusionMatrix(predRF3, train_2_validation$classe)

# boosting
modFitBoo1 <- train(classe~., data=smallT_sensor_columns, method="gbm") # start 5:50 end 6:05
predBoo1 <- predict(modFitBoo1, newdata=train_2_validation)
confusionMatrix(predBoo1, train_2_validation$classe) # results in 89.1% accuracy w small_train, 95% with train_2 size data -- used on prediction quiz

modFitBoo2 <- train(classe~., data=smallT_ts_sensor_columns, method="gbm") # start 5:20 end 4:43
predBoo2 <- predict(modFitBoo2, newdata=train_2_validation)
confusionMatrix(predBoo2, train_2_validation$classe) # results in 89.1% accuracy, 99% using timeseries data, only predicts A on raw_test

# model accuracy table
model = c("tree", "tree", "random forest", "random forest w/ pca", "random forest", "boosting", "boosting")
dataset = c("sensor", "sensor+ts", "sensor", "sensor", "sensor+ts", "sensor", "sensor+ts")
accuracy = c(confusionMatrix(predTree1, train_2_validation$classe)$overall[1], 
             confusionMatrix(predTree2, train_2_validation$classe)$overall[1],
             confusionMatrix(predRF1, train_2_validation$classe)$overall[1],
             confusionMatrix(predRF2, train_2_validation$classe)$overall[1],
             confusionMatrix(predRF3, train_2_validation$classe)$overall[1],
             confusionMatrix(predBoo1, train_2_validation$classe)$overall[1],
             confusionMatrix(predBoo2, train_2_validation$classe)$overall[1])
accuracy_table = data.frame(Model = model, Dataset = dataset, Accuracy = accuracy)

# train_2 model fit creation using winning prototype models
## create subset data
train2_sensor_columns <- subset_sensor_columns(train_2)
train2_ts_sensor_columns <- subset_ts_sensor_columns(train_2)

mf_rf_ts <- train(classe~., data=train2_ts_sensor_columns, method="rf") # start 5:17 end 5:47
mf_rf_sensor <- train(classe~., data=train2_sensor_columns, method="rf")
mf_boost_ts <- train(classe~., data=train2_ts_sensor_columns, method="gbm")
mf_boost_sensor <- train(classe~., data=train2_sensor_columns, method="gbm")

pred_rf_ts <- predict(mf_rf_ts, train_2_validation)

rbind(table(predBoo1), table(predBoo2))
