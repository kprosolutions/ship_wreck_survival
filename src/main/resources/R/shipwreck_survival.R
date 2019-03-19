#Load required packages
library(dplyr) 
library(Amelia) 
library(scales) 
library(caTools) 
library(e1071) 
library(rpart)
library(rpart.plot) 
library(randomForest) 
library(caret)

#Read the data
read_titanic_data = read.csv("Titanic_data.csv")
View(read_titanic_data)

#============================================================================
##Data Wrangling
#============================================================================

# Checking missing values (missing values or empty values)
colSums(is.na(read_titanic_data)|read_titanic_data=='')

#Missing Embarked Data Imputation
#Get the mode Create the function.
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
v = read_titanic_data$Embarked

result =getmode(v)

read_titanic_data$Embarked[read_titanic_data$Embarked == ""] = result

#Let's impute missing ages based on PCclass
# First, transform all feature to dummy variables.
dummy.vars <- dummyVars(~ ., data = read_titanic_data[, -1])
train.dummy <- predict(dummy.vars, read_titanic_data[, -1])
View(train.dummy)

# Now, impute!
pre.process <- preProcess(train.dummy, method = "bagImpute")
imputed.data <- predict(pre.process, train.dummy)
View(imputed.data)

read_titanic_data$Age <- imputed.data[, 6]
View(read_titanic_data)

# Checking missing values
colSums(is.na(read_titanic_data)|read_titanic_data=='')

#===========================================================================
#Feature Engineering
#===========================================================================

# Add a feature for family size.
read_titanic_data$FamilySize <- 1 + read_titanic_data$SibSp + read_titanic_data$Parch

# Adding a new feature Title
#Grab passenger title from passenger name
read_titanic_data$Title <- gsub("^.*, (.*?)\\..*$", "\\1", read_titanic_data$Name)

# Frequency of each title by sex
table(read_titanic_data$Sex, read_titanic_data$Title)

# There are so many categories let's combine few categories and put less frequent categories as other
read_titanic_data$Title[read_titanic_data$Title == 'Mlle' | read_titanic_data$Title == 'Ms'] <- 'Miss' 
read_titanic_data$Title[read_titanic_data$Title == 'Mme']  <- 'Mrs' 

Other <- c('Dona', 'Dr', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir')
read_titanic_data$Title[read_titanic_data$Title %in% Other]  <- 'Other'

# Let's check title 
table(read_titanic_data$Sex, read_titanic_data$Title)
#Let's check data now
View(read_titanic_data)

#Now we will keep only relevant data attributes in our data
feature= c("Survived", "Pclass", "Sex", "Age", "SibSp","Parch", "Fare", "Embarked","FamilySize","Title")
read_titanic_data= read_titanic_data[,feature]
View(read_titanic_data)

#==================================================================================
#Data preparation
#==================================================================================

#Encoding the categorical features as factors

read_titanic_data$Survived <- as.factor(read_titanic_data$Survived)
read_titanic_data$Pclass <- as.factor(read_titanic_data$Pclass)
read_titanic_data$Sex <- as.factor(read_titanic_data$Sex)
read_titanic_data$Embarked <- as.factor(read_titanic_data$Embarked)
read_titanic_data$Title = factor(read_titanic_data$Title)
#read_titanic_data$FamilySize = factor(read_titanic_data$FamilySize, levels=c("Single","Small","Large"))


#Splitting data into test and train data using Caret package 70% of data into training set and rest 30% into testing set
set.seed(32984)
indexes =createDataPartition(read_titanic_data$Survived,times = 1,p=0.7,list = FALSE)

#Creating Train and test data
training_data= read_titanic_data[indexes,]
test_data= read_titanic_data[-indexes,]

#Now we will verify proportions of our labels in data set and match it with original file
prop.table(table(read_titanic_data$Survived))
prop.table(table(training_data$Survived))
prop.table(table(test_data$Survived))

##Model training

##RF
set.seed(2020)

control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
modelRF <- train(Survived ~ ., data = training_data, method = "rf", trControl = control)
print(modelRF)
#DT
modelDT <- train(Survived~., data = training_data, method = "rpart", trControl = control)
print(modelDT)
##SVM
modelSVM <- train(Survived~., data = training_data, method = "svmLinear", trControl = control)
print(modelSVM)

#Predictions Against test Data

pred_rf <- predict(modelRF,test_data)
cm_rf <- confusionMatrix(pred_rf,test_data$Survived)

#Error rate
error_rf <- mean(test_data$Survived != pred_rf) # Misclassification error
paste('Accuracy',round(1-error_rf,4))

pred_tree <- predict(modelDT,test_data)
cm_tree <- confusionMatrix(pred_tree,test_data$Survived)

error_dt <- mean(test_data$Survived != pred_tree) # Misclassification error
paste('Accuracy',round(1-error_dt,4))

pred_svm <- predict(modelSVM,test_data)
cm_svm <- confusionMatrix(pred_svm,test_data$Survived)


error_svm <- mean(test_data$Survived != pred_svm) # Misclassification error
paste('Accuracy',round(1-error_svm,4))

#Comparing models
model_compare <- data.frame(Model = c('Random Forest', 'Trees','SVM'),
                            Accuracy = c(cm_rf$overall[1],cm_tree$overall[1], cm_svm$overall[1]))
#visualization

ggplot(aes(x=Model, y=Accuracy), data=model_compare) +
  geom_bar(stat='identity', fill = 'Black') +
  ggtitle('Comparative Accuracy of Models on Cross-Validation Data') +
  xlab('Models') +
  ylab('Overall Accuracy')

