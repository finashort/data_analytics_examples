'''
Fina Short Data200 Running Regressions: Modifications from Lab10 SupervisedRegression
'''
#install.packages("tree")
#install.packages("randomForest")
#install.packages("glmnet")
#install.packages("haven")
library(tree)
library(randomForest)
library(haven)
library(glmnet)

#large dataset with 817 variables
catest1_large <- read_dta("\\Users\\finas\\Downloads\\ca_school_testscore_1.dta")
#this is the small dataset
catest2_small <- read_dta("\\Users\\finas\\Downloads\\ca_school_testscore_2.dta")

#calculate in-sample root MSPE using training data, out-of-sample root MSPE using test data for:

########OLS using the small L dataset k=4##########
#need to extract 3 variables:  str_s, med_income_z, te_avgyr_s 

# Load data
data(catest2_small)
# Set seed
set.seed(673000)

#Set number of obs
n <- floor(0.75 * nrow(catest2_small)) # 75% of data for training

# Select training data
train_ind <- sample(seq_len(nrow(catest2_small)), size = n) #returns a set of indices of the sample data

# Split using index
ca2_train <- catest2_small[train_ind, ] #selecting an array of indices within dataset
#add in the mean calculations
ca2_means = colMeans(ca2_train)
ca2_test <- catest2_small[-train_ind, ] #cannot select train_ind indices

#standardize the data
ca2_train <- as.data.frame(scale(ca2_train, center=ca2_means, scale =TRUE))
ca2_test <- as.data.frame(scale(ca2_test, center=ca2_means, scale=TRUE))

# Run model
catest2_lm <- lm(testscore ~ str_s+ med_income_z+ te_avgyr_s, data = ca2_train)

# Find predictions on new data
catest2_pred <- predict(catest2_lm, newdata=ca2_test)

# Find In-sample root MSPE using the training data
is_casmall <- sqrt(mean((catest2_pred-catest2$testscore)^2))

#Find out-of-sample root MSPE, so all data apart from training data
catest2_lm_oos <- lm(testscore ~ str_s+ med_income_z+te_avgyr_s, data = ca2_test)
catest2_oospred <- predict(catest2_lm_oos, newdata=ca2_test)
oos_casmall <- sqrt(mean((catest2$testscore-catest2_oospred)^2))

########OLS using the large dataset k=817########

# Load data
data(catest1_large)
# Set seed
set.seed(673000)

#Set number of obs
n <- floor(0.75 * nrow(catest1_large)) # 75% of data for training

# Select training data
train_ind <- sample(seq_len(nrow(catest1_large)), size = n) #returns a set of indices of the sample data

# Split using index
ca1_train <- catest1_large[train_ind, ] #selecting an array of indices within dataset
ca1_test <- catest1_large[-train_ind, ] #cannot select train_ind indices

# Run model
catest1_lm <- lm(testscore ~., data = ca1_train)

# Find predictions on new data
catest1_pred <- predict(catest1_lm, newdata=ca1_test)

# Find In-sample root MSPE using the training data

is_calarge <- sqrt(mean((catest1_pred-ca1_test$testscore)^2))

#Find out of sample root MSPE
catest1_lm_oos <- lm(testscore ~., data = ca1_test_large)
catest1_oospred <- predict(catest1_lm_oos, newdata=catest1_large)
oos_calarge <- sqrt(mean((catest1_large$testscore-catest1_oospred[1])^2))

#######LASSO using the large dataset k=817############
library(glmnet)

# Load data
data(catest1_large)

x_var_lasso <-model.matrix(testscore~., data=catest1_large)[, -1]

#finding dependent variable
y_var_lasso <- y_var_ridge <- catest1_large%>%
  select(testscore)%>%
  unlist()%>%
  as.numeric()

# Set seed
set.seed(673000)

#Set number of obs
training = sample(1:nrow(x_var_lasso), nrow(x_var_lasso)/2) 
x_testing = (-training)
y_testing = y_var_lasso[x_testing]

library(glmnet)
lassomod <- cv.glmnet(x_var_lasso[training,],y_var_lasso[training], alpha=1, lambda=NULL)
predictlasso <- predict(lassomod, newx=x_var_lasso[x_testing,])

# Find In-sample root MSPE using the training data

is_calarge_lasso <- sqrt(mean((catest1_large$testscore-predictlasso[1])^2))

#Find out of sample root MSPE
catest1_lasso_oos <- cv.glmnet(x_var_lasso[x_testing,],y_var_lasso[x_testing], alpha=1, lambda=NULL)
catest1_oospred <- predict(catest1_lasso_oos, newx=x_var_lasso[x_testing,])
oos_calarge_lasso <- sqrt(mean((catest1_large$testscore-catest1_oospred[1])^2))

#######Ridge using the large dataset k=817############

library (car) # for VIF
library (ridge)
data(catest1_large)

#split training and test data
set.seed(673000) 
n <- floor(0.75 * nrow(catest1_large)) # 75% of data for training
ridge_train_ind <- sample(seq_len(nrow(catest1_large)), size = n) #returns a set of indices of the sample data

ca1_train_rid <- catest1_large[ridge_train_ind, ] #selecting an array of indices within dataset
#add in the mean calculations
ca1_means_rid = colMeans(ca1_train_rid)
ca1_test_rid <- catest1_large[-ridge_train_ind, ] #select everything except for train_ind indices

#in-sample root mspe
ridge_pred <- predict(lm_catest_ridge, data=ca2_train_rid)
ridge_ismspe <-sqrt(mean((ca1_train_rid$testscore-ridge_pred)^2))

#out of sample root mspe
ridge_oos = predict(lm_catest_ridge, newdata=ca2_test_rid)
ridge_oosmspe <-sqrt(mean((ca1_test_rid$testscore)-ridge_oos)^2)

#######Random forest using the medium dataset k=38#######
library(randomForest)
library(caret)

data(catest2_small)
set.seed(673000) 
n <- floor(0.75 * nrow(catest2_small)) # 75% of data for training
rf_train_ind <- sample(seq_len(nrow(catest2_small)), size = n) #returns a set of indices of the sample data

ca2_train_rf <- catest2_small[rf_train_ind, ] #selecting an array of indices within dataset
#add in the mean calculations
ca2_means_rf = colMeans(ca2_train_rid)
ca2_test_rf <- catest2_small[-rf_train_ind, ] #select everything except for train_ind indices

#use random forest
rf_model <- randomForest(testscore ~ ., data = ca2_train_rf)

#in-sample root mspe
rf_pred <- predict(rf_model, data=ca2_train_rf)
rf_ismspe <-sqrt(mean((ca2_train_rf$testscore-rf_pred)^2))

#out of sample root mspe
rf_oos = predict(rf_model, newdata=ca2_test_rf)
rf_oosmspe <-sqrt(mean((ca2_test_rf$testscore)-rf_oos)^2)

#generating a summary table with this information
summary_table<-matrix(c(is_casmall, is_calarge, ridge_ismspe, is_calarge_lasso, rf_ismspe, oos_casmall, oos_calarge, ridge_oosmspe, oos_calarge_lasso, rf_oosmspe), ncol = 5, byrow = TRUE)
colnames(summary_table)<-c("OLS Small", "OLS Large", "Ridge", "LASSO", "Random Forest")
rownames(summary_table)<-c("In sample root MSPE", "Out of sample root MSPE")
summary_table<-as.table(summary_table)
write.csv(summary_table, "Algorithm Summary Table Fina.csv")