########################################################
### Final Project: Capital Bike Share
########################################################
###
### Load Data Analytics Functions and Dataset
source("DataAnalyticsFunctions.R")
source("causal_source.R")
df <- read.csv("capitalbikeshare-complete.csv")
summary(df)
anyNA(df)
###
#######################################################
###
### Add rain, snow
library(dplyr)
df <- df %>%
  mutate(rain = if_else(is.na(df$rain_1h),0,1),
         snow = if_else(is.na(df$snow_1h),0,1))
df[is.na(df$rain_1h),"rain_1h"] <- 0  
df[is.na(df$snow_1h),"snow_1h"] <- 0
anyNA(df)
### Divide datetime into Year, Month, Day, Hour
names(df)[1] <- "datetime"
df$year <- as.factor(substr(df$datetime, 1, 4))
df$mon <- as.factor(substr(df$datetime, 6, 7))
# df$day <- as.factor(substr(df$datetime, 9, 10))
df$hour <- as.factor(substr(df$datetime, 12, 13))
df$datetime <- NULL
df$holiday <- as.factor(df$holiday)
df$workingday <- as.factor(df$workingday)
df$rain <- as.factor(df$rain)
df$snow <- as.factor(df$snow)
table(df$weather_main)
df$weather_main[df$weather_main %in% c("Squall", "Haze", "Smoke")] <- "Other"
df$weather_main <- as.factor(df$weather_main)
summary(df)
###
#######################################################
#######################################################
###
### Visualization
pkg <- c("readr","readxl","dplyr","stringr","ggplot2","tidyr","car","nycflights13", "gapminder", "Lahman","skimr","data.table","ggiraph","ggiraphExtra","plyr","corrplot")
pkgload <- lapply(pkg, require, character.only = TRUE)
###
### Boxplot of all numeric variables
box_plot <- df[,-c(2,3,15,16,17,18,19,20)]
par(mfrow=c(2,6))
for (i in 1:length(box_plot)) {
  boxplot(box_plot[,i], main=names(box_plot[i]), type="l",col="steelblue")
  
}
###
### Correlation between variables
par(mfrow=c(1,1))
cor <- data.frame(lapply(df[,-15], function(x) as.numeric(as.character(x))))
corrplot(cor(cor), method = "number", type = "upper")
###
### Relationship between humidity and rain_1h
rain <- subset(df, df$rain_1h > 10)
rain
ggplot(data = df, aes(x = rain_1h, y = humidity , size= count, color = round_any(count,200) )) + geom_point() + geom_smooth()
ggplot(data=df, aes(x=temp, y=feels_like)) + geom_point() + geom_smooth()
ggplot(data=df, aes(x=temp, y=pressure, size= count, color = round_any(count,200))) + geom_point() + geom_smooth()
###
#######################################################
#######################################################
###
### Unsupervised Learning
# Drop all the non-numeric variables
ana_bike <- df[,c(-1,-15)]
ana_bike <- data.frame(lapply(ana_bike, function(x) as.numeric(as.character(x))))
pca.bike <- prcomp(ana_bike, scale=TRUE)
plot(pca.bike,main="PCA: Variance Explained by Factors")
mtext(side=1, "Factors",  line=1, font=2)
# Based on the graph, we observe that the variance explained by factors shows a decreasing trend
# To conduct further analysis, we focus on the first three factors
bikepc <- predict(pca.bike)
bikepc
loadings <- pca.bike$rotation[,1:5]
round(loadings,2)
bikepc <- predict(pca.bike)

options(scipen = 999)
pc <- as.data.frame(round(bikepc[,1:5],2))

# Since the count variable is numeric, it is not easy to identify the trend simply by looking at the PCA result.
# We split the count info into four categories using the quantile(1st quantile, median, 3er quantile).
cati <- rep(NA,nrow(df))
for (i in 1:nrow(df)) {
  if (df$count[i] <= 59) {
    cati[i] <- 1
  } else if (df$count[i] <= 236) {
    cati[i] <- 2
  } else if (df$count[i] <= 495) {
    cati[i] <- 3
  } else {cati[i] <- 4}
}

pc$count_cati <- as.data.frame(cati)
cati1 <- pc[(pc$count_cati == 1),]
cati2 <- pc[(pc$count_cati == 2),]
cati3 <- pc[(pc$count_cati == 3),]
cati4 <- pc[(pc$count_cati == 4),]

# We calculate the mean of correlation which may somehow identify the overall trend.
round(colMeans(cati1),2)
round(colMeans(cati2),2)
round(colMeans(cati3),2)
round(colMeans(cati4),2)


### Find the critical variable in the dataset to explain the COUNT
library(glmnet)
options(scipen = 999)

colnames(df)
CVA_bike <- df[,-15]
CVA_bike <- data.frame(lapply(CVA_bike, function(x) as.numeric(as.character(x))))
y <- CVA_bike$count
x <- model.matrix( count ~ ., data = CVA_bike )
name <- colnames(CVA_bike)
for (i in 2:18){
  d <- CVA_bike[,i]
  cl <- round(CausalLinear(y,d,x),3)
  print(name[i])
  print(round(CausalLinear(y,d,x),3))
}
#######################################################
#######################################################
### Start modeling
### Create Hold Out Sample
set.seed(2)
n <- nrow(df)
holdout.indices <- sample(n,n*0.3)
holdout <- df[holdout.indices,]
train <- df[-holdout.indices,]
actual <- holdout$count
summary(train$count)
summary(holdout$count)
###
#######################################################
###
### Baseline: using average demand
p_base <- mean(train$count)
oos_base <- R2 (y=actual, pred = p_base)
###
#######################################################
###
### Linear Regression Model with Interaction with workingday
m_lm <- lm(count ~ ., data = train)
m_lm <- step(m_lm)
p_lm <- predict(m_lm, holdout)
oos_lm <- R2(y=actual, pred=p_lm)
###
#######################################################
###
### Linear Regression Model with Interaction with workingday
m_lm_w <- lm(count ~ .*workingday, data = train)
m_lm_w <- step(m_lm_w)
p_lm_w <- predict(m_lm_w, holdout)
oos_lm_w <- R2(y=actual, pred=p_lm_w)
###
#######################################################
###
### Lasso Estimates
library(glmnet)
Mx<- model.matrix(count ~ .*workingday, data=df)[,-1]
My<- df$count
lasso <- glmnet(Mx,My)
summary(lasso)
lassoCV <- cv.glmnet(Mx,My)
# Post Lasso Estimates
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)
data.min <- data.frame(Mx[,features.min],My)
m_post <- glm(My~., data=data.min[-holdout.indices,])
p_post <- predict(m_post,newdata=data.min[holdout.indices,],type="response")
oos_post <- R2(y=actual, pred=p_post)
# Lasso Estimates
m_lasso <- glmnet(Mx[-holdout.indices,], My[-holdout.indices], lambda = lassoCV$lambda.min)
p_lasso <- predict(m_lasso, Mx[holdout.indices,], type="response")
OOS_lasso <- R2(y=actual, pred=p_lasso)
###
#######################################################
###
### Ridge
lambdas <- 10^seq(2, -3, by = -.1)
m_ridge = glmnet(Mx[-holdout.indices,], My[-holdout.indices], nlambda = 10, alpha = 0, lambda = lambdas)
cv_ridge <- cv.glmnet(Mx[-holdout.indices,], My[-holdout.indices], alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda
p_ridge <- predict(m_ridge, s = optimal_lambda, newx = Mx[holdout.indices,])
OOS_ridge <- R2(actual,p_ridge)
###
#######################################################
###
### rpart tree
library(rpart)
m_rt <- rpart(count ~ .,data  = train,method  = "anova")
p_rt <- predict(m_rt,holdout)
OOS_rt <- R2(y=actual, pred=p_rt)
###
### Random Forest
library(randomForest)
m_rf <- randomForest(count~.,train, ntree=20)
p_rf <- predict(m_rf,holdout)
OOS_rf <- R2(y=actual, pred=p_rf)
###
###
###
### plotting the relatinoship between prediction and actual counts.
plot(actual,p_lm_w)
plot(actual,p_rf)
### Linear regression model does not seem to capture the relationship as well as random forest.
#######################################################
###
### K-fold
set.seed(2)
nfold <- 10
n <- nrow(df)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
### create an empty dataframe of results
OOS <- data.frame(rf=rep(NA,nfold)) 
### Use a for loop to run through the nfold trails
for(k in 1:nfold){ 
  cv <- which(foldid!=1) # train on all but fold `k'
  # Fit all the underlying models used in average
  model_rf <- randomForest(count~.,df[cv,], ntree=20)
  # Make predictions and average the predictions
  pred_rf <- predict(model_rf,df[-cv,])
  # Calculate out=of-sample R2 for each k fold
  OOS$rf[k] <- R2(y=df$count[-cv], pred=pred_rf)
  print(paste("Iteration",k,"of",nfold,"completed with R2 of", OOS$rf[k]))
}
# Use boxplot to visualize the out of sample R2 in with during different k folds
boxplot(OOS)
summary(OOS$rf)
#######################################################
#######################################################
### The following code runs a random forest with 500 trees to make the following forecasts.
### It might take some time to run.
Model <- randomForest(count~.,train)
### We use the data on Oct 7, 2021 18:00
test <- holdout[1,]
test[1,]<-c(NA, 0, 1, 24, 24, 17, 24, 1022, 58, 3, 310, 0, 0, 20, "Clouds", 0, 0, 2021, 10, 18)
d <- predict(Model, test)
d_full <- as.numeric(predict(Model, holdout[1,], predict.all=TRUE)[["individual"]])
hist(d_full)
p <- 1
c <- 0.28
profit <- 0
supply <- 0
for (s in (d-100):(d+100)){
  prob <- approx(sort(d_full), seq(0,1,,length(d_full)), s)$y
  new_profit <- p*(s*(1-prob) + d*prob)-c*s
  if (new_profit > profit){
    profit <- new_profit
    supply <- s
  }
}
print(paste("By supplying",round(supply,0),"bikes, you will gain a profit of $",round(profit,2)))

