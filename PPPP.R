.libPaths("C:/Users/Delia/R/win-library")
library(tm)
library(MASS)
library(caTools)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(e1071)
library(ROCR)
library(ggplot2)
library(rpart) # CART
library(rpart.plot) # CART plotting
library(caret) # cross validation
library(ROCR)
library(MASS)
library(randomForest)
library(gbm)
library(gdata)
library(forecast)
library(lubridate)
library(glmnet)
library(Matrix)
library(tictoc)
library(keras)
library(pls)
library(car) # for VIF
library(boot)
OSR2 <- function(predictions, train, test) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - SSE/SST
  return(r2)
}
#read in data
data <- read.csv("data.csv", header = TRUE)
data$Date <- ymd(data$Date,truncated = 30)
plot(data$Date,data$Sale.Price,type="l", )

##data prepossesion######################################################################################
#1 add log sale price
data$logSP <- log(data$Sale.Price)
plot(data$Date,log(data$Sale.Price), type="l", ylab="log sale-price")
data<- data[,!(names(data) %in% "Sale.Price")]

#2.deal with missing value in ZRI
#Since it is inside a decrease trend from the graph, we can just use moving average to fit the data
plot(data$Date,data$ZRI, type="line")
data$ZRI[62]=(data$ZRI[61]+data$ZRI[63])/2
plot(data$Date,data$ZRI, type="l")
head(data)

#4. Time series Data
pacfg <- pacf(data$logSP, lag.max=100)
data <- data %>%
  mutate(SPLM=c(NA, head(logSP, -1))) %>%
  mutate(SPL2M = c(NA, NA, head(logSP, -2)))
head(data)
data <- data[3:129,]
#5.seperate the data into two data: data A with all obversation; data B with all features
dropA <- c("BSI","MSInventory","ZRI","SLR","Days")
dataA <- data[,!(names(data) %in% dropA)]
dataB <- data[47:127,]
#separate training and testing for data A and B
trainA <- dataA %>% filter(year(dataA$Date) < 2018)
testA <- dataA %>% filter(year(dataA$Date) >= 2018)
trainB <- dataB %>% filter(year(dataB$Date) < 2018)
testB <- dataB %>% filter(year(dataB$Date) >= 2018)

#normalize A and B
trYA <- trainA$logSP
teYA <- testA$logSP
trYB <- trainB$logSP
teYB <- testB$logSP
drop <- c("Date","logSP")
trainA <- trainA[,!(names(trainA) %in% drop)]
testA <-testA[,!(names(testA) %in% drop)]
trainB <- trainB[,!(names(trainB) %in% drop)]
testB <-testB[,!(names(testB) %in% drop)]
preProc  <- preProcess(trainA, method = c("center","scale"))
trainA <- predict(preProc, trainA)
testA <- predict(preProc, testA)
preProc  <- preProcess(trainB, method = c("center","scale"))
trainB <- predict(preProc, trainB)
testB <- predict(preProc, testB)
trainA$logSP <-trYA
trainB$logSP <-trYB
testA$logSP <-teYA
testB$logSP <-teYB

### MODEL APPLY####################################################################################
##linear with A
fit <- lm(logSP~., data = trainA )
summary(fit)
pred.lm <- predict(fit, newdata = testA)
mean(abs(pred.lm - testA$logSP))
sqrt(mean((pred.lm - testA$logSP)^2))
OSR2(pred.lm, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red", type = "l")
lines(data$Date[107:127],pred.lm,type="l")
#linear with ridge A 
drop <- c("logSP")
feature = as.matrix(trainA[,!(names(trainA) %in% drop)])
x.test = as.matrix(testA[,!(names(testA) %in% drop)])
set.seed(242)
cv <- cv.glmnet(feature, trainA$logSP, alpha = 0)
pred.r <- predict(cv, newx = x.test)
mean(abs(pred.r - testA$logSP))
sqrt(mean((pred.r - testA$logSP)^2))
OSR2(pred.r, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red", type = "l")
lines(data$Date[107:127],pred.r,type="l")
#lasso A
set.seed(242)
la <- cv.glmnet(feature, trainA$logSP, alpha = 1)
pred.l <- predict(la, newx = x.test)
mean(abs(pred.l - testA$logSP))
sqrt(mean((pred.l - testA$logSP)^2))
OSR2(pred.l, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.l,type="l")
coef(cv, s=cv$lambda.min)
## LASSO A 0.9996831

##time series model:

fit <- auto.arima(trainA$logSP, stationary = FALSE, seasonal = FALSE)
fit <- arima(trainA$logSP, order=c(1,1,0))
preds <- forecast(fit2, h=21)
mean(abs(preds$mean - testA$logSP))
sqrt(mean((preds$mean - testA$logSP)^2))
OSR2(preds$mean, trainA$logSP, testA$logSP)
plot(preds)
lines(data$logSP)

#caret for A
set.seed(242)
cpVals = data.frame(cp = seq(0, .1, by=.001))
train.cart <- train(logSP~ .,
                    data = trainA,
                    method = "rpart",
                    tuneGrid = cpVals,
                    trControl = trainControl(method = "cv", number=10),
                    metric = "RMSE")
train.cart$results
train.cart
ggplot(train.cart$results, aes(x = cp, y = RMSE)) + geom_point(size = 3) + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))
ct = train.cart$finalModel
prp(ct, digits=3)
test.mm= as.data.frame(model.matrix(logSP ~.+0, data=testA))
pred.ct = predict(ct, newdata =test.mm)

mean(abs(pred.ct - testA$logSP))
sqrt(mean((pred.ct - testA$logSP)^2))
OSR2(pred.ct, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.ct,type="l")

#random forest A
set.seed(242)
rf <- train(logSP ~ .,
            data = trainA,
            method = "rf",
            tuneGrid = data.frame(mtry=1:24),
            trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
            metric = "RMSE")
# RMSE or Rsquared doesn't matter actually -- both will be generated for regression problems
rf$results
rf
best.rf <- rf$finalModel
pred.rf <- predict(best.rf, newdata = testA)

ggplot(rf$results, aes(x = mtry, y = RMSE)) + geom_point(size = 3) + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

mean(abs(pred.rf - testA$logSP))
sqrt(mean((pred.rf - testA$logSP)^2))
OSR2(pred.rf, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.rf,type="l")


##boosting for A 
set.seed(242)
tGrid = expand.grid(n.trees = (50:100)*1000, interaction.depth = c(1,2,4,6,8),
                    shrinkage = 0.002, n.minobsinnode = 10)
train.boost <- train(logSP ~ .,
                     data = trainA,
                     method = "gbm",   ## gradient boosting machine 
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "RMSE",
                     distribution = "gaussian")
train.boost
bt.model <- train.boost$finalModel
pred.b <- predict(bt.model, newdata = testA, n.trees = 92000, interaction.depth = 10) # can use same model matrix

ggplot(train.boost$results, aes(x = n.trees, y = RMSE, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")

mean(abs(pred.b - testA$logSP))
sqrt(mean((pred.b - testA$logSP)^2))
OSR2(pred.b, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.b,type="l")

###########since the data is not stationary, the result of trees are bad, lets use residue of time series to do the trees again to see the result.
train_re <- trainA
train_re$logSP <- as.numeric(fit$residuals)
test_re <- testA
test_re$logSP <- testA$logSP-as.numeric(preds$mean)

#random forest A
set.seed(242)
rfe <- train(logSP ~ .-SPLM-SPL2M,
            data = train_re,
            method = "rf",
            tuneGrid = data.frame(mtry=1:21),
            trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
            metric = "RMSE")
# RMSE or Rsquared doesn't matter actually -- both will be generated for regression problems
rfe$results
best.rfe <- rfe$finalModel
pred.rfe <- predict(best.rfe, newdata = test_re)

ggplot(rfe$results, aes(x = mtry, y = RMSE)) + geom_point(size = 3) + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

pred.rfe <- pred.rfe+as.numeric(preds$mean)
mean(abs(pred.rfe - testA$logSP))
sqrt(mean((pred.rfe - testA$logSP)^2))
OSR2(pred.rfe, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.rf,type="l")


##boosting for A 
set.seed(242)
tGrid = expand.grid(n.trees = (1:10)*10, interaction.depth = c(1,2,4,6,8,10),
                    shrinkage = 0.002, n.minobsinnode = 10)
train.boost <- train(logSP ~ .-SPLM-SPL2M,
                     data = train_re,
                     method = "gbm",   ## gradient boosting machine 
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "RMSE",
                     distribution = "gaussian")
train.boost
bte <- train.boost$finalModel
pred.be <- predict(bte, newdata = testA, n.trees = 20, interaction.depth = 2) # can use same model matrix

ggplot(train.boost$results, aes(x = n.trees, y = RMSE, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")
pred.be <- pred.be+as.numeric(preds$mean)
mean(abs(pred.be - testA$logSP))
sqrt(mean((pred.be - testA$logSP)^2))
OSR2(pred.be, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.b,type="l")
#PCR 
set.seed(242)
pcr.model <- pcr(logSP~., data=trainA, scale = TRUE, validation = "CV")
summary(pcr.model)
validationplot(pcr.model)
pred.pcr <- predict(pcr.model, newdata = testA, ncomp = 10)
mean(abs(pred.pcr - testA$logSP))
sqrt(mean((pred.pcr - testA$logSP)^2))
OSR2(pred.pcr, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.pcr,type="l")

#forward stepwise:
null <- lm(logSP~1, data = trainA)
full <- formula(lm(logSP~., data = trainA))
fwd <- step(null, direction = "forward", scope = full)
pred.fd <- predict(fwd, newdata = testA)
mean(abs(pred.fd - testA$logSP))
sqrt(mean((pred.fd - testA$logSP)^2))
OSR2(pred.fd, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.fd,type="l")

#blending:

ts = trainA$logSP-as.numeric(fit$residuals)
re = predict(best.rfe, newdata = trainA)
be <- predict(bte, newdata = trainA, n.trees = 20, interaction.depth = 2)
pcf = predict(pcr.model, newdata = trainA, ncomp = 10)
bl_train <- data.frame(logSP = trainA$logSP, be = be+ts,
                       pcf = pcf, re = re+ts,
                       fd = predict(fwd, newdata = trainA))
bl <- lm(logSP ~ . -1, data = bl_train)
summary(bl)

ts = preds$mean
re = predict(best.rfe, newdata = testA)
be <- predict(bte, newdata = testA, n.trees = 20, interaction.depth = 2)
bl_test <- data.frame(logSP = testA$logSP, be = be+ts,
                       pcf = predict(pcr.model, newdata = testA, ncomp = 10),
                       re = re+ts, be = be, fd = predict(fwd, newdata = testA))
pred.bl <- predict(bl, newdata = bl_test)
mean(abs(pred.bl - testA$logSP))
sqrt(mean((pred.bl - testA$logSP)^2))
OSR2(pred.bl, trainA$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.bl,type="l")

#################################### dataB
##linear model
fit <- lm(logSP~., data = trainB )
summary(fit)
pred.lm <- predict(fit, newdata = testB)
mean(abs(pred.lm - testB$logSP))
sqrt(mean((pred.lm - testB$logSP)^2))
OSR2(pred.lm, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red", type = "l")
lines(data$Date[107:127],pred.lm,type="l")
#linear with ridge 
drop <- c("logSP")
feature = as.matrix(trainB[,!(names(trainB) %in% drop)])
x.test = as.matrix(testB[,!(names(testB) %in% drop)])
set.seed(242)
cv <- cv.glmnet(feature, trainB$logSP, alpha = 0)
pred.r <- predict(cv, newx = x.test)
mean(abs(pred.r - testB$logSP))
sqrt(mean((pred.r - testB$logSP)^2))
OSR2(pred.r, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red", type = "l")
lines(data$Date[107:127],pred.r,type="l")
#lasso A
set.seed(242)
la <- cv.glmnet(feature, trainB$logSP, alpha = 1)
pred.l <- predict(la, newx = x.test)
mean(abs(pred.l - testB$logSP))
sqrt(mean((pred.l - testB$logSP)^2))
OSR2(pred.l, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.l,type="l")
coef(la, s=la$lambda.min)
## arima
fit <- auto.arima(trainB$logSP, stationary = FALSE, seasonal = FALSE)
preds <- forecast(fit, h=21)
mean(abs(preds$mean - testB$logSP))
sqrt(mean((preds$mean - testB$logSP)^2))
OSR2(preds$mean, trainB$logSP, testB$logSP)
plot(preds)
lines(data$logSP)

fit <- arima(trainB$logSP,order = c(1, 1, 1))
preds <- forecast(hr, h=21)
mean(abs(preds$mean - testB$logSP))
sqrt(mean((preds$mean - testB$logSP)^2))
OSR2(preds$mean, trainB$logSP, testB$logSP)

#random forest B
set.seed(242)
rf <- train(logSP ~ .,
            data = trainB,
            method = "rf",
            tuneGrid = data.frame(mtry=1:28),
            trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
            metric = "RMSE")
# RMSE or Rsquared doesn't matter actually -- both will be generated for regression problems
rf$results
rf
best.rf <- rf$finalModel
pred.rf <- predict(best.rf, newdata = testB)

ggplot(rf$results, aes(x = mtry, y = RMSE)) + geom_point(size = 3) + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

mean(abs(pred.rf - testB$logSP))
sqrt(mean((pred.rf - testB$logSP)^2))
OSR2(pred.rf, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.rf,type="l")


##boosting for A 
set.seed(242)
tGrid = expand.grid(n.trees = (1:8)*50000, interaction.depth = c(1,2,4,6,8,10),
                    shrinkage = 0.002, n.minobsinnode = 10)
train.boost <- train(logSP ~ .,
                     data = trainB,
                     method = "gbm",   ## gradient boosting machine 
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "RMSE",
                     distribution = "gaussian")
train.boost
bt.model <- train.boost$finalModel
pred.b <- predict(bt.model, newdata = testB, n.trees = 250000, interaction.depth = 4) # can use same model matrix

ggplot(train.boost$results, aes(x = n.trees, y = RMSE, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")

mean(abs(pred.b - testB$logSP))
sqrt(mean((pred.b - testB$logSP)^2))
OSR2(pred.b, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.b,type="l")

###########since the data is not stationary, the result of trees are bad, lets use residue of time series to do the trees again to see the result.
train_re <- trainB
train_re$logSP <- as.numeric(fit$residuals)
test_re <- testB
test_re$logSP <- testB$logSP-as.numeric(preds$mean)

#random forest
set.seed(242)
rfe <- train(logSP ~ .-SPLM-SPL2M,
             data = train_re,
             method = "rf",
             tuneGrid = data.frame(mtry=1:26),
             trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
             metric = "RMSE")
# RMSE or Rsquared doesn't matter actually -- both will be generated for regression problems
rfe$results
best.rfe <- rfe$finalModel
pred.rfe <- predict(best.rfe, newdata = test_re)

ggplot(rfe$results, aes(x = mtry, y = RMSE)) + geom_point(size = 3) + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

pred.rfe <- pred.rfe+as.numeric(preds$mean)
mean(abs(pred.rfe - testB$logSP))
sqrt(mean((pred.rfe - testB$logSP)^2))
OSR2(pred.rfe, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.rfe,type="l")


##boosting  
set.seed(242)
tGrid = expand.grid(n.trees = (1:70), interaction.depth = c(1,2,4,6,8,10),
                    shrinkage = 0.002, n.minobsinnode = 10)
train.boost <- train(logSP ~ .-SPLM-SPL2M,
                     data = train_re,
                     method = "gbm",   ## gradient boosting machine 
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "RMSE",
                     distribution = "gaussian")
train.boost
bte <- train.boost$finalModel
pred.be <- predict(bte, newdata = testA, n.trees = 2, interaction.depth = 1) # can use same model matrix

ggplot(train.boost$results, aes(x = n.trees, y = RMSE, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV RMSE") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")
pred.be <- pred.be+as.numeric(preds$mean)
mean(abs(pred.be - testB$logSP))
sqrt(mean((pred.be - testB$logSP)^2))
OSR2(pred.be, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.be,type="l")

#PCR 
set.seed(242)
pcr.model <- pcr(logSP~., data=trainB, scale = TRUE, validation = "CV")
summary(pcr.model)
validationplot(pcr.model)
pred.pcr <- predict(pcr.model, newdata = testB, ncomp = 13)
mean(abs(pred.pcr - testB$logSP))
sqrt(mean((pred.pcr - testB$logSP)^2))
OSR2(pred.pcr, trainB$logSP, testA$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.pcr,type="l")
#forward stepwise:
null <- lm(logSP~1, data = trainB)
full <- formula(lm(logSP~., data = trainB))
fwd <- step(null, direction = "forward", scope = full)
pred.fd <- predict(fwd, newdata = testB)
mean(abs(pred.fd - testB$logSP))
sqrt(mean((pred.fd - testB$logSP)^2))
OSR2(pred.fd, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.fd,type="l")

#### blending
ts = trainB$logSP-as.numeric(fit$residuals)
re = predict(best.rfe, newdata = trainB)
be = predict(bte, newdata = trainB, n.trees = 2, interaction.depth = 1)
pcf = predict(pcr.model, newdata = trainB, ncomp = 13)
bl_train <- data.frame(logSP = trainB$logSP, be = ts+be,
                       pcf = pcf, re = re+ts,
                       l = predict(la, newx = feature))
bl <- lm(logSP ~ . -1, data = bl_train)
summary(bl)

ts = preds$mean
re = predict(best.rfe, newdata = testB)
be <- predict(bte, newdata = testB, n.trees = 2, interaction.depth = 1)
bl_test <- data.frame(logSP = testB$logSP, be = be+ts,
                      pcf = predict(pcr.model, newdata = testB, ncomp = 13),
                      re = re+ts, be = be,l = predict(la, newx= x.test))
pred.bl <- predict(bl, newdata = bl_test)
mean(abs(pred.bl - testB$logSP))
sqrt(mean((pred.bl - testB$logSP)^2))
OSR2(pred.bl, trainB$logSP, testB$logSP)
plot(data$Date,data$logSP, col = "red")
lines(data$Date[107:127],pred.bl,type="l")

#### output assessment
#the best model is the forward stepwise from dataA
mean_squared_error <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  RMSE <- sqrt(mean((responses - predictions)^2))
  return(RMSE)
}

mean_absolute_error <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  MAE <- mean(abs(responses - predictions))
  return(MAE)
}

OS_R_squared <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  baseline <- data$baseline[index]
  SSE <- sum((responses - predictions)^2)
  SST <- sum((responses - baseline)^2)
  r2 <- 1 - SSE/SST
  return(r2)
}

all_metrics <- function(data, index) {
  rmse <- mean_squared_error(data, index)
  mae <- mean_absolute_error(data, index)
  OSR2 <- OS_R_squared(data, index)
  return(c(rmse, mae, OSR2))
}


###### Random Forests + Plots ###### 
FWD_test_set = data.frame(response = testB$logSP, prediction = pred.fd, baseline = mean(trainA$logSP))

# sanity check
all_metrics(FWD_test_set, 1:21)
sqrt(mean((pred.fd - testB$logSP)^2))
mean(abs(pred.fd - testB$logSP))
OSR2(pred.fd, trainA$logSP, testA$logSP)


# do bootstrap
set.seed(242)
FWD_boot <- boot(FWD_test_set, all_metrics, R = 10000)
FWD_boot


# make plots
FWD_boot_plot_results = data.frame(osr2estimates = FWD_boot$t[,3], delta = FWD_boot$t[,3] - FWD_boot$t0[3])

ggplot(FWD_boot_plot_results) + geom_histogram(aes(x = osr2estimates), binwidth = 0.005, color = "blue") + 
  ylab("Count") + xlab("Bootstrap OSR2 Estimate") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

# manual CI + plot

quantile(FWD_boot_plot_results$delta, c(0.025, 0.975))

ggplot(FWD_boot_plot_results) + geom_histogram(aes(x = delta), binwidth = 0.005, color = "green") + 
  ylab("Count") + xlab("Boot OSR2 - Test Set OSR2") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) +
  geom_vline(xintercept = -0.05964194) +
  geom_vline(xintercept = 0.05359415)

test_set_OSR2 = OSR2(pred.fd, trainA$logSP, testA$logSP)
lower = test_set_OSR2 - 0.05359415
upper = test_set_OSR2 - -0.05964194


# get confidence intervals (not manually)
boot.ci(FWD_boot, index = 1, type = "basic")
boot.ci(FWD_boot, index = 2, type = "basic")
boot.ci(FWD_boot, index = 3, type = "basic")
