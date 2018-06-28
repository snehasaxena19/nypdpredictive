
setwd("C:/Users/sneh/Desktop/Data mining/data files")
arrest <- read.csv("hope.csv")

head(arrest)
sapply(arrest,class)
sapply(arrest,typeof)


arrest$arstmade <- ifelse(arrest$arstmade =="Y",1,0)

#binning the variables
binned.timestop = cut(arrest$timestop, breaks = c(-Inf,500,1000,1500,2000,Inf), labels = 0:4)
binned.detailCM = cut(arrest$detailCM, breaks = c(-Inf,22,40,60,Inf), labels = 0:3)
binned.age = cut(arrest$age, breaks = c(-Inf,22,40,60,Inf), labels = 0:3)
arrest <- data.frame(arrest,binned.age,binned.detailCM,binned.timestop)


arrest<-na.omit(arrest)
arrest_sample <- arrest[sample(nrow(arrest), 1000), ]
train <- sample(nrow(arrest_sample), 0.7*nrow(arrest_sample))
arrest_train <- arrest_sample[train,]
arrest_validation <- arrest_sample[-train,]

#Linear Model
m1 = lm(arstmade~.-sb_other-cs_other-cs_drgtr-cs_lkout-cs_casng-rf_othsw-radio-pf_hcuff-addrpct-ac_time-ac_incid-ac_assoc-ac_evasv-cs_cloth-rf_vcact-cs_descr-cs_objcs-rf_attir-rf_vcrim-pf_other-pf_pepsp-pf_baton-pf_ptwep-pf_drwep-pf_grnd-pf_wall-pf_hands-riflshot-pistol-timestop-age-ht_feet-ht_inch-weight, data=arrest_train)
summary(m1)

cutoff<-0.5
predicted.probability = predict(m1, newdata = arrest_train)
Actual<- arrest_train$arstmade
Predicted <- ifelse( predicted.probability > cutoff, 1, 0)
confusion1 <- table(Actual, Predicted)
confusion1
specificity1 <- confusion1[1]/(confusion1[1]+confusion1[3])
specificity1
sensitivity1 <- confusion1[4]/(confusion1[4]+confusion1[2])
sensitivity1
predicted.probability.test <- predict(m1, type = "response", newdata = arrest_validation)
Predicted <- ifelse( predicted.probability.test > cutoff, 1, 0)
Actual <- arrest_validation$arstmade
confusion2<- table(Actual, Predicted)
confusion2
sensitivity2 <- confusion2[4]/(confusion2[4]+confusion2[2])
sensitivity2
specificity2 <- confusion2[1]/(confusion2[1]+confusion2[3])
specificity2
accuracy.validation1<-(confusion2[4]+ confusion2[1])/(confusion2[1]+confusion2[3]+confusion2[4]+confusion2[2])
accuracy.validation1

#Logistic Regression
m2 = glm(arstmade~.-sb_other-cs_other-cs_drgtr-cs_lkout-cs_casng-rf_othsw-radio-pf_hcuff-addrpct-ac_time-ac_incid-ac_assoc-ac_evasv-cs_cloth-rf_vcact-cs_descr-cs_objcs-rf_attir-rf_vcrim-pf_other-pf_pepsp-pf_baton-pf_ptwep-pf_drwep-pf_grnd-pf_wall-pf_hands-riflshot-pistol-timestop-age-ht_feet-ht_inch-weight, data=arrest_train, family = binomial)
summary(m2)
predicted.probability2 = predict(m2, newdata = arrest_validation, type = "response")
Actual<- arrest_validation$arstmade
Predicted2 <- ifelse( predicted.probability2 > cutoff, 1, 0)
confusion3 <- table(Actual, Predicted2)
confusion3
specificity3 <- confusion3[1]/(confusion3[1]+confusion3[3])
specificity3
sensitivity3 <- confusion3[4]/(confusion3[4]+confusion3[2])
sensitivity3
confusion3
accuracy.validation2<-(confusion3[4]+ confusion3[1])/(confusion3[1]+confusion3[3]+confusion3[4]+confusion3[2])
accuracy.validation2

#We now want to compare the predictive performance of the model on the training sample and the validation sample. 

cutoff <- seq(0, 1, length = 1000)
fpr <- numeric(1000)
tpr <- numeric(1000)
Actual <- arrest_train$arstmade

roc.table <- data.frame(Cutoff = cutoff, FPR = fpr,TPR = tpr)
for (i in 1:1000) {
  roc.table$FPR[i] <- sum(predicted.probability > cutoff[i] & Actual == 0)/sum(Actual == 0)
  roc.table$TPR[i] <- sum(predicted.probability > cutoff[i] & Actual == 1)/sum(Actual == 1)
}
plot(TPR ~ FPR, data = roc.table, type = "s",xlab="1 - Specificity",ylab="Sensitivity",col="blue")
abline(a = 0, b = 1, lty = 2,col="red")

cutoff <- seq(0, 1, length = 1000)
fpr <- numeric(1000)
tpr <- numeric(1000)
Actual <- arrest_validation$arstmade
for (i in 1:1000) {
  roc.table$FPR[i] <- sum(predicted.probability.test > cutoff[i] & Actual == 0)/sum(Actual == 0)
  roc.table$TPR[i] <- sum(predicted.probability.test > cutoff[i] & Actual == 1)/sum(Actual == 1)
}

lines(TPR~FPR,data = roc.table, type="s",col="green")


#Is there a difference in the prediction accuracy of two samples (the training data and the validation data)?

accuracy.train<-(confusion1[4]+ confusion1[1])/(confusion1[1]+confusion1[3]+confusion1[4]+confusion1[2])
accuracy.train
accuracy.validation<-(confusion2[4]+ confusion2[1])/(confusion2[1]+confusion2[3]+confusion2[4]+confusion2[2])
accuracy.validation



####################################


arrest_sample[,2:52] <- apply(arrest_sample[,c(2:52)],2,normalize)
View(arrest_sample)
library("caret")
library(e1071)
train1 <- createDataPartition(na.omit(arrest_sample$arstmade), p=0.7, list=FALSE)
#train1 <- sample(nrow(arrest_sample), 0.7*nrow(arrest_sample))
dftrain <- arrest_sample[train1,]
dfvalidation <- arrest_sample[-train1,]



library(class)
train_input <- as.matrix(dftrain)
train_output <- as.vector(dftrain[,1])
validate_input <- as.matrix(dfvalidation)
kmax <- 15
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)

#------------------------------------------------------------------------------

for (i in 1:kmax){
  prediction <- knn(train_input, train_input,train_output, k=i, prob=TRUE)
  prediction2 <- knn(train_input, validate_input,train_output, k=i,prob=TRUE)
  
  CM1 <- table(prediction, dftrain$arstmade)
  
  ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1)
  
  CM2 <- table(prediction2, dfvalidation$arstmade)
  ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2)
}
plot(c(1,kmax),c(0,0.3),type="n", xlab="k",ylab="Error Rate")
lines(ER1,col="red")
lines(ER2,col="blue")
legend(9, 0.1, c("Training","Validation"),lty=c(1,1), col=c("red","blue"))
z <- which.min(ER2)
cat("Minimum Validation Error rate k:", z)

prediction <- knn(train_input, train_input,train_output, k=z)
prediction2 <- knn(train_input, validate_input,train_output, k=z)

CM1 <- table( dftrain$arstmade,prediction)
CM2 <- table( dfvalidation$arstmade, prediction2)

CM1
CM2

#-------------------------------------------------------------------------------
library("caret")
par(mfrow=c(3,4))
for(i in 1:10)
{
  set.seed(i)
  inTrain <- createDataPartition(arrest_sample$arstmade, p=0.7, list=FALSE)
  
  dftrain <- data.frame(arrest_sample[inTrain,])
  dfvalidation <- data.frame(arrest_sample[-inTrain,])
  library(class)
  train_input <- as.matrix(dftrain[,-1])
  train_output <- as.vector(dftrain[,1])
  validate_input <- as.matrix(dfvalidation[,-1])
  kmax <- 15
  ER1 <- rep(0,kmax)
  ER2 <- rep(0,kmax)
  
  for (i in 1:kmax){
    prediction <- knn(train_input, train_input,train_output, k=i)
    prediction2 <- knn(train_input, validate_input,train_output, k=i)
    
    
    CM1 <- table(prediction, dftrain$arstmade)
    
    ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1)
    
    CM2 <- table(prediction2, dfvalidation$arstmade)
    ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2)
  }
  plot(c(1,kmax),c(0,0.3),type="n", xlab="k",ylab="Error Rate")
  lines(ER1,col="red")
  lines(ER2,col="blue")
  legend(9, 0.1, c("Training","Validation"),lty=c(1,1), col=c("red","blue"))
  z <- which.min(ER2)
  cat("Minimum Validation Error k:", z)
}





############################
#knn

arrest <- read.csv("hope.csv")
arrest$arstmade <- ifelse(arrest$arstmade =="Y",1,0)
arrest$arstmade <- as.numeric(arrest$arstmade)
arrest[arrest==""] <- NA
#arrest <- arrest[, colSums(is.na(arrest)) == 0]
arrest<-na.omit(arrest)



arrest$city <- as.character(arrest$city)
arrest$city <- sapply(arrest$city,switch,'MANHATTAN'=1,'QUEENS'=2,'BRONX'=3,'BROOKLYN'=4,'STATEN ISLAND'=5)
arrest$sex <- as.character(arrest$sex)
arrest$sex <- sapply(arrest$sex,switch,'M'= 1,'F ' = 4,'Z'= 3)
arrest$sex[sapply(arrest$sex, is.null)] <- 4
arrest$sex <- as.numeric(arrest$sex)

arrest$build <- as.character(arrest$build)
arrest$build <- sapply(arrest$build,switch,'H'=1,'M'=2,'T'=3,'U'=4,'Z'=5)

arrest$trhsloc <- as.character(arrest$trhsloc)
arrest$trhsloc <- sapply(arrest$trhsloc,switch,'H'=1,'P'=2,'T'=3)

arrest$typeofid <- as.character(arrest$typeofid)
arrest$typeofid <- sapply(arrest$typeofid,switch,'O'=1,'P'=2,'R'=3,'V'=4)

arrest$othpers <- ifelse(arrest$othpers=="Y",1,0)

arrest$sumissue <- ifelse(arrest$sumissue=="Y",1,0)
arrest$offunif <- ifelse(arrest$offunif=="Y",1,0)
arrest$frisked <- ifelse(arrest$frisked=="Y",1,0)
arrest$searched <- ifelse(arrest$searched=="Y",1,0)
arrest$adtlrept <- ifelse(arrest$adtlrept=="Y",1,0)
arrest$pistol <- ifelse(arrest$pistol=="Y",1,0)
arrest$riflshot <- ifelse(arrest$riflshot=="Y",1,0)
arrest$pf_hands <- ifelse(arrest$pf_hands=="Y",1,0)
arrest$pf_wall <- ifelse(arrest$pf_wall=="Y",1,0)
arrest$pf_grnd <- ifelse(arrest$pf_grnd=="Y",1,0)
arrest$pf_drwep <-ifelse(arrest$pf_drwep=="Y",1,0)
arrest$pf_ptwep <- ifelse(arrest$pf_ptwep=="Y",1,0)
arrest$pf_baton <- ifelse(arrest$pf_baton=="Y",1,0)
arrest$pf_hcuff <- ifelse(arrest$pf_hcuff=="Y",1,0)
arrest$pf_pepsp <- ifelse(arrest$pf_pepsp=="Y",1,0)
arrest$pf_other <- ifelse(arrest$pf_other=="Y",1,0)
arrest$radio <- ifelse(arrest$radio=="Y",1,0)
arrest$rf_vcrim <- ifelse(arrest$rf_vcrim=="Y",1,0)
arrest$rf_othsw <- ifelse(arrest$rf_othsw=="Y",1,0)
arrest$rf_attir <- ifelse(arrest$rf_attir=="Y",1,0)
arrest$cs_objcs <- ifelse(arrest$cs_objcs=="Y",1,0)
arrest$cs_descr <- ifelse(arrest$cs_descr=="Y",1,0)
arrest$cs_casng <- ifelse(arrest$cs_casng=="Y",1,0)
arrest$cs_lkout <- ifelse(arrest$cs_lkout=="Y",1,0)
arrest$cs_cloth <- ifelse(arrest$cs_cloth=="Y",1,0)
arrest$rf_vcact <- ifelse(arrest$rf_vcact=="Y",1,0)
arrest$cs_drgtr <- ifelse(arrest$cs_drgtr=="Y",1,0)
arrest$ac_evasv <- ifelse(arrest$ac_evasv=="Y",1,0)
arrest$ac_assoc <- ifelse(arrest$ac_assoc=="Y",1,0)
arrest$cs_other <- ifelse(arrest$cs_other=="Y",1,0)
arrest$ac_incid <- ifelse(arrest$ac_incid=="Y",1,0)
arrest$ac_time <- ifelse(arrest$ac_time=="Y",1,0)
arrest$sb_other <- ifelse(arrest$sb_other=="Y",1,0)



arrest_sample <- arrest[sample(nrow(arrest), 1000), ]


str(arrest_sample)


arrest_sample<- arrest_sample[,-c(16,22,24)]
normalize <- function(x){ 
  m <- mean(x) 
  std <- sd(x) 
  (x - m)/(std) 
} 




arrest_sample[,2:52] <- apply(arrest_sample[,c(2:52)],2,normalize)
View(arrest_sample)
library("caret")
library(e1071)
train1 <- createDataPartition(na.omit(arrest_sample$arstmade), p=0.7, list=FALSE)
#train1 <- sample(nrow(arrest_sample), 0.7*nrow(arrest_sample))
dftrain <- arrest_sample[train1,]
dfvalidation <- arrest_sample[-train1,]



library(class)
train_input <- as.matrix(dftrain)
train_output <- as.vector(dftrain[,1])
validate_input <- as.matrix(dfvalidation)
kmax <- 15
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)


########

###Comparing the Models
#############################
arrest1 <- read.csv("hope.csv")
arrest1$arstmade <- ifelse(arrest1$arstmade =="Y",1,0)
arrest1[arrest1==""] <- NA
#arrest1 <- arrest1[, colSums(is.na(arrest1)) == 0]
arrest1<-na.omit(arrest1)

arrest1$city <- as.character(arrest1$city)
arrest1$city <- as.factor(sapply(arrest1$city,switch,'MANHATTAN'=1,'QUEENS'=2,'BRONX'=3,'BROOKLYN'=4,'STATEN ISLAND'=5))
arrest1$sex <- as.character(arrest1$sex)
arrest1$sex <- sapply(arrest1$sex,switch,'M'= 1,'F ' = 4,'Z'= 3)
arrest1$sex[sapply(arrest1$sex, is.null)] <- 4
arrest1$sex <- as.factor(as.numeric(arrest1$sex))
arrest1$build <- as.character(arrest1$build)
arrest1$build <- as.factor(sapply(arrest1$build,switch,'H'=1,'M'=2,'T'=3,'U'=4,'Z'=5))
arrest1$trhsloc <- as.character(arrest1$trhsloc)
arrest1$trhsloc <- as.factor(sapply(arrest1$trhsloc,switch,'H'=1,'P'=2,'T'=3))
arrest1$typeofid <- as.character(arrest1$typeofid)
arrest1$typeofid <- as.factor(sapply(arrest1$typeofid,switch,'O'=1,'P'=2,'R'=3,'V'=4))
arrest1$othpers <- as.factor(ifelse(arrest1$othpers=="Y",1,0))
arrest1$sumissue <- as.factor(ifelse(arrest1$sumissue=="Y",1,0))
arrest1$offunif <- as.factor(ifelse(arrest1$offunif=="Y",1,0))
arrest1$frisked <- as.factor(ifelse(arrest1$frisked=="Y",1,0))
arrest1$searched <- as.factor(ifelse(arrest1$searched=="Y",1,0))
arrest1$adtlrept <- as.factor(ifelse(arrest1$adtlrept=="Y",1,0))
arrest1$pistol <- as.factor(ifelse(arrest1$pistol=="Y",1,0))
arrest1$riflshot <- as.factor(ifelse(arrest1$riflshot=="Y",1,0))
arrest1$pf_hands <- as.factor(ifelse(arrest1$pf_hands=="Y",1,0))
arrest1$pf_wall <- as.factor(ifelse(arrest1$pf_wall=="Y",1,0))
arrest1$pf_grnd <- as.factor(ifelse(arrest1$pf_grnd=="Y",1,0))
arrest1$pf_drwep <-as.factor(ifelse(arrest1$pf_drwep=="Y",1,0))
arrest1$pf_ptwep <- as.factor(ifelse(arrest1$pf_ptwep=="Y",1,0))
arrest1$pf_baton <- as.factor(ifelse(arrest1$pf_baton=="Y",1,0))
arrest1$pf_hcuff <- as.factor(ifelse(arrest1$pf_hcuff=="Y",1,0))
arrest1$pf_pepsp <- as.factor(ifelse(arrest1$pf_pepsp=="Y",1,0))
arrest1$pf_other <- as.factor(ifelse(arrest1$pf_other=="Y",1,0))
arrest1$radio <- as.factor(ifelse(arrest1$radio=="Y",1,0))
arrest1$rf_vcrim <- as.factor(ifelse(arrest1$rf_vcrim=="Y",1,0))
arrest1$rf_othsw <- as.factor(ifelse(arrest1$rf_othsw=="Y",1,0))
arrest1$rf_attir <- as.factor(ifelse(arrest1$rf_attir=="Y",1,0))
arrest1$cs_objcs <- as.factor(ifelse(arrest1$cs_objcs=="Y",1,0))
arrest1$cs_descr <- as.factor(ifelse(arrest1$cs_descr=="Y",1,0))
arrest1$cs_casng <- as.factor(ifelse(arrest1$cs_casng=="Y",1,0))
arrest1$cs_lkout <- as.factor(ifelse(arrest1$cs_lkout=="Y",1,0))
arrest1$cs_cloth <- as.factor(ifelse(arrest1$cs_cloth=="Y",1,0))
arrest1$rf_vcact <- as.factor(ifelse(arrest1$rf_vcact=="Y",1,0))
arrest1$cs_drgtr <- as.factor(ifelse(arrest1$cs_drgtr=="Y",1,0))
arrest1$ac_evasv <- as.factor(ifelse(arrest1$ac_evasv=="Y",1,0))
arrest1$ac_assoc <- as.factor(ifelse(arrest1$ac_assoc=="Y",1,0))
arrest1$cs_other <- as.factor(ifelse(arrest1$cs_other=="Y",1,0))
arrest1$ac_incid <- as.factor(ifelse(arrest1$ac_incid=="Y",1,0))
arrest1$ac_time <- as.factor(ifelse(arrest1$ac_time=="Y",1,0))
arrest1$sb_other <- as.factor(ifelse(arrest1$sb_other=="Y",1,0))



binned.timestop = cut(arrest1$timestop, breaks = c(-Inf,500,1000,1500,2000,Inf), labels = 0:4)
binned.detailCM = cut(arrest1$detailCM, breaks = c(-Inf,22,40,60,Inf), labels = 0:3)
binned.age = cut(arrest1$age, breaks = c(-Inf,22,40,60,Inf), labels = 0:3)
binned.addrpct = cut(arrest1$addrpct, breaks = c(-Inf,20,40,60,80,Inf), labels = 0:4)
binned.weight = cut(arrest1$weight, breaks = c(-Inf,70,140,210,280,Inf), labels = 0:4)
binned.height = cut(arrest1$ht_feet, breaks = c(-Inf,3,4,5,6,Inf), labels = 0:4)
binned.revcmd = cut(arrest1$revcmd, breaks = c(-Inf,200,400,600,Inf), labels = 0:3)
binned.repcmd = cut(arrest1$repcmd, breaks = c(-Inf,200,400,600,Inf), labels = 0:3)
binned.perobs = cut(arrest1$perobs, breaks = c(-Inf,20,40,60,Inf), labels = 0:3)
binned.sernum = cut(arrest1$ser_num, breaks = c(-Inf,500,1000,1500,Inf), labels = 0:3)
binned.pct = cut(arrest1$pct, breaks = c(-Inf,30,60,90,Inf), labels = 0:3)

arrest1<- arrest1[,-c(2,3,4,6,43,44,46,47,48,49,52,53,54,55)]

arrest1$binned.timestop <- binned.timestop
arrest1$binned.detailCM <- binned.detailCM
arrest1$binned.age <- binned.age
arrest1$binned.addrpct <- binned.addrpct
arrest1$binned.weight <- binned.weight
arrest1$binned.height <- binned.height
arrest1$binned.revcmd <- binned.revcmd
arrest1$binned.repcmd <- binned.repcmd
arrest1$binned.perobs <- binned.perobs
arrest1$binned.sernum <- binned.sernum
arrest1$binned.pct <- binned.pct







arrest_sample1 <- arrest1[sample(nrow(arrest1), 1000), ]
library(e1071)
train1 <- sample(nrow(arrest_sample1), 0.7*nrow(arrest_sample1))
arrest_train1 <- arrest_sample1[train1,]
arrest_validation1 <- arrest_sample1[-train1,]

arrest_train1


model <- naiveBayes(as.factor(arstmade)~., data=arrest_train1)
model
#View(arrest_validation1)
prediction <- predict(model,newdata = arrest_validation1[,-1])
prediction
table(arrest_validation1$arstmade,prediction,dnn=list('actual','predicted'))
model$apriori


# bayes

par(mfrow=c(1,1))
cutoff <- seq(0, 1, length = 100)
fpr <- numeric(100)
tpr <- numeric(100)
roc.table <- data.frame(Cutoff = cutoff,FPR = fpr,TPR = tpr)

p_prob <- predict(model, arrest_validation1[-1],type="raw")[,2]

for (i in 1:100) {
  roc.table$FPR[i] <- sum(p_prob > cutoff[i] & arrest_validation1$arstmade == 0)/sum(arrest_validation1$arstmade == 0)
  roc.table$TPR[i] <- sum(p_prob > cutoff[i] & arrest_validation1$arstmade == 1)/sum(arrest_validation1$arstmade == 1)
}

plot(roc.table$TPR ~ roc.table$FPR, data = roc.table, type = "s", main="ROC Curve", xlab="1-Specificity",ylab="Sensitivity",col="blue")


#KNN

cutoff <- seq(0, 1, length = 100)
cutoff
fpr <- numeric(100)
tpr <- numeric(100)
roc.table3 <- data.frame(Cutoff = cutoff, FPR = fpr,TPR = tpr)
Actual= as.factor(arrest_validation1$arstmade)
Actual
prediction5 <- knn(train_input, validate_input,train_output, k=5, prob = T)
prediction5
pred_prob = attr(prediction5, "prob")
pred_prob = ifelse(prediction5==1, pred_prob, 1-pred_prob)

pred_prob
for (i in 1:100) {
  roc.table3$FPR[i] <- sum(pred_prob >= cutoff[i] & Actual == 0)/sum(Actual == 0)
  roc.table3$TPR[i] <- sum(pred_prob >= cutoff[i] & Actual == 1)/sum(Actual == 1)
}
#nrow(roc.table3$FPR)
lines(TPR ~ FPR, data = roc.table3, col="black")
abline(a = 0, b = 1, lty = 2,col="black")
legend(0.6, 0.7, c("logistic","NB","KNN","Linear"),lty=c(1,1), col=c("red","blue","black","purple"))

#logistic

par(mfrow=c(1,1))
Performance= glm(dfvalidation$arstmade~.-sb_other-cs_other-cs_drgtr-cs_lkout-cs_casng-rf_othsw-radio-pf_hcuff-addrpct-ac_time-ac_incid-ac_assoc-ac_evasv-cs_cloth-rf_vcact-cs_descr-cs_objcs-rf_attir-rf_vcrim-pf_other-pf_ptwep-pf_drwep-pf_grnd-pf_wall-pf_hands-pistol-timestop-age-ht_feet-ht_inch-weight ,data = dfvalidation, family = binomial)
Performance

cutoff <- 0.5

predicted.probability.test <- predict(Performance, type = "response")
Predicted <- ifelse( predicted.probability.test > cutoff, 1, 0)

cutoff <- seq(0, 1, length = 100)
fpr <- numeric(100)
tpr <- numeric(100)
Actual<- dfvalidation$arstmade
roc.table2 <- data.frame(Cutoff = cutoff, FPR = fpr,TPR = tpr)
for (i in 1:100) {
  roc.table2$FPR[i] <- sum(predicted.probability.test > cutoff[i] & Actual == 0)/sum(Actual == 0)
  roc.table2$TPR[i] <- sum(predicted.probability.test > cutoff[i] & Actual == 1)/sum(Actual == 1)
}

lines(TPR ~ FPR, data = roc.table,col="red")

#linear



setwd("C:/Users/sarth/Desktop/Data mining/data files")
arrest <- read.csv("hope.csv")

head(arrest)
sapply(arrest,class)
sapply(arrest,typeof)


arrest$arstmade <- ifelse(arrest$arstmade =="Y",1,0)

arrest<-na.omit(arrest)
arrest_sample <- arrest[sample(nrow(arrest), 1000), ]
train <- sample(nrow(arrest_sample), 0.7*nrow(arrest_sample))
arrest_train <- arrest_sample[train,]
arrest_validation <- arrest_sample[-train,]
m1 = lm(arstmade~.-sb_other-cs_other-cs_drgtr-cs_lkout-cs_casng-rf_othsw-radio-pf_hcuff-addrpct-ac_time-ac_incid-ac_assoc-ac_evasv-cs_cloth-rf_vcact-cs_descr-cs_objcs-rf_attir-rf_vcrim-pf_other-pf_pepsp-pf_baton-pf_ptwep-pf_drwep-pf_grnd-pf_wall-pf_hands-riflshot-pistol-timestop-age-ht_feet-ht_inch-weight, data=arrest_train)
summary(m1)

cutoff<-0.5
predicted.probability = predict(m1, newdata = arrest_train)
Actual<- arrest_train$arstmade
Predicted <- ifelse( predicted.probability > cutoff, 1, 0)
confusion1 <- table(Actual, Predicted)
confusion1
specificity1 <- confusion1[1]/(confusion1[1]+confusion1[3])
specificity1
sensitivity1 <- confusion1[4]/(confusion1[4]+confusion1[2])
sensitivity1
predicted.probability.test <- predict(m1, type = "response", newdata = arrest_validation)
Predicted <- ifelse( predicted.probability.test > cutoff, 1, 0)
Actual <- arrest_validation$arstmade
confusion2<- table(Actual, Predicted)
confusion2
sensitivity2 <- confusion2[4]/(confusion2[4]+confusion2[2])
sensitivity2
specificity2 <- confusion2[1]/(confusion2[1]+confusion2[3])
specificity2


#We now want to compare the predictive performance of the model on the training sample and the validation sample. 

cutoff <- seq(0, 1, length = 1000)
fpr <- numeric(1000)
tpr <- numeric(1000)
abline(a = 0, b = 1, lty = 2,col="red")

Actual <- arrest_validation$arstmade
roc.table <- data.frame(Cutoff = cutoff, FPR = fpr,TPR = tpr)
for (i in 1:1000) {
  roc.table$FPR[i] <- sum(predicted.probability.test > cutoff[i] & Actual == 0)/sum(Actual == 0)
  
  roc.table$TPR[i] <- sum(predicted.probability.test > cutoff[i] & Actual == 1)/sum(Actual == 1)
}

lines(TPR~FPR,data = roc.table, type="s",col="purple")



#Random forest 
library(caret)
library(tree)
attach(arrest)
set.seed(12345)
tree_train= tree(as.factor(arstmade)~.,arrest_train)
summary(tree_train)
plot(tree_train)
text(tree_train,pretty = 0)
tree_pred = predict(tree_train,arrest_validation, type = "class")
rfull=table(tree_pred,arrest_validation$arstmade)

Accuracy_rfull= (rfull[4]+rfull[1])/(rfull[1]+rfull[3]+rfull[4]+rfull[2])
Accuracy_rfull

Sensitivity_rfull = rfull[4]/(rfull[4]+rfull[2])
Specificity_rfull = rfull[1]/(rfull[1]+rfull[3])

Sensitivity_rfull
Specificity_rfull

cv.train= cv.tree(tree_train, FUN = prune.misclass,K=10)
cv.train
plot(cv.train$size,cv.train$dev, type="b")


############################ 
# Plot the best pruned tree
prune.arrest=prune.tree(tree_train,best=9)
plot(prune.arrest)
text(prune.arrest,pretty=0)

summary(prune.arrest)

tree_pred2 = predict(prune.arrest,arrest_validation, type = "class")
rf = table(tree_pred2,arrest_validation$arstmade)
Accuracy_rf= (rf[4]+rf[1])/(rf[1]+rf[3]+rf[4]+rf[2])
Sensitivity_rf = rf[4]/(rf[4]+rf[2])
Specificity_rf = rf[1]/(rf[1]+rf[3])
Accuracy_rf
Sensitivity_rf
Specificity_rf

#########################
## Clustering on arrest made

arrest <- read.csv("hope.csv")
arrest$arstmade <- ifelse(arrest$arstmade =="Y",1,0)
arrest$arstmade <- as.numeric(arrest$arstmade)
arrest[arrest==""] <- NA
#arrest <- arrest[, colSums(is.na(arrest)) == 0]
arrest<-na.omit(arrest)



arrest$city <- as.character(arrest$city)
arrest$city <- sapply(arrest$city,switch,'MANHATTAN'=1,'QUEENS'=2,'BRONX'=3,'BROOKLYN'=4,'STATEN ISLAND'=5)
arrest$sex <- as.character(arrest$sex)
arrest$sex <- sapply(arrest$sex,switch,'M'= 1,'F ' = 4,'Z'= 3)
arrest$sex[sapply(arrest$sex, is.null)] <- 4
arrest$sex <- as.numeric(arrest$sex)

arrest$build <- as.character(arrest$build)
arrest$build <- sapply(arrest$build,switch,'H'=1,'M'=2,'T'=3,'U'=4,'Z'=5)

arrest$trhsloc <- as.character(arrest$trhsloc)
arrest$trhsloc <- sapply(arrest$trhsloc,switch,'H'=1,'P'=2,'T'=3)

arrest$typeofid <- as.character(arrest$typeofid)
arrest$typeofid <- sapply(arrest$typeofid,switch,'O'=1,'P'=2,'R'=3,'V'=4)

arrest$othpers <- ifelse(arrest$othpers=="Y",1,0)

arrest$sumissue <- ifelse(arrest$sumissue=="Y",1,0)
arrest$offunif <- ifelse(arrest$offunif=="Y",1,0)
arrest$frisked <- ifelse(arrest$frisked=="Y",1,0)
arrest$searched <- ifelse(arrest$searched=="Y",1,0)
arrest$adtlrept <- ifelse(arrest$adtlrept=="Y",1,0)
arrest$pistol <- ifelse(arrest$pistol=="Y",1,0)
arrest$riflshot <- ifelse(arrest$riflshot=="Y",1,0)
arrest$pf_hands <- ifelse(arrest$pf_hands=="Y",1,0)
arrest$pf_wall <- ifelse(arrest$pf_wall=="Y",1,0)
arrest$pf_grnd <- ifelse(arrest$pf_grnd=="Y",1,0)
arrest$pf_drwep <-ifelse(arrest$pf_drwep=="Y",1,0)
arrest$pf_ptwep <- ifelse(arrest$pf_ptwep=="Y",1,0)
arrest$pf_baton <- ifelse(arrest$pf_baton=="Y",1,0)
arrest$pf_hcuff <- ifelse(arrest$pf_hcuff=="Y",1,0)
arrest$pf_pepsp <- ifelse(arrest$pf_pepsp=="Y",1,0)
arrest$pf_other <- ifelse(arrest$pf_other=="Y",1,0)
arrest$radio <- ifelse(arrest$radio=="Y",1,0)
arrest$rf_vcrim <- ifelse(arrest$rf_vcrim=="Y",1,0)
arrest$rf_othsw <- ifelse(arrest$rf_othsw=="Y",1,0)
arrest$rf_attir <- ifelse(arrest$rf_attir=="Y",1,0)
arrest$cs_objcs <- ifelse(arrest$cs_objcs=="Y",1,0)
arrest$cs_descr <- ifelse(arrest$cs_descr=="Y",1,0)
arrest$cs_casng <- ifelse(arrest$cs_casng=="Y",1,0)
arrest$cs_lkout <- ifelse(arrest$cs_lkout=="Y",1,0)
arrest$cs_cloth <- ifelse(arrest$cs_cloth=="Y",1,0)
arrest$rf_vcact <- ifelse(arrest$rf_vcact=="Y",1,0)
arrest$cs_drgtr <- ifelse(arrest$cs_drgtr=="Y",1,0)
arrest$ac_evasv <- ifelse(arrest$ac_evasv=="Y",1,0)
arrest$ac_assoc <- ifelse(arrest$ac_assoc=="Y",1,0)
arrest$cs_other <- ifelse(arrest$cs_other=="Y",1,0)
arrest$ac_incid <- ifelse(arrest$ac_incid=="Y",1,0)
arrest$ac_time <- ifelse(arrest$ac_time=="Y",1,0)
arrest$sb_other <- ifelse(arrest$sb_other=="Y",1,0)



arrest_sample <- arrest[sample(nrow(arrest), 1000), ]


str(arrest_sample)


arrest_sample<- arrest_sample[,-c(16,22,24)]
normalize <- function(x){ 
  m <- mean(x) 
  std <- sd(x) 
  (x - m)/(std) 
} 


arrest_sample[,2:52] <- apply(arrest_sample[,c(2:52)],2,normalize)


kc <- kmeans(arrest_sample, 2) 
kc
table(arrest_sample$arstmade, kc$cluster)
plot(arrest_sample[,1], col=kc$cluster)
points(kc$centers[,1], col=1:3, pch=8, cex=2)
#############################################33
## Association rules

arrest <- read.csv("hope.csv")
arrest$arstmade <- ifelse(arrest$arstmade =="Y",1,0)
arrest[arrest==""] <- NA
#arrest1 <- arrest1[, colSums(is.na(arrest1)) == 0]
arrest<-na.omit(arrest)


arrest$city <- as.character(arrest$city)
arrest$city <- sapply(arrest$city,switch,'MANHATTAN'=1,'QUEENS'=2,'BRONX'=3,'BROOKLYN'=4,'STATEN ISLAND'=5)
arrest$sex <- as.character(arrest$sex)
arrest$sex <- sapply(arrest$sex,switch,'M'= 1,'F ' = 4,'Z'= 3)
arrest$sex[sapply(arrest$sex, is.null)] <- 4
arrest$sex <- as.numeric(arrest$sex)

arrest$build <- as.character(arrest$build)
arrest$build <- sapply(arrest$build,switch,'H'=1,'M'=2,'T'=3,'U'=4,'Z'=5)

arrest$trhsloc <- as.character(arrest$trhsloc)
arrest$trhsloc <- sapply(arrest$trhsloc,switch,'H'=1,'P'=2,'T'=3)

arrest$typeofid <- as.character(arrest$typeofid)
arrest$typeofid <- sapply(arrest$typeofid,switch,'O'=1,'P'=2,'R'=3,'V'=4)

arrest$othpers <- ifelse(arrest$othpers=="Y",1,0)

arrest$sumissue <- ifelse(arrest$sumissue=="Y",1,0)
arrest$offunif <- ifelse(arrest$offunif=="Y",1,0)
arrest$frisked <- ifelse(arrest$frisked=="Y",1,0)
arrest$searched <- ifelse(arrest$searched=="Y",1,0)
arrest$adtlrept <- ifelse(arrest$adtlrept=="Y",1,0)
arrest$pistol <- ifelse(arrest$pistol=="Y",1,0)
arrest$riflshot <- ifelse(arrest$riflshot=="Y",1,0)
arrest$pf_hands <- ifelse(arrest$pf_hands=="Y",1,0)
arrest$pf_wall <- ifelse(arrest$pf_wall=="Y",1,0)
arrest$pf_grnd <- ifelse(arrest$pf_grnd=="Y",1,0)
arrest$pf_drwep <-ifelse(arrest$pf_drwep=="Y",1,0)
arrest$pf_ptwep <- ifelse(arrest$pf_ptwep=="Y",1,0)
arrest$pf_baton <- ifelse(arrest$pf_baton=="Y",1,0)
arrest$pf_hcuff <- ifelse(arrest$pf_hcuff=="Y",1,0)
arrest$pf_pepsp <- ifelse(arrest$pf_pepsp=="Y",1,0)
arrest$pf_other <- ifelse(arrest$pf_other=="Y",1,0)
arrest$radio <- ifelse(arrest$radio=="Y",1,0)
arrest$rf_vcrim <- ifelse(arrest$rf_vcrim=="Y",1,0)
arrest$rf_othsw <- ifelse(arrest$rf_othsw=="Y",1,0)
arrest$rf_attir <- ifelse(arrest$rf_attir=="Y",1,0)
arrest$cs_objcs <- ifelse(arrest$cs_objcs=="Y",1,0)
arrest$cs_descr <- ifelse(arrest$cs_descr=="Y",1,0)
arrest$cs_casng <- ifelse(arrest$cs_casng=="Y",1,0)
arrest$cs_lkout <- ifelse(arrest$cs_lkout=="Y",1,0)
arrest$cs_cloth <- ifelse(arrest$cs_cloth=="Y",1,0)
arrest$rf_vcact <- ifelse(arrest$rf_vcact=="Y",1,0)
arrest$cs_drgtr <- ifelse(arrest$cs_drgtr=="Y",1,0)
arrest$ac_evasv <- ifelse(arrest$ac_evasv=="Y",1,0)
arrest$ac_assoc <- ifelse(arrest$ac_assoc=="Y",1,0)
arrest$cs_other <- ifelse(arrest$cs_other=="Y",1,0)
arrest$ac_incid <- ifelse(arrest$ac_incid=="Y",1,0)
arrest$ac_time <- ifelse(arrest$ac_time=="Y",1,0)
arrest$sb_other <- ifelse(arrest$sb_other=="Y",1,0)

arrest_sample <- arrest[sample(nrow(arrest), 10000), ]

bank_scaled <- scale(arrest)


k_model = kmeans(arrest_sample,2,nstart=20) 
summary(k_model)

dist(k_model$centers)
bank_scaled$Cluster <- k_model$cluster
table(arrest_sample$arstmade, bank_scaled$Cluster)

hist(bank_scaled$Cluster,col='light blue')

aggregate(arrest_sample$arstmade, by=list(bank_scaled$Cluster), FUN=mean, simplify=TRUE)




arrest1 <- read.csv("hope.csv")
arrest1$arstmade <- ifelse(arrest1$arstmade =="Y",1,0)
arrest1[arrest1==""] <- NA
#arrest1 <- arrest1[, colSums(is.na(arrest1)) == 0]
arrest1<-na.omit(arrest1)


arrest1$city <- as.character(arrest1$city)
arrest1$city <- as.factor(sapply(arrest1$city,switch,'MANHATTAN'=1,'QUEENS'=2,'BRONX'=3,'BROOKLYN'=4,'STATEN ISLAND'=5))
arrest1$sex <- as.character(arrest1$sex)
arrest1$sex <- sapply(arrest1$sex,switch,'M'= 1,'F ' = 4,'Z'= 3)
arrest1$sex[sapply(arrest1$sex, is.null)] <- 4
arrest1$sex <- as.factor(as.numeric(arrest1$sex))

arrest1$build <- as.character(arrest1$build)
arrest1$build <- as.factor(sapply(arrest1$build,switch,'H'=1,'M'=2,'T'=3,'U'=4,'Z'=5))

arrest1$trhsloc <- as.character(arrest1$trhsloc)
arrest1$trhsloc <- as.factor(sapply(arrest1$trhsloc,switch,'H'=1,'P'=2,'T'=3))

arrest1$typeofid <- as.character(arrest1$typeofid)
arrest1$typeofid <- as.factor(sapply(arrest1$typeofid,switch,'O'=1,'P'=2,'R'=3,'V'=4))

arrest1$othpers <- as.factor(ifelse(arrest1$othpers=="Y",1,0))

arrest1$sumissue <- as.factor(ifelse(arrest1$sumissue=="Y",1,0))
arrest1$offunif <- as.factor(ifelse(arrest1$offunif=="Y",1,0))
arrest1$frisked <- as.factor(ifelse(arrest1$frisked=="Y",1,0))
arrest1$searched <- as.factor(ifelse(arrest1$searched=="Y",1,0))
arrest1$adtlrept <- as.factor(ifelse(arrest1$adtlrept=="Y",1,0))
arrest1$pistol <- as.factor(ifelse(arrest1$pistol=="Y",1,0))
arrest1$riflshot <- as.factor(ifelse(arrest1$riflshot=="Y",1,0))
arrest1$pf_hands <- as.factor(ifelse(arrest1$pf_hands=="Y",1,0))
arrest1$pf_wall <- as.factor(ifelse(arrest1$pf_wall=="Y",1,0))
arrest1$pf_grnd <- as.factor(ifelse(arrest1$pf_grnd=="Y",1,0))
arrest1$pf_drwep <-as.factor(ifelse(arrest1$pf_drwep=="Y",1,0))
arrest1$pf_ptwep <- as.factor(ifelse(arrest1$pf_ptwep=="Y",1,0))
arrest1$pf_baton <- as.factor(ifelse(arrest1$pf_baton=="Y",1,0))
arrest1$pf_hcuff <- as.factor(ifelse(arrest1$pf_hcuff=="Y",1,0))
arrest1$pf_pepsp <- as.factor(ifelse(arrest1$pf_pepsp=="Y",1,0))
arrest1$pf_other <- as.factor(ifelse(arrest1$pf_other=="Y",1,0))
arrest1$radio <- as.factor(ifelse(arrest1$radio=="Y",1,0))
arrest1$rf_vcrim <- as.factor(ifelse(arrest1$rf_vcrim=="Y",1,0))
arrest1$rf_othsw <- as.factor(ifelse(arrest1$rf_othsw=="Y",1,0))
arrest1$rf_attir <- as.factor(ifelse(arrest1$rf_attir=="Y",1,0))
arrest1$cs_objcs <- as.factor(ifelse(arrest1$cs_objcs=="Y",1,0))
arrest1$cs_descr <- as.factor(ifelse(arrest1$cs_descr=="Y",1,0))
arrest1$cs_casng <- as.factor(ifelse(arrest1$cs_casng=="Y",1,0))
arrest1$cs_lkout <- as.factor(ifelse(arrest1$cs_lkout=="Y",1,0))
arrest1$cs_cloth <- as.factor(ifelse(arrest1$cs_cloth=="Y",1,0))
arrest1$rf_vcact <- as.factor(ifelse(arrest1$rf_vcact=="Y",1,0))
arrest1$cs_drgtr <- as.factor(ifelse(arrest1$cs_drgtr=="Y",1,0))
arrest1$ac_evasv <- as.factor(ifelse(arrest1$ac_evasv=="Y",1,0))
arrest1$ac_assoc <- as.factor(ifelse(arrest1$ac_assoc=="Y",1,0))
arrest1$cs_other <- as.factor(ifelse(arrest1$cs_other=="Y",1,0))
arrest1$ac_incid <- as.factor(ifelse(arrest1$ac_incid=="Y",1,0))
arrest1$ac_time <- as.factor(ifelse(arrest1$ac_time=="Y",1,0))
arrest1$sb_other <- as.factor(ifelse(arrest1$sb_other=="Y",1,0))

binned.timestop = cut(arrest1$timestop, breaks = c(-Inf,500,1000,1500,2000,Inf), labels = 0:4)
binned.detailCM = cut(arrest1$detailCM, breaks = c(-Inf,22,40,60,Inf), labels = 0:3)
binned.age = cut(arrest1$age, breaks = c(-Inf,22,40,60,Inf), labels = 0:3)
binned.addrpct = cut(arrest1$addrpct, breaks = c(-Inf,20,40,60,80,Inf), labels = 0:4)
binned.weight = cut(arrest1$weight, breaks = c(-Inf,70,140,210,280,Inf), labels = 0:4)
binned.height = cut(arrest1$ht_feet, breaks = c(-Inf,3,4,5,6,Inf), labels = 0:4)
binned.revcmd = cut(arrest1$revcmd, breaks = c(-Inf,200,400,600,Inf), labels = 0:3)
binned.repcmd = cut(arrest1$repcmd, breaks = c(-Inf,200,400,600,Inf), labels = 0:3)
binned.perobs = cut(arrest1$perobs, breaks = c(-Inf,20,40,60,Inf), labels = 0:3)
binned.sernum = cut(arrest1$ser_num, breaks = c(-Inf,500,1000,1500,Inf), labels = 0:3)
binned.pct = cut(arrest1$pct, breaks = c(-Inf,30,60,90,Inf), labels = 0:3)

arrest1<- arrest1[,-c(2,3,4,6,43,44,46,47,48,49,52,53,54,55)]

arrest1$binned.timestop <- binned.timestop
arrest1$binned.detailCM <- binned.detailCM
arrest1$binned.age <- binned.age
arrest1$binned.addrpct <- binned.addrpct
arrest1$binned.weight <- binned.weight
arrest1$binned.height <- binned.height
arrest1$binned.revcmd <- binned.revcmd
arrest1$binned.repcmd <- binned.repcmd
arrest1$binned.perobs <- binned.perobs
arrest1$binned.sernum <- binned.sernum
arrest1$binned.pct <- binned.pct
arrest1$arstmade <- as.factor(arrest1$arstmade)

arrest_sample <- arrest1[sample(nrow(arrest1), 10000), ]

arrest_sample<- arrest_sample[,-c(3)]

library(arules)
library(arulesViz)

bank_ar <- arrest_sample



rules<-apriori(data=bank_ar, parameter=list(supp=0.05,conf = 0.08), appearance = list(default="lhs",rhs="arstmade=1"), control = list(verbose=F))
rules<-sort(rules, decreasing=TRUE,by="lift")

inspect(rules[1])