library(data.table)
library(ggplot2)
library(plyr)

data <- fread("/loan.csv", header = T, encoding="UTF-8", integer="double")
dim(data)
str(data)
summary(data)
d <- data[,c("loan_status","grade","purpose","term","home_ownership","addr_state","emp_length",
             "loan_amnt","annual_inc","int_rate","installment","dti","delinq_2yrs","inq_last_6mths")]

apply(d, 2, anyNA)
summary(d[,c("annual_inc","delinq_2yrs","inq_last_6mths")])
df <- na.omit(d)
apply(df, 2, anyNA)
df <- df[df$emp_length!="n/a",]
count(df, "loan_status")
df <- df[!(df$loan_status=="Current"|df$loan_status=="Issued"),]

df$loan_status_2 <- ifelse(df$loan_status=="Fully Paid"|df$loan_status=="Does not meet the credit policy. Status:Fully Paid","GOOD","BAD")
df$loan_status_3 <- ifelse(df$loan_status=="Fully Paid"|df$loan_status=="Does not meet the credit policy. Status:Fully Paid",0,1)
nrow(df) #266010
table(df$loan_status_2) #10:3

ggplot(df, aes(grade, fill=loan_status_2)) + geom_bar(position="stack")
ggplot(df, aes(term, fill=loan_status_2)) + geom_bar(position="stack")
ggplot(df, aes(home_ownership, fill=loan_status_2)) + geom_bar(position="stack")
table(df$home_ownership,df$loan_status_2)
ggplot(df, aes(emp_length, fill=loan_status_2)) + geom_bar(position="stack")
table(df$emp_length,df$loan_status_2)
ggplot(df, aes(purpose, fill=loan_status_2)) + geom_bar(position="stack")
table(df$purpose,df$loan_status_2)

#Naive Bayes model############################################################################################
#AUC function
calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}
#Building single-variable models - Using categorical features
mkPredC <- function(outCol,varCol,appCol) {
  pPos <- sum(outCol==pos)/length(outCol) #default的機率
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[pos]
  vTab <- table(as.factor(outCol),varCol) #dTrain V欄位的值 與default欄位 之table
  pPosWv <- (vTab[2,]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(appCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred
}
#Building single-variable models - Using numeric features
mkPredN <- function(outCol,varCol,appCol) {
  cuts <- unique(as.numeric(quantile(varCol,probs=seq(0, 1, 0.1),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}
#Naieve Bayes
nBayes <- function(pPos,pf) {
  pNeg <- 1 - pPos
  smoothingEpsilon <- 1.0e-5
  scorePos <- log(pPos + smoothingEpsilon) + rowSums(log(pf/pPos + smoothingEpsilon))
  scoreNeg <- log(pNeg + smoothingEpsilon) + rowSums(log((1-pf)/(1-pPos) + smoothingEpsilon))
  m <- pmax(scorePos,scoreNeg)
  expScorePos <- exp(scorePos-m)
  expScoreNeg <- exp(scoreNeg-m)
  expScorePos/(expScorePos+expScoreNeg)
}

##################################################################################################
fold <- 7
#for logistic regression
x <- c("purpose","term","home_ownership","addr_state","emp_length",
       "loan_amnt","annual_inc","int_rate","installment","dti","delinq_2yrs","inq_last_6mths")
fmla <- paste("loan_status_3", paste(x, collapse="+"), sep="~")

#for naive bayes
outcome <- "loan_status_3"
pos <- 1

NBTrainAccuracy <- c()
NBCalAccuracy <- c()
NBTestAccuracy <- c()
NBTrainPrecision <- c()
NBCalPrecision <- c()
NBTestPrecision <- c()
NBTrainRecall <- c()
NBCalRecall <- c()
NBTestRecall <- c()
GLMTrainAccuracy <- c()
GLMCalAccuracy <- c()
GLMTestAccuracy <- c()
GLMTrainPrecision <- c()
GLMCalPrecision <- c()
GLMTestPrecision <- c()
GLMTrainRecall <- c()
GLMCalRecall <- c()
GLMTestRecall <- c()

set.seed(729375)
for(n in 1:fold){
  df$rgroup <- runif(dim(df)[[1]])
  dTrain <- subset(df,df$rgroup<=(fold-2)/fold)
  dCal <- subset(df,df$rgroup>(fold-2)/fold & df$rgroup<=(fold-1)/fold)
  dTest <- subset(df,df$rgroup>(fold-1)/fold)
  
  #for naive bayes
  vars <- setdiff(colnames(dTrain), c('loan_status','loan_status_2','loan_status_3','rgroup'))
  catVars <- vars[sapply(dTrain[,vars],class) %in% c('factor','character')]
  numericVars <- vars[sapply(dTrain[,vars],class) %in% c('numeric','integer')]
  
  for(v in numericVars) {
    pi <- paste('pred',v,sep='')
    dTrain[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTrain[,v])
    dTest[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTest[,v])
    dCal[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dCal[,v])
  }
  
  for(v in catVars) {
    pi <- paste('pred',v,sep='')
    dTrain[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTrain[,v])
    dCal[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dCal[,v])
    dTest[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTest[,v])
  }
  
  pVars <- paste('pred',c(numericVars,catVars),sep='')
  pPos <- sum(dTrain[,outcome]==pos)/length(dTrain[,outcome])
  dTrain$nbpredl <- round(nBayes(pPos,dTrain[,pVars]))
  dCal$nbpredl <- round(nBayes(pPos,dCal[,pVars]))
  dTest$nbpredl <- round(nBayes(pPos,dTest[,pVars]))

  NBTraincontab <- table(dTrain$nbpredl,dTrain[,outcome]==pos)
  NBTrainAccuracy <- c(NBTrainAccuracy,(NBTraincontab[1,1]+NBTraincontab[2,2])/(NBTraincontab[1,1]+NBTraincontab[1,2]+NBTraincontab[2,1]+NBTraincontab[2,2]))
  NBTrainPrecision <- c(NBTrainPrecision,(NBTraincontab[2,2])/(NBTraincontab[2,1]+NBTraincontab[2,2]))
  NBTrainRecall <- c(NBTrainRecall,(NBTraincontab[2,2])/(NBTraincontab[1,2]+NBTraincontab[2,2]))
  NBCalcontab <- table(dCal$nbpredl,dCal[,outcome]==pos)
  NBCalAccuracy <- c(NBCalAccuracy,(NBCalcontab[1,1]+NBCalcontab[2,2])/(NBCalcontab[1,1]+NBCalcontab[1,2]+NBCalcontab[2,1]+NBCalcontab[2,2]))
  NBCalPrecision <- c(NBCalPrecision,(NBCalcontab[2,2])/(NBCalcontab[2,1]+NBCalcontab[2,2]))
  NBCalRecall <- c(NBCalRecall,(NBCalcontab[2,2])/(NBCalcontab[1,2]+NBCalcontab[2,2]))
  NBTestcontab <- table(dTest$nbpredl,dTest[,outcome]==pos)
  NBTestAccuracy <- c(NBTestAccuracy,(NBTestcontab[1,1]+NBTestcontab[2,2])/(NBTestcontab[1,1]+NBTestcontab[1,2]+NBTestcontab[2,1]+NBTestcontab[2,2]))
  NBTestPrecision <- c(NBTestPrecision,(NBTestcontab[2,2])/(NBTestcontab[2,1]+NBTestcontab[2,2]))
  NBTestRecall <- c(NBTestRecall,(NBTestcontab[2,2])/(NBTestcontab[1,2]+NBTestcontab[2,2]))
  
  
  #for logistic regression
  train_logit <- glm(fmla,data = dTrain,family = binomial)
  cal_logit <- glm(fmla,data = dCal,family = binomial)
  test_logit <- glm(fmla,data = dTest,family = binomial)
  
  dTrain$glmpred <- predict(train_logit,newdata=dTrain,type="response")
  dCal$glmpred <- predict(cal_logit,newdata=dCal,type="response")
  dTest$glmpred <- predict(test_logit,newdata=dTest,type="response")

  GLMTraincontab <- table(round(dTrain$glmpred),dTrain[,outcome]==pos)
  GLMTrainAccuracy <- c(GLMTrainAccuracy,(GLMTraincontab[1,1]+GLMTraincontab[2,2])/(GLMTraincontab[1,1]+GLMTraincontab[1,2]+GLMTraincontab[2,1]+GLMTraincontab[2,2]))
  GLMTrainPrecision <- c(GLMTrainPrecision,(GLMTraincontab[2,2])/(GLMTraincontab[2,1]+GLMTraincontab[2,2]))
  GLMTrainRecall <- c(GLMTrainRecall,(GLMTraincontab[2,2])/(GLMTraincontab[1,2]+GLMTraincontab[2,2]))
  GLMCalcontab <- table(round(dCal$glmpred),dCal[,outcome]==pos)
  GLMCalAccuracy <- c(GLMCalAccuracy,(GLMCalcontab[1,1]+GLMCalcontab[2,2])/(GLMCalcontab[1,1]+GLMCalcontab[1,2]+GLMCalcontab[2,1]+GLMCalcontab[2,2]))
  GLMCalPrecision <- c(GLMCalPrecision,(GLMCalcontab[2,2])/(GLMCalcontab[2,1]+GLMCalcontab[2,2]))
  GLMCalRecall <- c(GLMCalRecall,(GLMCalcontab[2,2])/(GLMCalcontab[1,2]+GLMCalcontab[2,2]))
  GLMTestcontab <- table(round(dTest$glmpred),dTest[,outcome]==pos)
  GLMTestAccuracy <- c(GLMTestAccuracy,(GLMTestcontab[1,1]+GLMTestcontab[2,2])/(GLMTestcontab[1,1]+GLMTestcontab[1,2]+GLMTestcontab[2,1]+GLMTestcontab[2,2]))
  GLMTestPrecision <- c(GLMTestPrecision,(GLMTestcontab[2,2])/(GLMTestcontab[2,1]+GLMTestcontab[2,2]))
  GLMTestRecall <- c(GLMTestRecall,(GLMTestcontab[2,2])/(GLMTestcontab[1,2]+GLMTestcontab[2,2]))
  
}

set <- c("trainning","calibration","test")
NBaccuracy <- c(mean(NBTrainAccuracy), mean(NBCalAccuracy), mean(NBTestAccuracy))
GLMaccuracy <- c(mean(GLMTrainAccuracy), mean(GLMCalAccuracy), mean(GLMTestAccuracy))
NBprecision <- c(mean(NBTrainPrecision), mean(NBCalPrecision), mean(NBTestPrecision))
GLMprecision <- c(mean(GLMTrainPrecision), mean(GLMCalPrecision), mean(GLMTestPrecision))
NBrecall <- c(mean(NBTrainRecall), mean(NBCalRecall), mean(NBTestRecall))
GLMrecall <- c(mean(GLMTrainRecall), mean(GLMCalRecall), mean(GLMTestRecall))
res <- data.frame(set,NBaccuracy,GLMaccuracy,NBprecision,GLMprecision,NBrecall,GLMrecall,stringsAsFactors=FALSE)
rownames(res) <- NULL
print(res)

#############################################################################################
