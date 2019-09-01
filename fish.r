setwd("~/Google Drive/MSISS SS my files/data analytics/r/fish")
library(mice)
library(ModelMetrics)
library(hydroGOF)
library(gbm)

fish=read.csv("fish.csv",header=TRUE)
econ=read.csv("economic.csv",header=TRUE)

# get rid of accounting variables 

econ$Sundry.receipts=NULL
econ$Depreciation=NULL
econ$GROSS.PROFIT=NULL
econ$TOTAL.COSTS=NULL
econ$Total.Fixed.Costs=NULL
econ$Sundry.fixed.costs=NULL
econ$Legal.Fees=NULL
econ$Accountancy=NULL
econ$Loan.Interest=NULL
econ$Total.Variable.Costs=NULL
econ$Sundry.Variable.Costs=NULL
econ$Dues...Levies=NULL
econ$Ice=NULL
econ$Provisions=NULL
econ$Filters.Lube.Oil=NULL
econ$Repairs...Maintenance=NULL
econ$Energy.Costs..Fuel.=NULL
econ$Wages=NULL
econ$Insurance=NULL
econ$Total.Income=NULL

# check for zero variance

econVarZero=nearZeroVar(econ, saveMetrics = T)
econVarZero

# are there any missing entries?

miss=aggr(econ) 
miss

# going to assume that data is missing completely by random

pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(econ,2,pMiss)  # what percentage of data is missing 
# get rid of variable that has a missing percentage of over 30%

econ$Non.Fishing.Income=NULL
econ$FTE=NULL
econ$Total.Jobs=NULL

# get rid of fishing.income as will be merging fish data anyway

econ$Fishing.Income=NULL

# check missing variables again

miss2=aggr(econ) 
miss2

# however there could be missing data disguised as zeros

zerocap=econ[which(econ$Capacity..GT.==0),] # 43 zeros, doesn't make sense to be zero
zeroseg=econ[which(econ$Segment==0),] # no zeros
zeroactive=econ[which(econ$Total.Active.Vessels.in.Segment==0),] # no zeros
zerocapseg=econ[which(econ$Total.Capacity..GT...for.Segment.Size==0),] # no zeros

econ[which(econ$Capacity..GT.==0),]$Capacity..GT.=NA # get rid of zeros and replace with null

miss3=aggr(econ) 
miss3 # now Capacity..GT. 0's show up as missing data points

# now to impute variables with a low percentage of missing data 

tempData=mice(econ,method = "cart")
summary(tempData)
econ= complete(tempData)

miss4=aggr(econ) 
miss4 # now no missing data



# correlation matrix 

#econ$Segment=as.numeric(econ$Segment) only use when calculating cor matrix, as will loose segment names
#econ$Size.Category=as.numeric(econ$Size.Category)

summary(econ)

econCor=round(cor(econ),2)
head(econCor)
meltedcor=melt(econCor)
ggplot(data=meltedcor,aes(x=Var1,y=Var2,fill=value)) + 
  geom_tile(colour="white") +scale_fill_gradient2(low = "yellow", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab",  name="Pearson\nCorrelation") +theme_minimal()+  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1))+coord_fixed()

# should we get rid of variables with high corrleation? yes

# capacity index = size index = length = capacity GT = size category -> keep 1? as they are all measures of size in essence

# keep capacity GT, gte rid of rest 

econ$Capacity.Index=NULL
econ$Size.Index=NULL
econ$Length=NULL
econ$Size.Category=NULL

# Total.Number.of.vessels.in.Segment and Total.Active.of.vessels.in.Segment highly correlated, makes more sense to keep total active 

econ$Total.Number.of.vessels.in.Segment=NULL

# can I get rid of anything else?

# Total.Capacity..GT...for.Segment.Size and Average.Length.for.Segment and Total.Engine.Power..kW..for.Segment.Size highly corrleated 

econ$Average.Length.for.Segment=NULL
econ$Total.Engine.Power..kW..for.Segment.Size=NULL

# examine the data after cleaning 

summary(econ)
econ$Segment=as.factor(econ$Segment)
colnames(econ)[8]="NetProfit"

# merge fish data and economic data 

# change vessel id col name on economic data 

colnames(econ)[2]="vid"
# check its changed
names(econ)

# merge econ and fish
alldata= merge(econ,fish,by=c("vid","Year"))

# now can drop vid year and reference number

alldata$vid=NULL
alldata$Year=NULL
alldata$Reference.Number=NULL

# explore data 


# should investigate Capacity GT as it the capacity of a vessel should have an effect on NetProfit

min(alldata$Capacity..GT.)
max(alldata$Capacity..GT.)
mean(alldata$Capacity..GT.)
quantile(alldata$Capacity..GT.)

ggplot(alldata,aes(x=Capacity..GT.))+
  geom_density(alpha = 0.3,fill="darkorange")+
  ggtitle("Distribution of Capacity..GT.")+
  theme(plot.title = element_text(hjust = 0.5)) 

ggplot(alldata,aes(x=Capacity..GT.,y=NetProfit  ))+
  geom_point(alpha=0.3, colour="darkorange")+
  ggtitle("Relationship between Capacity..GT. and NetProfit")+
  theme(plot.title = element_text(hjust = 0.5)) 

ggplot(alldata,aes(Segment,Capacity..GT.)) +geom_boxplot(aes(colour=Segment))+
  ggtitle("Capacity in each Segment")+  
  theme(plot.title = element_text(hjust = 0.5))

# Segment

ggplot(alldata,aes(Segment,Total.Active.Vessels.in.Segment)) +geom_boxplot(aes(colour=Segment))+
  ggtitle("Total Active Boats in each Segment")+  
  theme(plot.title = element_text(hjust = 0.5))# 

ggplot(alldata,aes(Segment,..count..)) +
  geom_bar(aes(colour=Segment,fill=Segment))+
  ggtitle("Count of how many boats in each Segment")+  
  theme(plot.title = element_text(hjust = 0.5))# 

ggplot(alldata,aes(Segment,NetProfit)) +geom_boxplot(aes(colour=Segment))+
  ggtitle("Relationship between Segment and NetProfit")+  
  theme(plot.title = element_text(hjust = 0.5))# 

ggplot(alldata, aes( Segment, NetProfit )) +geom_boxplot(aes(colour=Segment)) 
+ggtitle("Relationship between Attrition and Monthly Income")+  
  theme(plot.title = element_text(hjust = 0.5))# 

# Net Profit

# descriptive statistics
min(alldata$NetProfit)
max(alldata$NetProfit)
mean(alldata$NetProfit)
quantile(alldata$NetProfit)

# examine the distribution of net profit 

#density plot

ggplot(alldata, aes(NetProfit)) + 
  geom_density(alpha = 0.3,fill="cornflowerblue")+
  ggtitle("Distribution of Monthly Income")+
  theme(plot.title = element_text(hjust = 0.5))

# split into training and testing

set.seed(765443) #seed is set to reproduce the same train and test data
sample = sample(c(0,1), nrow(alldata), replace=TRUE, prob=c(0.3,0.7)) #70:30 split was deemed appropriated considering size of the sample
train = alldata[sample == 1,]
test = alldata[sample == 0,]

# fit a single tree
names(alldata)

fit =rpart(NetProfit~.,train,method="anova")
fit
plotcp(fit)
summary(fit)
fit$cptable

i.min<-which.min(fit$cptable[,"xerror"])
i.se<-which.min(abs(fit$cptable[,"xerror"]
                    -(fit$cptable[i.min,"xerror"]+fit$cptable[i.min,"xstd"])))
cp.best<-fit$cptable[i.se,"CP"]
cp.best ## 0.01 the same as default cp

# prune tree

bestcpfit =rpart(NetProfit~.,train,method="anova", cp=0.01699023)
rpart.plot(bestcpfit,type=1,extra=101) # prodcues and awful tree

rpart.plot(fit,type=1,extra=101)

# test default predictions

unprunedpred <- predict(fit, test)
plot(unprunedpred, test$NetProfit, main="Predicted values vs Actual values for default cp single decision tree", ylab="Actual Net Profit", xlab="Predicted Net Profit", col="purple")
abline(a=0,b=1,lwd=2,lty=2,col="gray")



predictCor=cor(unprunedpred,test$NetProfit)
predictCor # correlation between predicts vs observed is 0.6945726

# calculate the mse 

mean((unprunedpred-test$NetProfit)^2) # 1.7 bn mse

# calculate the mae

mae(unprunedpred,test$NetProfit) # 143540.6

# calculate rmse

rmse(unprunedpred,test$NetProfit) # 423466.6

# test best cp
# test default predictions

prunedpred <- predict(bestcpfit, test)
plot(prunedpred, test$NetProfit, main="Predicted values vs Actual values for best cp single decision tree", ylab="Actual Net Profit", xlab="Predicted Net Profit", col="magenta")
abline(a=0,b=1,lwd=2,lty=2,col="gray")

predictCorbestcp=cor(prunedpred,test$NetProfit)
predictCorbestcp # correlation between predicts vs observed is 0.6731076

# calculate the mse 

mean((prunedpred-test$NetProfit)^2) # 189011466901 mse

# calculate the mae

mae(prunedpred,test$NetProfit) # 146070.5

# calculate rmse

rmse(prunedpred,test$NetProfit) # 423466.6

# try a very small cp 

smallcp =rpart(NetProfit~.,train,method="anova", cp=0.0001)
rpart.plot(smallcp,type=1,extra=101) # prodcues and awful tree

# test default predictions

smallcppred <- predict(smallcp, test)
plot(smallcppred, test$NetProfit, main="Predicted values vs Actual values for default cp single decision tree", ylab="Actual Net Profit", xlab="Predicted Net Profit", col="purple")
abline(a=0,b=1,lwd=2,lty=2,col="gray")

predictCorsmall=cor(smallcppred,test$NetProfit)
predictCorsmall # correlation between predicts vs observed is 0.7041487

# calculate the mse 

mean((smallcppred-test$NetProfit)^2) # 1.7 bn

# calculate the mae

mae(smallcppred,test$NetProfit) # 143325.9

# calculate rmse

rmse(smallcppred,test$NetProfit) # 417802.6


# random forest

# use tuneRf to find the best mtry
bestmtry = tuneRF(train[,-25], train[,25], stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry,main="Best Mtry")

rf=randomForest(NetProfit~., data=train, importance=TRUE, proximity=TRUE, ntree=500, keep.forest=TRUE ,mtry=35)
plot(rf, main="error rate as no. of trees increase") # 
rf

rfpred <- predict(rf, test)
plot(rfpred, test$NetProfit, main="Predicted values vs Actual values for best Random Forest 500 trees", ylab="Actual Net Profit", xlab="Predicted Net Profit", col="magenta")
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# test 500 rf 

predictCorRF=cor(rfpred,test$NetProfit)
predictCorRF #0.6666126

# calculate the mae

mae(rfpred,test$NetProfit) # 141811.2

# calculate rmse

rmse(rfpred,test$NetProfit) # 437585.4

# mse 

mean((rfpred-test$NetProfit)^2) # 191481000000

# rf 70 trees

rf70=randomForest(NetProfit~., data=train, importance=TRUE, proximity=TRUE, ntree=70, keep.forest=TRUE ,mtry=35)

rfpred70 <- predict(rf70, test)
plot(rfpred70, test$NetProfit, main="Predicted values vs Actual values for best Random Forest 70 trees", ylab="Actual Net Profit", xlab="Predicted Net Profit", col="purple")
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# calc correlation
predictCorRF70=cor(rfpred70,test$NetProfit)
predictCorRF70 # 0.6687501

# calculate the mae

mae(rfpred70,test$NetProfit) # 142477.4

# calculate rmse

rmse(rfpred70,test$NetProfit) # 435604.8

# mse 

mean((rfpred70-test$NetProfit)^2) # 189751577569


# variable importance
varImpPlot(rf70) 
importance(rf70)

# partial dependancy plot

partialPlot(rf70, train, Capacity..GT., main="Partial Dependency of Capacity..GT.", col="red", xlab="GT")

partialPlot(rf70, train, Total.Active.Vessels.in.Segment, main="Partial Dependency of Total.Active.Vessels.in.Segment", col="blue", xlab="Count")


partialPlot(rf150, train, YearsSinceLastPromotion, col="blue", add=TRUE)
legend("bottomright", c("TotalWorkingYears","YearsSinceLastPromotion"), lty = c(1,1),col =
         c("red", "blue"))

max(train$TotalWorkingYears)



# boosting 

# change number of trees

ntree=2000
gbmModel=gbm(NetProfit~., train, distribution="gaussian",n.trees = ntree,keep.data = TRUE,shrinkage = 0.01, n.minobsinnode=5)

gbmpred=predict(gbmModel,test,n.tree=ntree)
plot(gbmpred,test$NetProfit)
abline(0,1)

stats2000=c(cor(gbmpred,test$NetProfit),mean((gbmpred-test$NetProfit)^2),rmse(gbmpred,test$NetProfit),mae(gbmpred,test$NetProfit))

stats200
stats300
stats500
stats2000

# change shrinkage
shrink=0.0001
gbmModel=gbm(NetProfit~., train, distribution="gaussian",n.trees = 500,keep.data = TRUE,shrinkage = shrink, n.minobsinnode=5)

gbmpred=predict(gbmModel,test,n.tree=500)
plot(gbmpred,test$NetProfit)
abline(0,1)

stats.0001=c(cor(gbmpred,test$NetProfit),mean((gbmpred-test$NetProfit)^2),rmse(gbmpred,test$NetProfit),mae(gbmpred,test$NetProfit))

stats.1
stats.01=stats500
stats.001
stats.0001

# change n.minobsinnode (no of terminal nodes)

term=4
gbmModel=gbm(NetProfit~., train, distribution="gaussian",n.trees = 500,keep.data = TRUE,shrinkage = 0.01, n.minobsinnode=term)

gbmpred=predict(gbmModel,test,n.tree=500)
plot(gbmpred,test$NetProfit)
abline(0,1)

stats4=c(cor(gbmpred,test$NetProfit),mean((gbmpred-test$NetProfit)^2),rmse(gbmpred,test$NetProfit),mae(gbmpred,test$NetProfit))

stats2
stats3
stats4
stats5
stats6
stats7
stats8

# optimum boosting tree

gbmOptimumModel=gbm(NetProfit~., train, distribution="gaussian",n.trees = 500,keep.data = TRUE,shrinkage = 0.01, n.minobsinnode=3)

gbmpredOpt=predict(gbmOptimumModel,test,n.tree=500)
plot(gbmpredOpt, test$NetProfit, main="Predicted values vs Actual values for optimum boosting", ylab="Actual Net Profit", xlab="Predicted Net Profit", col="purple")
abline(a=0,b=1,lwd=2,lty=2,col="gray")


