library(dplyr)
library(tidyverse)
library(tidyr)
library(gamlr)
library(foreach)
library(ggplot2)
library(stargazer)
library(randomForest)
library(caret)
library(estimatr)
library(lubridate)
library(modelr)
library(rsample)
library(mosaic)
library(parallel)
library(foreach)
library(ggcorrplot)

load("/Users/hannahjones/Desktop/pfi_pu_pert_rdata")
ppfi <- pfi_pu_pert

#remove homeschool data-- only look at in-person school observations
ppfi <- ppfi[-c(11:97)]

#remove online school data
ppfi <- ppfi[-c(44:70)]

#remove non-sampled children information
ppfi <- ppfi[-c(276:291)]
 #save with weighting and imputation flags
ppfi_withweighting <- ppfi

#remove weighting info
ppfi <- ppfi[-c(276:358)]

#if there were missing values, they were imputed as shown by flags in data.  We remove this data
ppfi <- ppfi[-c(276:615)]

#cut data for schools dont give grades (SEGRADES == 5)
ppfi<- ppfi %>%
  filter(SEGRADES != 5)


#columns 87-91? parent's satisfaction with school

#feature engineering:
#Combine bros/sis into siblings, ,mom/dad into parents, gma/gpa/aunt/unc/cuz into extended fam

#forest1 = randomForest( SEGRADES ~ . -SEGRADEQ - BASMID, data = ppfi_train)
#modelr::rmse(forest1, ppfi_test)
#yhat_test = predict(forest1, ppfi_test)
#plot(yhat_test, ppfi_test$SEGRADES)
#varImpPlot(forest1)

#change all -1 to 0
ppfi_recode <- lapply(ppfi, as.character)
ppfi_recode[ppfi_recode == "-1"] <- "0"
ppfi_recode<- lapply(ppfi_recode, as.numeric)
ppfi_recode<- data.frame(ppfi_recode)

ppfi_split = initial_split(ppfi_recode)
n = nrow(ppfi_recode)
n_train = floor(0.8*n)
n_test = n - n_train
train_cases = sample.int(n, size=n_train, replace=FALSE)
ppfi_train = training(ppfi_split)
ppfi_test = testing(ppfi_split)


pca_ppfi = prcomp(ppfi_train, scale=TRUE)
loadings = pca_ppfi$rotation
scores = pca_ppfi$x
summary(pca_ppfi)

var <- apply(pca_ppfi$x, 2, var)  
prop <- var / sum(var)
cumsum(prop) # 75% of variance explained by PC 1 - 263
plot(cumsum(pca_ppfi$sdev^2/sum(pca_ppfi$sdev^2)))

train = data.frame(pca_ppfi$x[,1:60])
train['SEGRADES']= ppfi_train$SEGRADES
train_load = pca_ppfi$rotation[,1:60]

test_pre <- scale(ppfi_test) %*% train_load
test <- as.data.frame(test_pre)
test['SEGRADES']=ppfi_test$SEGRADES

# run a random forest using PCA variables in train_author

train$SEGRADES = factor(train$SEGRADES) 

grades_forest = randomForest(SEGRADES ~ .,
                             data = train, importance = TRUE)

yhat = predict(grades_forest, test)

comp_table<-as.data.frame(table(yhat,as.factor(test$SEGRADES)))
predicted<-yhat
actual<-as.factor(test$SEGRADES)
comp_table<-as.data.frame(cbind(actual,predicted))
comp_table$flag<-ifelse(comp_table$actual==comp_table$predicted,1,0)
sum(comp_table$flag)
sum(comp_table$flag)*100/nrow(comp_table)

#look at important PC
varImpPlot(grades_forest)

#examine top 5(ish) principle components to understand constituent parts- PC6, PC1, PC30, PC27, PC5, PC2

o1 = order(loadings[,6], decreasing=TRUE)
colnames(ppfi_recode)[head(o1,3)]
colnames(ppfi_recode)[tail(o1,3)]

o2 = order(loadings[,27], decreasing=TRUE)
colnames(ppfi_recode)[head(o2,3)]
colnames(ppfi_recode)[tail(o2,3)]

##FOR SEGRADES-- higher number implies worse grade

##INTERESTING GRAPHS TO PROVE POINTS

#PARENT RACE VS INVOLVEMENT -- FEATURE ENGINEERING for involvement?





