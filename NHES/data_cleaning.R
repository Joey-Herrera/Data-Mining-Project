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


#columns 87-91? parent's satisfaction with school

#feature engineering:
#Combine bros/sis into siblings, ,mom/dad into parents, gma/gpa/aunt/unc/cuz into extended fam

ppfi_split = initial_split(ppfi)
n = nrow(ppfi)
n_train = floor(0.8*n)
n_test = n - n_train
train_cases = sample.int(n, size=n_train, replace=FALSE)
ppfi_train = training(ppfi_split)
ppfi_test = testing(ppfi_split)

forest1 = randomForest( SEGRADES ~ . -SEGRADEQ - BASMID, data = ppfi_train)
modelr::rmse(forest1, ppfi_test)
yhat_test = predict(forest1, ppfi_test)
plot(yhat_test, ppfi_test$SEGRADES)
varImpPlot(forest1)

#change all -1 to 0
ppfi_recode <- lapply(ppfi, as.character)
ppfi_recode[ppfi_recode == "-1"] <- "0"
ppfi_recode<- data.frame(ppfi_recode)

pca_ppfi = prcomp(ppfi_recode, rank=2, scale=TRUE)

loadings_ID = pca_ID$rotation
scores_ID = pca_ID$x
summary(pca_ID)


