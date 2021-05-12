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
library(utils)

ppfi_trimmed <- read.csv('https://raw.githubusercontent.com/Joey-Herrera/Data-Mining-Project/main/NHES/ppfi_trimmed.csv')


#columns 87-91? parent's satisfaction with school

#feature engineering:
#Combine bros/sis into siblings, ,mom/dad into parents, gma/gpa/aunt/unc/cuz into extended fam

#change all -1 to 0
ppfi_recode <- lapply(ppfi_trimmed, as.character)
ppfi_recode[ppfi_recode == "-1"] <- "0"
ppfi_recode<- lapply(ppfi_recode, as.numeric)
ppfi_recode<- data.frame(ppfi_recode)

#intro graphs

#frequency of attending school meetings, avg days spent helping with hw, 
ppfi_recode %>%
  filter(CWHITE == 1) %>%
  summarize(avg_grades = mean(SEGRADES),avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

ppfi_recode %>%
  filter(CBLACK == 1) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

ppfi_recode %>%
  filter(CASIAN == 1) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

ppfi_recode %>%
  filter(CHISPAN == 1) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

#speak spanish or other non-english at home
ppfi_recode %>%
  filter(CSPEAKX == 3 | CSPEAKX == 5) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

#english at home
ppfi_recode %>%
  filter(CSPEAKX == 2) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

#compare grades to a few different factor starting with FO-- zoo and whatnot


#Building a model for success

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





