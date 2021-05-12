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
library(fpc)

ppfi_trimmed <- read.csv('https://raw.githubusercontent.com/Joey-Herrera/Data-Mining-Project/main/NHES/ppfi_trimmed.csv')


#columns 87-91? parent's satisfaction with school

#feature engineering:
#Combine bros/sis into siblings, ,mom/dad into parents, gma/gpa/aunt/unc/cuz into extended fam

#first do an analysis on who did not answer grades q then remove no responses

#remove all -1
ppfi_recode <- ppfi_trimmed %>%
  filter(SEGRADES != -1)

ppfi_recode<- lapply(ppfi_recode, as.numeric)
ppfi_recode<- data.frame(ppfi_recode)


#intro graphs

#frequency of attending school meetings, avg days spent helping with hw, 

#based on child's race
ppfi_recode %>%
  group_by(RACEETH) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

#based on parent language
ppfi_recode %>%
  group_by(P1SPEAK) %>%
  summarize(avg_grades = mean(SEGRADES),avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

#how difficult is it for parent to participate due to language diff 1-very, 2-somewhat, 3-not diff
ppfi_recode %>%
  group_by(P1DIFFI) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

#does school translate comms into parent native language? 1-yes, 2-no
ppfi_recode %>%
  group_by(P1WRMTL) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))

#parent hours of work a week
#feature engineer for 1=0-10 hrs, 2=10-20 hrs, 3=20-30 hrs, 4=30-40 hrs, 5=40-50hrs, 6=50+

ppfi_recode <- ppfi_recode %>%
  mutate(P1HRSWK_bins = ifelse(P1HRSWK>0 & P1HRSWK<=10, 1, 
                               ifelse(P1HRSWK>10 & P1HRSWK<=20, 2,
                                      ifelse(P1HRSWK>20 & P1HRSWK<=30, 3,
                                             ifelse(P1HRSWK>30 & P1HRSWK<=40, 4,
                                                    ifelse(P1HRSWK>40 & P1HRSWK<=50, 5,
                                                           ifelse(P1HRSWK>50, 6, 0)))))))

ppfi_recode %>%
  group_by(P1HRSWK_bins) %>%
  summarize(avg_grades = mean(SEGRADES), avg_school_meetings = mean(FSFREQ), avg_days_hwhelp = mean(FHHELP), avg_dinners_together = mean(FODINNERX))


#speak spanish or other non-english at home
ppfi_recode %>%
  group_by(CSPEAKX) %>%
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

train = data.frame(pca_ppfi$x[,1:70])
train['SEGRADES']= ppfi_train$SEGRADES
train_load = pca_ppfi$rotation[,1:70]

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

o1 = order(loadings[,19], decreasing=TRUE)
colnames(ppfi_recode)[head(o1,5)]
colnames(ppfi_recode)[tail(o1,3)]

o2 = order(loadings[,4], decreasing=TRUE)
colnames(ppfi_recode)[head(o2,5)]
colnames(ppfi_recode)[tail(o2,3)]

o2 = order(loadings[,1], decreasing=TRUE)
colnames(ppfi_recode)[head(o2,5)]
colnames(ppfi_recode)[tail(o2,3)]

o2 = order(loadings[,27], decreasing=TRUE)
colnames(ppfi_recode)[head(o2,5)]
colnames(ppfi_recode)[tail(o2,3)]

ggplot(train)+
  geom_violin(aes(x = PC1, y=SEGRADES))+ 
  scale_y_discrete(breaks = 1:4, labels=c("A's","B's","C's","D's"))

#clustering
ppfi_clust_train = subset(ppfi_recode, select = -c(SEGRADES, X, BASMID))

ppfi_clust_train_scaled = scale(ppfi_clust_train, center=TRUE, scale=TRUE)

# Extract the centers and scales from the rescaled data (which are named attributes)
mu = attr(ppfi_clust_train_scaled,"scaled:center")
sigma = attr(ppfi_clust_train_scaled,"scaled:scale")

# Run k-means with 2 clusters and 25 starts
clust1 = kmeans(ppfi_clust_train_scaled, 4, nstart=25)

plotcluster(ppfi_recode, clust1$cluster)
  

##FOR SEGRADES-- higher number implies worse grade

##INTERESTING GRAPHS TO PROVE POINTS

#PARENT RACE VS INVOLVEMENT -- FEATURE ENGINEERING for involvement?





