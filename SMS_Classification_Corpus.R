######################################################################################
############################# Corpus method ##########################################
######################################################################################

rm(list = ls())

load_lb <- function()
{
  suppressPackageStartupMessages(library(doMC))
  registerDoMC(cores = 8)
  suppressPackageStartupMessages(library(readxl))
  suppressPackageStartupMessages(library(tidyr))
  suppressPackageStartupMessages(library(dplyr))
  suppressPackageStartupMessages(library(caret))
  suppressPackageStartupMessages(library(rpart))
  suppressPackageStartupMessages(library(tree))
  suppressPackageStartupMessages(library(MASS))
  suppressPackageStartupMessages(library(mice))
  suppressPackageStartupMessages(require(h2o))
  suppressPackageStartupMessages(require(data.table))
  suppressPackageStartupMessages(require(Matrix))
}

load_lb()

library(textfeatures)
library(doParallel)
library(pROC)
#Find out how many cores are available (if you don't already know)
cores<-detectCores()
#Create cluster with desired number of cores, leave one open for the machine         
#core processes
cl <- makeCluster(cores[1]-1)
#Register cluster
registerDoParallel(cl)

stopCluster(cl)
## Input the file

df_1 <- read_excel("E:\\Study\\R Projects\\Common files\\Sales\\train_text.xlsx")
score <- read_excel("E:\\Study\\R Projects\\Common files\\Sales\\score_text.xlsx")

glimpse(df)
glimpse(score)


# term document matrix conversion from files
library(tm)
library(stringr)

# removing non-graphical characters

df_1 <- df_1[!duplicated(tolower(substr(df_1$Message,1,100))),]
df <- df_1$Message

df <- str_replace_all(df,"[^[:graph:]]"," ")
df <- gsub("\\["," ",df)
df <- gsub("\\]"," ",df)
df <- gsub("\\?"," ",df)
df <- gsub("\\:"," ",df)
df <- gsub("\\-"," ",df)
df <- gsub("\\."," ",df)
df <- gsub("\\,"," ",df)
df <- gsub("\\$"," ",df)
df <- gsub("\\\\"," ",df)
df <- gsub("  "," ",df)
df <- gsub("   "," ",df)

df_tdm_tr <- VectorSource(df) %>% 
  VCorpus()

getTransformations()

clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, 
                   c(stopwords("en")))
  corpus1 <- corpus
  corpus <- tm_map(corpus, stemDocument)
  #corpus <- tm_map(corpus, stemCompletion, corpus1)
  return(corpus)
}

df_tdm_tr <- clean_corpus(df_tdm_tr)
df_dtm_t2 <- DocumentTermMatrix(df_tdm_tr)
dim(df_dtm_t2)

# most frequent terms

freq_words <- findFreqTerms(df_dtm_t2, lowfreq = 10)
str(freq_words)

df_dtm_t2 <- df_dtm_t2[,freq_words]
dim(df_dtm_t2)


df_new <- as.data.frame(data.matrix(df_dtm_t2), stringsAsFactors = FALSE)

df_new_1 <- bind_cols(Target = df_1$Target,df_new)

prc <- prcomp(df_new_1,
              scale = TRUE)

dim(prc$x)
pca.var <- prc$sdev^2
PVE1 <- pca.var/sum(pca.var)

qplot(c(1:5),head(PVE1,5)) +
  geom_bar(stat = "identity") 

sum(PVE1[1:400])

pca_com <- as.data.frame(prc$x[,c(1:400)])

df_new_1 <- bind_cols(Target = df_1$Target,pca_com)

ind <- createDataPartition(df_new_1$Target, p = 0.7, list = FALSE)
df_train_1 <- df_new_1[ind,]
df_test_1 <- df_new_1[-ind,]


df_train_1 %>% 
  mutate(Target = as.factor(factor(Target, levels = unique(df_train_1$Target)))) -> df_train_1

df_test_1 %>% 
  mutate(Target = as.factor(factor(Target, levels = unique(df_test_1$Target)))) -> df_test_1             


############################### Naive-Bayes (Kernel) ####################################

search_grid <- expand.grid(usekernel = c(TRUE),
                           fL = 0:2,
                           adjust = seq(0,5,by=1))

train_ctrl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           verboseIter = FALSE)

mod_nb_k_comp <- caret::train(Target~.,
                              data = df_train_1,
                              method = "nb",
                              trControl = train_ctrl,
                              tuneGrid = search_grid)


pred_nb_K_tune <- predict(mod_nb_k_comp, newdata = df_test_1[,-1])
confusionMatrix(df_test_1$Target, pred_nb_K_tune)

roc(df_tst$Target,as.numeric(pred_nb_K_tune))

plot(mod_nb_1)

############################ Model - SVM ##############################


ctrl <- trainControl(method = "repeatedcv",
                     number = 3,
                     repeats = 3,
                     verboseIter = FALSE)

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))

mod_lsvm <- caret::train(Target~.,
                         data = df_tr,
                         method = "svmLinear",
                         trControl = ctrl,
                         tuneGrid = grid,
                         tuneLength = 10)

mod_lsvm # C = 0.25
pred_lsvm <- predict(mod_lsvm, newdata = df_tst[,-1], response = "response")
confusionMatrix(df_test$Target, pred_lsvm)
# 98.25%
roc(df_tst$Target,as.numeric(pred_lsvm))
# 0.9406
# 0.9707

############################## xgb #####################################################

rm(ctrl)
ctrl <- trainControl(method = "cv",
                     number = 5,
                     verboseIter = FALSE)

mod_xgb <- caret::train(Target~.,
                        data = df_train_1,
                        method = "xgbTree",
                        trControl = ctrl)

pred_xgb <- predict(mod_xgb, newdata = df_test_1[,-1])
confusionMatrix(df_test_1$Target, pred_xgb)

roc(df_test_1$Target,as.numeric(pred_xgb))


######################## Clustering ###############################################

df <- sapply(text_df_2[,-1], function(x) scale(x))
df


# Distance and similarity matrix

fviz <- function(x)
{
  fviz_dist(x, 
            gradient = list(low="#00AFBB",mid="white",high="#FC4E07"))
}

library(factoextra)
dist.euc <- get_dist(df)
dist.man <- get_dist(df,
                     method = "manhattan")


fviz(dist.euc)

# K means

k.2 <- kmeans(df, centers = 2, nstart = 25)
str(k.2)

fviz_cluster(k.2, data = df)








