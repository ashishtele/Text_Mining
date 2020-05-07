#################################### Text feature Approach #################################
############################################################################################

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

#stopCluster(cl)
## Input the file

df <- read_excel("E:\\Study\\R Projects\\Common files\\Sales\\train_text.xlsx")
score <- read_excel("E:\\Study\\R Projects\\Common files\\Sales\\score_text.xlsx")

glimpse(df)
glimpse(score)

# Dropping duplicates

df_1 <- df[!duplicated(tolower(substr(df$Message,1,100))),]

rm(df)

df_1 %>% 
  rename(text = Message) -> df_1
score %>% 
  rename(text = Message) %>% 
  mutate(Target = "test") %>% 
  dplyr::select(Target, text)-> score

# Combine train and test

complete <- bind_rows(df_1, score)
rm(df_1,score)

# Creating text features (train)

text_df_1 <- textfeatures(complete,
                          sentiment = TRUE,
                          word2vec_dims = 20,
                          threads = 10,
                          normalize = FALSE)
# text_df_11 <- textfeatures(complete,
#                            sentiment = TRUE,
#                            word2vec_dims = 20,
#                            threads = 10,
#                            normalize = TRUE)


text_df_1 <- text_df_1 %>% 
  mutate(RCW = n_caps/n_words,
         RSC = n_nonasciis/n_chars,
         RPC = n_puncts/n_chars)


text_df_1$RCW[is.na(text_df_1$RCW)] <- 0
df_complete <- text_df_1

rm(text_df_1)

df_test <- df_complete %>% 
  filter(Target == "test")

df_train <- df_complete %>% 
  filter(!Target == "test")


# Data partition

ind <- createDataPartition(df_train$Target, p = 0.7, list = FALSE)
df_train_1 <- df_train[ind,]
df_test_1 <- df_train[-ind,]

df_train_1 %>% 
  mutate(Target = as.factor(factor(Target, levels = unique(df_train_1$Target)))) -> df_train_1

df_test_1 %>% 
  mutate(Target = as.factor(factor(Target, levels = unique(df_test_1$Target)))) -> df_test_1

set.seed(1992)
library(rsample)
cv_spl <- vfold_cv(df_train_1,
                   v = 10,
                   repeats = 3,
                   strata = "Target")

cv_spl$splits[[1]] %>% analysis() %>% nrow()
cv_spl$splits[[1]] %>% assessment() %>% nrow()

conv_rsmpl <- rsample2caret(cv_spl)

ctrl <- trainControl(method = "repeatedcv",
                     savePredictions = "final")

ctrl$index <- conv_rsmpl$index
ctrl$indexOut <- conv_rsmpl$indexOut


library(recipes)

df_rec <- recipe(Target ~ ., 
                 data = df_train_1) %>% 
  step_center(all_predictors()) %>% 
  step_scale(all_predictors())


df_rec_trained <- prep(df_rec, training = df_train_1,
                   retain = TRUE,
                   verbose = TRUE)

df_train_2 <- df_rec_trained %>% juice()

############################### Logistic Reg ####################################


mod_glm_1 <- caret::train(df_rec,
                        data = df_train_1,
                        method = "glm",
                        family = "binomial",
                        trControl = train_ctrl1)

getTrainPerf(mod_glm)

pred_glm_1 <- predict(mod_glm_1, newdata = df_test_1[,-1])
confusionMatrix(df_test_1$Target, pred_glm)
# 0.9926
library(pROC)
roc(df_test_1$Target,as.numeric(pred_glm))
# 0.9826

############################### Naive-Bayes (simple) #############################


# mod_nb <- caret::train(df_train[,-1],
#                        y = df_train$Target,
#                        method = "nb",
#                        trControl = train_ctrl1)
# 
# pred_nb <- predict(mod_nb, newdata = df_test[,-1])
# confusionMatrix(df_test$Target, pred_nb)
# library(pROC)
# roc(df_test$Target,as.numeric(pred_nb))
# # 0.9483

############################### Naive-Bayes (Kernel) ####################################

search_grid <- expand.grid(usekernel = c(TRUE),
                           fL = 0:2,
                           adjust = seq(0,5,by=1))

mod_nb_1 <- caret::train(df_train_1[,-1],
                         y = df_train_1$Target,
                         method = "nb",
                         trControl = train_ctrl1,
                         tuneGrid = search_grid)
# top models:

mod_nb_1$results %>% 
  top_n(3, wt = Accuracy) %>% 
  arrange(-Accuracy)

pred_nb_tune <- predict(mod_nb_1, newdata = df_test_1[,-1])
confusionMatrix(df_test_1$Target, pred_nb_tune)
roc(df_test_1$Target,as.numeric(pred_nb_tune))
# 0.9602

plot(mod_nb_1)

################################ SVMLinear #####################################


grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))

mod_lsvm <- caret::train(Target~.,
                         data = df_train_1,
                         method = "svmLinear",
                         trControl = train_ctrl1,
                         tuneGrid = grid,
                         tuneLength = 10)


mod_lsvm # C = 0.25,0.05
pred_lsvm <- predict(mod_lsvm, newdata = df_test_1[,-1])
confusionMatrix(df_test_1$Target, pred_lsvm)
# 0.9945
roc(df_test_1$Target,as.numeric(pred_lsvm))
# .9804


###################################### xgb ##########################################

mod_xgb <- caret::train(df_rec,
                        data = df_train_1,
                        method = "xgbTree",
                        trControl = ctrl)

mod_xgb$bestTune
plot(varImp(mod_xgb))

pred_xgb <- predict(mod_xgb, newdata = df_test_1[,-1])
confusionMatrix(df_test_1$Target, pred_xgb)
# 0.9935
roc(df_test_1$Target,as.numeric(pred_xgb))
# 0.9766


xgbGrid <- expand.grid(nrounds = c(100, 150,200),
                       max_depth = c(1,3,4,5),
                       eta = c(.1, 0.3,.4),
                       gamma = 0,
                       colsample_bytree = c(0.6,.7,0.8),
                       min_child_weight = 1,
                       subsample = c(.8, 0.9, 1))

set.seed(931992)
mod_xgb_tune <-       train(Target~.,
                            data = df_train, 
                            method = "xgbTree", 
                            trControl = ctrl1,
                            metric = "ROC", 
                            tuneGrid = xgbGrid)

mod_xgb_tune$bestTune
plot(varImp(mod_xgb_tune))

pred_xgb_tune <- predict(mod_xgb_tune, newdata = df_test[,-1])
confusionMatrix(df_test$Target, pred_xgb_tune)
# 0.9935
roc(df_test$Target,as.numeric(pred_xgb_tune))
# 0.9766


final <- data.frame(pred_lsvm = pred_lsvm,
                    pred_xgb = pred_xgb,
                    pred_nb_tune = pred_nb_tune,
                    pred_glm = pred_glm,
                    Target = df_test_1$Target)

final$m_v <- as.factor(ifelse(final$pred_lsvm=='yes' & final$pred_xgb=='yes' & final$pred_glm=='yes','yes'
                              ,ifelse(final$pred_lsvm=='yes' & final$pred_nb_tune=='yes'& final$pred_glm=='yes'
                                      ,'yes',ifelse(final$pred_xgb=='yes' & final$pred_nb_tune=='yes' & final$pred_glm=='yes','yes'
                                                    ,ifelse(final$pred_xgb=='yes' & final$pred_nb_tune=='yes' & final$pred_lsvm=='yes','yes','no')))))

confusionMatrix(final$m_v,final$Target)
# 99.54%
roc(final$Target,as.numeric(final$m_v))
# .9809


