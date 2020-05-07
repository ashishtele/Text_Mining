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
  suppressPackageStartupMessages(require(data.table))
  suppressPackageStartupMessages(require(Matrix))
  library(textfeatures)
  library(doParallel)
  library(pROC)
  library(cutpointr)
}

load_lb()


## Input the file

df <- read_excel("E:\\Study\\R Projects\\Common files\\Sales\\train_text.xlsx")

score <- read_excel("E:\\Study\\R Projects\\Common files\\Sales\\score_text.xlsx")

new_tr <- read_excel("E:\\Study\\R Projects\\Common files\\Sales\\train_t.xlsx")

df$nchar <- nchar(df$Message)

table(df$Target) %>% 
  prop.table() %>%
  as.data.frame() %>% 
  ggplot(aes(Var1,Freq, fill = Var1)) +
  geom_bar(stat = "identity")+
  geom_text(aes(label = scales::percent(Freq)))+
  labs(x = "Target",
       y = "Proportion")

df %>% 
  filter(nchar < 300) %>% 
  ggplot(aes(nchar, fill = Target))+
  geom_density(alpha = 0.4)

glimpse(new_tr)
glimpse(df)
glimpse(score)

# Dropping duplicates

new_tr <- new_tr[!duplicated(tolower(substr(new_tr$Message,1,100))),]
df_1 <- df[!duplicated(tolower(substr(df$Message,1,100))),]

nrow(df) - nrow(df_1)

df_1 %>% 
  rename(text = Message) -> df_1
new_tr %>% 
  rename(text = Message) -> new_tr
score %>% 
  rename(text = Message) %>% 
  mutate(Target = "test") %>% 
  dplyr::select(Target, text)-> score

# Combine train and test

complete <- bind_rows(df_1, score)
rm(df_1,score)


gc()
# Creating text features (train)

text_df_1 <- textfeatures(complete,
                          sentiment = TRUE,
                          word2vec_dims = 30,
                          threads = 10,
                          normalize = FALSE)
text_df_11 <- textfeatures(complete,
                          sentiment = TRUE,
                          word2vec_dims = 30,
                          threads = 10,
                          normalize = TRUE)

text_df_2 <- text_df_11

text_df_1 <- text_df_1 %>% 
  mutate(CHAR = nchar(complete$text),
         RCW = n_caps/n_words,
         RSC = n_nonasciis/n_chars,
         RPC = n_puncts/n_chars)

text_df_2$Char = text_df_1$CHAR
text_df_2$RCW = text_df_1$RCW
text_df_2$RSC = text_df_1$RSC
text_df_2$RPC = text_df_1$RPC

text_df_2$RCW[is.na(text_df_2$RCW)] <- 0
df_complete <- text_df_2

rm(text_df_1,text_df_11,text_df_2)

df_test <- df_complete %>% 
  filter(Target == "test")

df_train <- df_complete %>% 
  filter(!Target == "test")


#################################### Data partition ########################################

rm(index)
index <- createDataPartition(df_train$Target, p = 0.70, list = FALSE)
df_train_1 <- df_train[index,]
df_test_1 <- df_train[-index,]

df_train_1 %>% 
  mutate(Target = as.factor(factor(Target, levels = unique(df_train_1$Target)))) -> df_train_1

df_test_1 %>% 
  mutate(Target = as.factor(factor(Target, levels = unique(df_test_1$Target)))) -> df_test_1

#################################### Correlation check #####################################################
df_train_1 %>% 
  filter(Target == "yes") %>% 
  select_if(is.numeric) %>% 
  cor() %>% 
  corrplot::corrplot()

#################################### Normalization check ###################################################
df_train_1 %>% 
  dplyr::select(-Target) %>% 
  gather(metric, value) %>% 
  ggplot(aes(metric, fill =value))+
  geom_density(show.legend = FALSE)+
  facet_wrap(~metric, scales = "free")
## Normalised


#################################### xgb ##########################################


df_train_1 <- bind_cols(df_train_1[,1],df_train_1[,var_xgb$names])
df_test_1 <- bind_cols(df_test_1[,1],df_test_1[,var_xgb$names])


df_tr_fnl <- bind_rows(df_train_1,df_test_1)
df_ts_fnl <- df_test

rm(df_train_1,df_test_1)

# multifold
set.seed(9)
cv.fld <- createMultiFolds(df_tr_fnl$Target,
                           k = 10,
                           times = 3)

train_ctrl1 <- trainControl(method = "repeatedcv",
                            number = 10,
                            repeats = 3,
                            verboseIter = FALSE,
                            index = cv.fld,
                            preProcOptions = c("center","scale"),
                            classProbs = TRUE,
                            savePredictions = TRUE,
                            summaryFunction = twoClassSummary)


xgbGrid <- expand.grid(nrounds = 125,
                       max_depth = 4,
                       eta = 0.5340,
                       gamma = 0,
                       colsample_bytree = 0.6987,
                       min_child_weight = 1,
                       subsample = 0.5286)

cores<-detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)

rm(mod_xgb)
mod_xgb <- caret::train(Target~.,
                        data = df_tr_fnl,
                        method = "xgbTree",
                        trControl = train_ctrl1,
                        metric = "ROC",
                        tuneGrid = xgbGrid)
stopCluster(cl)

mod_xgb$bestTune
plot(varImp(mod_xgb))


# Bayesian Optimization

library(MlBayesOpt)
set.seed(123)
res0 <- xgb_cv_opt(data = df_tr_fnl,
                   label = Target,
                   objectfun = "binary:logistic",
                   evalmetric = "auc",
                   n_folds = 5,
                   acq = "ucb",
                   init_points = 10,
                   n_iter = 20)


## Table of imp features
varImp(mod_xgb, scale = FALSE)$importance %>% 
  mutate(names = row.names(.)) %>% 
  filter(Overall > 0) %>% 
  arrange(-Overall) -> var_xgb

pred_xgb <- predict(mod_xgb, newdata = df_ts_fnl)
pred_xgb_p <- predict(mod_xgb, newdata = df_ts_fnl, type = "prob")

table(pred_xgb) %>% 
  prop.table()

confusionMatrix(df_test_1$Target, as.factor(pred_xgb), positive = "yes")
# 0.9935
pROC::roc(df_test_1$Target,as.numeric(pred_xgb))
# 0.9766

Pred_table <- data.frame(Msg = score$Message, 
                         xgb_opt = pred_xgb,
                         xgb_opt_p = pred_xgb_p$yes)

xlsx::write.xlsx(Pred_table,"E:\\Study\\R Projects\\Common files\\Sales\\pred_tbl2.xlsx")

################################# Optimal cutoff ##########################################

pred_xgb_p <- predict(mod_xgb, newdata = df_test_1[,-1], type = "prob")
pred_xgb_p$act <- df_test_1$Target

opt_cut <- cutpointr(Pred_table, X,X, pos_class = "yes",
                     neg_class = "no", method = maximize_metric, metric = youden)
opt_cut$optimal_cutpoint
pred_xgb_p$pred <- as.factor(ifelse(pred_xgb_p$no > 0.9009, 'no', 'yes'))

confusionMatrix(pred_xgb_p$act, as.factor(pred_xgb_p$pred), positive = "yes")



plot(opt_cut)

xgbGrid <- expand.grid(nrounds = c(100,150,200),
                       max_depth = c(1,3,4),
                       eta = c(.1, 0.3,.4),
                       gamma = 0,
                       colsample_bytree = c(0.5,0.6,.7),
                       min_child_weight = 1,
                       subsample = c(0.4,0.5,0.6))

set.seed(931992)

cores<-detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)
mod_xgb_tune <-       train(Target~.,
                             data = df_train_2, 
                             method = "xgbTree", 
                             trControl = train_ctrl1,
                             metric = "ROC", 
                             tuneGrid = xgbGrid)

stopCluster(cl)
mod_xgb_tune$bestTune
plot(varImp(mod_xgb_tune))

pred_xgb_tune <- predict(mod_xgb_tune, newdata = df_test_2[,-1])
confusionMatrix(df_test_2$Target, pred_xgb_tune)
# 0.9935
pROC::roc(df_test_2$Target,as.numeric(pred_xgb_tune))
# 0.9766

## Optimal cutoff
pred_xgb1_p <- predict(mod_xgb_tune, newdata = df_test_2[,-1], type = "prob")
pred_xgb1_p$act <- df_test_2$Target

opt_cut <- cutpointr(pred_xgb1_p, no, act, pos_class = "no",
                     neg_class = "yes", method = maximize_metric, metric = youden)
opt_cut$optimal_cutpoint
pred_xgb1_p$pred <- as.factor(ifelse(pred_xgb1_p$no > 0.96075, 'no', 'yes'))

confusionMatrix(pred_xgb1_p$act, as.factor(pred_xgb1_p$pred), positive = "yes")
pROC::roc(pred_xgb1_p$act, as.numeric(pred_xgb1_p$pred))



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

############################### Logistic Reg ####################################

# multifold
set.seed(1234)
cv.fld <- createMultiFolds(df_train_2$Target,
                           k = 10,
                           times = 3)

train_ctrl1 <- trainControl(method = "repeatedcv",
                            number = 10,
                            repeats = 3,
                            verboseIter = FALSE,
                            index = cv.fld,
                            preProcOptions = c("center","scale"),
                            classProbs = TRUE,
                            savePredictions = TRUE,
                            summaryFunction = twoClassSummary)

cores<-detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)

mod_glm <- caret::train(df_train_2[,-1],
                        y = df_train_2$Target,
                        method = "glm",
                        family = "binomial",
                        trControl = train_ctrl1)
stopCluster(cl)                

summary(mod_glm)
pred_glm <- predict(mod_glm, newdata = df_test_1[,-1])

confusionMatrix(df_test_1$Target, pred_glm)
# 0.9926
library(pROC)
pROC::roc(df_test_1$Target,as.numeric(pred_glm))
# 0.9826

## Optimal Cutoff
pred_glm_p <- predict(mod_glm, newdata = df_test_1[,-1], type = "prob")
pred_glm_p$act <- df_test_1$Target
opt_cut <- cutpointr(pred_glm_p, no, act, pos_class = "yes",
                     neg_class = "no", method = maximize_metric, metric = youden)
opt_cut$optimal_cutpoint
plot(opt_cut)
pred_glm_p$pred <- as.factor(ifelse(pred_xgb_p$no > 0.1548, 'no', 'yes'))


glimpse(pred_xgb_p)
confusionMatrix(pred_glm_p$act, as.factor(pred_glm_p$pred))
pROC::roc(pred_glm_p$act, as.numeric(pred_glm_p$pred))

############################### Naive-Bayes (Kernel) ####################################



cores<-detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)

search_grid <- expand.grid(usekernel = c(TRUE),
                           fL = 0:2,
                           adjust = seq(0,5,by=1))

mod_nb_1 <- caret::train(df_train_1[,-1],
                         y = df_train_1$Target,
                         method = "nb",
                         trControl = train_ctrl1)

pred_nb_tune <- predict(mod_nb_1, newdata = df_test_1[,-1])
confusionMatrix(df_test_1$Target, pred_nb_tune)
pROC::roc(df_test_1$Target,as.numeric(pred_nb_tune))
# 0.9602

## Optimal Cutoff
pred_nb_p <- predict(mod_nb_1, newdata = df_test_1[,-1], type = "prob")
pred_nb_p$act <- df_test_1$Target
opt_cut <- cutpointr(pred_nb_p, no, act, pos_class = "yes",
                     neg_class = "no", method = maximize_metric, metric = youden)
opt_cut$optimal_cutpoint
plot(opt_cut)
pred_nb_p$pred <- as.factor(ifelse(pred_nb_p$no > 0.99999, 'no', 'yes'))


glimpse(pred_xgb_p)
confusionMatrix(pred_nb_p$act, as.factor(pred_nb_p$pred))
pROC::roc(pred_nb_p$act, as.numeric(pred_nb_p$pred))

################################ SVMLinear #####################################


grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))

cores<-detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)


mod_lsvm <- caret::train(Target~.,
                         data = df_train_1,
                         method = "svmLinear",
                         trControl = train_ctrl1,
                         tuneGrid = grid,
                         tuneLength = 10)
stopCluster(cl)


mod_lsvm # C = 0.25,0.05
pred_lsvm <- predict(mod_lsvm, newdata = df_test_1[,-1])
confusionMatrix(df_test_1$Target, pred_lsvm)
# 0.9945
pROC::roc(df_test_1$Target,as.numeric(pred_lsvm))
# .9804

## Optimal Cutoff
pred_ls_p <- predict(mod_lsvm, newdata = df_test_2[,-1], type = "prob")
pred_ls_p$act <- df_test_2$Target
opt_cut <- cutpointr(pred_ls_p, no, act, pos_class = "yes",
                     neg_class = "no", method = maximize_metric, metric = youden)
opt_cut$optimal_cutpoint
plot(opt_cut)
pred_ls_p$pred <- as.factor(ifelse(pred_xgb_p$no > 0.90285, 'no', 'yes'))

confusionMatrix(pred_ls_p$act, as.factor(pred_ls_p$pred))
pROC::roc(pred_ls_p$act, as.numeric(pred_ls_p$pred))



################################ LightGBM #####################################


library(lightgbm)

glimpse(df_train_1)
df_train_1 %>% 
  mutate(Target = as.numeric(as.factor(factor(Target, levels = unique(df_train_1$Target))))-1) -> df_train_1

df_test_1 %>% 
  mutate(Target = as.numeric(as.factor(factor(Target, levels = unique(df_test_1$Target))))-1) -> df_test_1

lgb.train <- lgb.Dataset(as.matrix(df_train_1[,colnames(df_train_1) != "Target"]),
                         label = df_train_1$Target)

lgb.test <- lgb.Dataset(as.matrix(df_test_1[,colnames(df_test_1) != "Target"]),
                         label = df_test_1$Target)


df_train_fnl <- bind_rows(df_train_1, df_test_1)

lgb.train.fnl <- lgb.Dataset(as.matrix(df_train_fnl[,colnames(df_train_1) != "Target"]),
                         label = df_train_fnl$Target)



rm(param.lgb)
lgb.grid = list(objective = "binary",
                metric = "auc",
                # min_sum_hessian_in_leaf = 1,
                feature_fraction = 0.7,
                bagging_fraction = 0.7,
                # bagging_freq = 5,
                # min_data = 100,
                # max_bin = 50,
                # lambda_l1 = 8,
                # lambda_l2 = 1.3,
                # min_data_in_bin=100,
                # min_gain_to_split = 10,
                min_data_in_leaf = 1,
                is_unbalance = TRUE)


lgb.model.cv = lgb.cv(params = lgb.grid, 
                      data = lgb.train.fnl,
                      10,
                      nfold = 5,
                      min_data = 1,
                      learning_rate = 0.01,
                      early_stopping_rounds = 20,
                      # learning_rate = 0.02, 
                      num_leaves = 7,
                      num_threads = 2 , 
                      nrounds = 500)
                      # early_stopping_rounds = 50,
                      # nfold = 5, stratified = TRUE)
lgb.model.cv$best_iter

lgb.model <- lgb.train(
  params = lgb.grid
  , data = lgb.train.fnl
  , valids = list(test = lgb.test)
  , learning_rate = 0.01
  , num_leaves = 7
  , num_threads = 2
  , nrounds = 142
  , early_stopping_rounds = 20
)

lgb.model$best_iter
lgb.model$record_evals[["test"]][["auc"]][["eval"]]


lgb.test = predict(lgb.model,
                   data = as.matrix(df_test[,colnames(df_test) != "Target"]),
                   n = lgb.model$best_iter)

pROC::roc(df_test_1$Target, lgb.test)

# table saving
Pred_table <- data.frame(Msg = score$Message, 
                         light = lgb.test)

xlsx::write.xlsx(Pred_table,"E:\\Study\\R Projects\\Common files\\Sales\\pred_light.xlsx")



## ROC validation
rm(dummy)

dummy1 <- read_excel("E:\\Study\\R Projects\\Common files\\Sales\\Pred_4.xlsx")

pROC::roc(dummy$EM_CLASSTARGET,(dummy1$EM_EVENTPROBABILITY))

