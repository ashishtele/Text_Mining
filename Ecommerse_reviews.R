
rm(list = ls())

load_lb <- function()
{
  suppressPackageStartupMessages(library(readxl))
  suppressPackageStartupMessages(library(readr))
  suppressPackageStartupMessages(library(tidyverse))
  suppressPackageStartupMessages(library(lime))
  suppressPackageStartupMessages(library(caret))
  suppressPackageStartupMessages(library(xgboost))
  suppressPackageStartupMessages(library(ggthemes))
  suppressPackageStartupMessages(library(data.table))
  suppressPackageStartupMessages(library(text2vec))
}

load_lb()

theme_set(theme_minimal())

df <- fread("E:\\Study\\R Projects\\Common files\\Text interpretation\\ecommerce.csv",
            na.strings = c("","NA"))


df %>% 
  mutate(Liked = as.factor(ifelse(Rating == 5, 1, 0)),
         text = paste(Title, `Review Text`),
         text = gsub("NA","",text),
         V1 = as.integer(V1),
         `Clothing ID` = as.integer(`Clothing ID`),
         Age = as.integer(Age)) -> df
glimpse(df)

# response distribution

df %>% 
  ggplot(aes(Liked, fill = Liked))+
  geom_bar()+
  scale_fill_tableau(palette = "tableau20")+
  guides(fill = FALSE)

# data separation

set.seed(931992)
index <- createDataPartition(df$Liked,
                             p = 0.8,
                             list = FALSE) 
train <- df[index,]  
test <- df[-index,]  

## text to DTM

# preprocessing steps in function to feed it to lime

mtrx <- function(text){
  tok <- itoken(text, progressbar = FALSE)
  create_dtm(tok, vectorizer = hash_vectorizer())
}

  
train_dtm <- mtrx(train$text) 
str(train_dtm)  
test_dtm <- mtrx(test$text)  
str(test_dtm)  

model_xgb <- xgb.train(list(max_depth = 7,
                            eta = 0.1,
                            objective = "binary:logistic",
                            eval_metric = "error",
                            nthread = 1),
                       xgb.DMatrix(train_dtm,
                                   label = train$Liked == "1"),
                       nrounds = 50)  
model_xgb  

pred <- predict(model_xgb, test_dtm)  
confusionMatrix(test$Liked, as.factor(round(pred, digits = 0)))  
  
  
# lime() function inputs
# test input that was used to construct the model
# the trained model
# preprocessing function

expl <- lime(train$text,
             model_xgb,
             preprocess = mtrx)

interactive_text_explanations(expl)  
  
# explain() function
# test data
# explainer defined with lime()
# the no. of lables want to have explanations
# no. of features to be included in the explanations

explanation <- lime::explain(test$text[1:4],
                             expl,
                             n_labels = 1,
                             n_features = 5)

plot_text_explanations(explanation)
plot_features(explanation)

# stem words
# remove stop words
# create itoken
# create vocab
# prune vocabulary
# transform to vector space

stem_token <- function(x){
  lapply(word_tokenizer(x),
         SnowballC::wordStem,
         language = "en")
}

stop_words = tm::stopwords(kind = "en")

# pruned vocab
vacab_tr <- itoken(train$text,
                   preprocess_function = tolower,
                   tokenizer = stem_token,
                   progressbar = FALSE)
v <- create_vocabulary(vacab_tr,
                       stopwords = stop_words)
pruned_vocab <- prune_vocabulary(v,
                                 doc_proportion_max = 0.99,
                                 doc_proportion_min = 0.01)
vector_tr <- vocab_vectorizer(pruned_vocab)


# preprocessing function

dtm_mat <- function(text, vectorizer = vector_tr){
  vocab <- itoken(text,
                  preprocess_function = tolower,
                  tokenizer = stem_token,
                  progressbar = FALSE)
dtm <- create_dtm(vocab,
                  vectorizer = vectorizer)
  
tfidf <- TfIdf$new()
fit_transform(dtm, tfidf)  
}

train_dtm_1 <- dtm_mat(train$text)
test_dtm_1 <- dtm_mat(test$text)

# new xgb model

model_xgb_1 <- xgb.train(list(max_depth = 10,
                            eta = 0.2,
                            objective = "binary:logistic",
                            eval_metric = "error",
                            nthread = 1),
                       xgb.DMatrix(train_dtm_1,
                                   label = train$Liked == "1"),
                       nrounds = 500)  
model_xgb_1  

pred1 <- predict(model_xgb_1, test_dtm_1)  
confusionMatrix(test$Liked, as.factor(round(pred1, digits = 0)))  

# new explainer

expl1 <- lime(train$text,
              model_xgb_1,
              preprocess = dtm_mat)
explanation1 <- lime::explain(test$text[1:4],
                                     expl1,
                              n_labels = 1,
                              n_features = 4)
plot_text_explanations(explanation1)
