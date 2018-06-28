
rm(list = ls())

library(tidytext)
library(tm)
library(tidyverse)

# loading files

setwd("E:\\Study\\R Projects\\Common files\\trump")

files <- list.files(pattern = "*.txt")
df <-  lapply(files, read_file) %>% 
  unlist() 
#df[1]
class(df)

# removing non-graphical characters

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
df[1]

# term document matrix conversion from files
df_tdm <- VectorSource(df) %>% 
  VCorpus()

getTransformations()

clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, 
                   c(stopwords("en"), "where", "can", "i", "get"))
  corpus1 <- corpus
  corpus <- tm_map(corpus, stemDocument)
  #corpus <- tm_map(corpus, stemCompletion, corpus1)
  return(corpus)
}

df_tdm <- clean_corpus(df_tdm)
df_tdm <- TermDocumentMatrix(df_tdm)

inspect(df_tdm)
  
terms <- Terms(df_tdm)
head(terms)

## TDM to tidy()

df_tidy <- tidy(df_tdm)
unique(df_tidy$document)

## tidy to TDM

df_tdm1 <- df_tidy %>% cast_dtm(document,term,count)
inspect(df_tdm1)




