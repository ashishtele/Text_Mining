
rm(list = ls())

# loading the required libraries, if not install

library(tidyverse)
library(stringr)
library(tidytext)
library(data.table)
library(ggplot2)

# loading the text file

getwd()
setwd("E:\\Study\\R Projects\\Common files\\trump")

files <- list.files(pattern = "*.txt")
df_all <- lapply(files, read_file) %>% 
  unlist() %>%
  tibble()
colnames(df_all) <- c("text")

# adding speech seq. number for analysis purpose
df_all <- tibble(speech = seq_along(df_all$text), text = df_all$text)

df_all %>% 
  unnest_tokens(output = bigram,
                input = text,
                token = "ngrams",                          # bigram
                n = 2,
                to_lower = TRUE) -> df_all_un
rm(df_all)

# most common bigrams

df_all_un %>% 
  group_by(bigram) %>% 
  summarise(n = n()) %>% 
  arrange(-n) %>% 
  top_n(10) %>% 
  ggplot(aes(reorder(bigram,n),n,fill = bigram)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = n), hjust = -0.2)+
  labs(title = "Top 10 bigrams in all speeches", y = "Count", x = "Bigrams") +
  coord_flip()


df_all_un %>% 
  separate(bigram, c("w1","w2"), sep = " ") %>% 
  filter(!w1 %in% stop_words$word,
         !w2 %in% stop_words$word) %>% 
  group_by(w1,w2) %>% 
  summarise(n = n()) %>% 
  unite("bigram",c(w1,w2), sep = " ") %>% 
  arrange(-n) %>% 
  top_n(10) %>% 
  ggplot(aes(reorder(bigram,n),n,fill = bigram)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = n), hjust = -0.2)+
  labs(title = "Top 10 bigrams in all speeches (non stopwords)", y = "Count", x = "Bigrams") +
  coord_flip()

## Interesting!! hillary tops in speeches 

# speech wise bigrams

df_all_un %>% 
  separate(bigram, c("w1","w2"), sep = " ") %>% 
  filter(!w1 %in% stop_words$word,
         !w2 %in% stop_words$word) %>% 
  group_by(speech,w1,w2) %>%
  summarise(n=n()) %>% 
  unite("bigram", c(w1,w2), sep = " ") %>% 
  arrange(-n) %>% 
  top_n(10) %>% 
  filter(speech < 5) %>% 
  ggplot(aes(reorder(bigram,n),n,fill = speech)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ speech, scales = "free")+
  geom_text(aes(label = n), hjust = -0.2)+
  labs(title = "Top 10 bigrams speech wise (non stopwords)", y = "Count", x = "Bigrams") +
  coord_flip()

# Analyzing n-grams

df_all_un %>% 
  separate(bigram, c("w1","w2"), sep = " ") %>% 
  filter(!w1 %in% stop_words$word,
         !w2 %in% stop_words$word) %>% 
  group_by(speech,w1,w2) %>%
  summarise(n=n()) %>% 
  unite("bigram", c(w1,w2), sep = " ") %>% 
  bind_tf_idf(bigram, speech, n) %>% 
  arrange(-tf_idf) %>% 
  top_n(10) %>% 
  filter(speech < 5) %>% 
  ggplot(aes(reorder(bigram,tf_idf),tf_idf,fill = speech)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ speech, scales = "free")+
  geom_text(aes(label = scales::percent(tf_idf)), hjust = -0.2)+
  labs(title = "Highest tf-idf bi-grams (non stopwords)", y = "tf_idf", x = "Bigrams") +
  coord_flip()
  
# bi-grams preceded by negative words

AFN <- get_sentiments("afinn")

negation_words <- c("not","no","never","without","hardly")

df_all_un %>% 
  separate(bigram, c("w1","w2"), sep = " ") %>% 
  filter(w1 %in% negation_words) %>% 
  left_join(AFN, by = c(w2 = "word")) %>% 
  filter(!is.na(score)) %>% 
  group_by(w1,w2,score) %>% 
  summarise(n=n()) %>% 
  arrange(-n) %>% 
  mutate(contribution = n * score) %>% 
  arrange(-contribution) %>% 
  ggplot(aes(reorder(w2,contribution),contribution, fill = contribution > 0))+
  geom_bar(stat = "identity", show.legend = FALSE)+
  labs(title = "Misspecifying sentiments")+
  coord_flip()

  
  
# n-grams network

library(igraph)

df_all_un %>% 
  separate(bigram, c("w1","w2"), sep = " ") %>% 
  filter(!w1 %in% stop_words$word,
         !w2 %in% stop_words$word) %>% 
  group_by(w1,w2) %>%
  summarise(n=n()) %>% 
  unite("bigram", c(w1,w2), sep = " ") %>%
  filter(n > 20) %>% 
  graph_from_data_frame() -> bigram_graph

library(ggraph)
set.seed(1234)  

a <- grid::arrow(type = "closed", length = unit(.15,"inches"))
a

ggraph(bigram_graph, layout = "fr")+
  geom_edge_link() +
  geom_node_point(color = "lightblue", size = 5)+
  geom_node_text(aes(label= name), vjust = 1, hjust = 1)+
  theme_void()

# Word correlation

df_all %>% 
  unnest_tokens(output = word,
                input = text,
                token = "words", 
                to_lower = TRUE) -> df_all_uni
rm(df_all)

df_all_uni %>% 
  filter(!word %in% stop_words$word)

# pairwise count
library(widyr)

df_all_uni %>% 
  filter(!word %in% stop_words$word) %>% 
  pairwise_count(word,speech,sort = TRUE) -> pair_count


pair_count %>% 
  filter(item1 == "hillary")

## phi coefficient

df_all_uni %>% 
  group_by(word) %>% 
  filter(n() > 20) %>% 
  pairwise_cor(word, speech) %>% 
  filter(!is.na(correlation), correlation > 0.65) %>% 
  graph_from_data_frame() %>% 
  ggraph(layout = "fr")+
  geom_edge_link(aes(edge_alpha = correlation), show.legend = FALSE) +
  geom_node_point(color = "lightblue", size = 5)+
  geom_node_text(aes(label= name), vjust = 1, hjust = 1)+
  theme_void()






