
## Sentiment Analysis

library(tidytext)

# 4 sentiments lexicoms 
sentiments %>%                     # AFINN - (-5) neg to (5) positive
  distinct(lexicon)                # nrc - binary fashion - pos,neg,anger,joy etc
get_sentiments("nrc")              # bing - binary fashion - positive and negative
                                   

# nrc sentimet check


df_all_un %>% 
  left_join(get_sentiments("nrc"), by = "word") %>% 
  filter(!(is.na(sentiment))) %>% 
  group_by(sentiment) %>% 
  summarise(senti = n()) %>% 
  arrange(-senti) %>% 
  top_n(10)-> df_nrc

# 'nrc' lexicon plot

ggplot(df_nrc, aes(x = reorder(sentiment,senti), y = senti, fill = sentiment)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = senti), hjust = 0.2) +
  labs(title = "Sentiments based on NRC Lexicons",
       x = "Sentiments", y = "Senti Count") +
  coord_flip()

## Negative and fear are 3rd and 4th contributors.


# We have minimum ~1500 words in a speech
# lets break in 30 words
df_all_un %>% 
  group_by(speech) %>% 
  summarise(wrd = n()) %>% 
  arrange(wrd)


df_all_un %>% 
  group_by(speech) %>% 
  mutate(word_count = 1:n(),
         index = word_count %/% 30 + 1) %>% 
  left_join(get_sentiments("bing"), by = "word") %>% 
  filter(!(is.na(sentiment))) %>% 
  group_by(speech,index,sentiment) %>% 
  summarise(n = n()) %>% 
  ungroup() %>% 
  spread(sentiment,n, fill = 0) %>% 
  mutate(sentiment = positive - negative) %>% 
  filter(speech < 7) %>% 
  ggplot(aes(x = index,y = sentiment, fill = speech)) +
  geom_bar(alpha = 0.5, stat = "identity", show.legend = FALSE) +
  facet_wrap(~ speech, ncol = 2, scales = "free_x" )

# we can see how the sentiments are changing as time passes

## Lexicon comparison

df_all_un %>% 
  group_by(speech) %>% 
  mutate(word_count = 1:n(),
         index = word_count %/% 30 + 1) %>% 
  left_join(get_sentiments("bing"), by = "word") %>% 
  filter(!(is.na(sentiment))) %>% 
  group_by(speech,index,sentiment) %>% 
  summarise(n = n()) %>% 
  ungroup() %>% 
  spread(sentiment,n, fill = 0) %>% 
  mutate(sentiment = positive - negative) %>% 
  filter(speech < 7) %>% 
  ggplot(aes(x = index,y = sentiment, fill = speech)) +
  geom_bar(alpha = 0.5, stat = "identity", show.legend = FALSE) +
  facet_wrap(~ speech, ncol = 2, scales = "free_x" ) -> p_bing


df_all_un %>% 
  group_by(speech) %>% 
  mutate(word_count = 1:n(),
         index = word_count %/% 30 + 1) %>% 
  left_join(get_sentiments("nrc") %>% 
              filter(sentiment %in% c("positive","negative"))) %>% 
  filter(!(is.na(sentiment))) %>% 
  group_by(speech,index,sentiment) %>% 
  summarise(n = n()) %>% 
  ungroup() %>% 
  spread(sentiment,n, fill = 0) %>% 
  mutate(sentiment = positive - negative) %>% 
  filter(speech < 7) %>% 
  ggplot(aes(x = index,y = sentiment, fill = speech)) +
  geom_bar(alpha = 0.5, stat = "identity", show.legend = FALSE) +
  facet_wrap(~ speech, ncol = 2, scales = "free_x" ) -> p_nrc

gridExtra::grid.arrange(p_bing, p_nrc, ncol = 2)

## Common sentiment words

df_all_un %>% 
  left_join(get_sentiments("bing")) %>% 
  filter(!(is.na(sentiment))) %>% 
  count(word,sentiment) %>% 
  top_n(10) %>% 
  ungroup() %>% 
  ggplot(aes(x = reorder(word,n), y = n, fill = sentiment)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ sentiment, scales = "free_y")+
  coord_flip()


# Sentiment analysis (larger units, more than unigrams)

files <- list.files(pattern = "*.txt")
df_all <- lapply(files, read_file) %>% 
  unlist() %>%
  tibble()
colnames(df_all) <- c("text")

df_all <- tibble(speech = seq_along(df_all$text), text = df_all$text)
df_all %>% 
  unnest_tokens(output = sentence,
                input = text,
                token = "sentences") %>% 
                distinct(speech,sentence)-> df_all_sent

df_all_sent %>% 
  group_by(speech) %>% 
  mutate(sent_num = 1:n(),
         index = round(sent_num/n(),2)) %>% 
  unnest_tokens(word,sentence) %>% 
  left_join(get_sentiments("afinn")) %>% 
  filter(!(is.na(score))) %>% 
  group_by(speech,index) %>% 
  summarise(sentiment = sum(score, na.rm = TRUE)) %>% 
  arrange(-sentiment) -> senti

ggplot(senti, aes(index, factor(speech, levels = sort(unique(speech),decreasing = TRUE)), fill = sentiment))  +
  geom_tile(color = "white") +
  scale_fill_gradient2() +
  scale_x_continuous(labels = scales::percent, expand = c(0,0))+
  scale_y_discrete(expand = c(0,0)) +
  labs(x = "Speech progression", y = "Speech") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),       # remove the grid lines
        panel.grid.minor = element_blank(),
        legend.position = "top")














