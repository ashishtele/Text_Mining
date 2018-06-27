
# loading previous dataset

df_all_un %>% 
  group_by(speech,word) %>% 
  summarise(n = n()) %>% 
  arrange(-n)-> sp_words

df_all_un %>% 
  group_by(speech) %>% 
  summarise(total = n()) -> total_words

sp_words <- left_join(sp_words,total_words, by = "speech") 

# words distribution

sp_words %>% 
  mutate(ratio = n/total) %>% 
  ggplot(aes(ratio, fill = "red")) +
  geom_histogram() +
  scale_x_log10()

# Zipf's law states that within a group or corpus of documents,
# the frequency of any word is inversely proportional to its rank 
# in a frequency table. Thus the most frequent word will occur
# approximately twice as often as the second most frequent word, 
# three times as often as the third most frequent word


sp_words %>% 
  group_by(speech) %>% 
  mutate(rank = row_number(),
         term_freq = n / total) %>% 
  filter(rank < 500) %>% 
  ggplot(aes(rank, term_freq, color = speech)) +
  geom_line() +
  scale_x_log10() +
  scale_y_log10()

## Inverse document frequency 

# tf-idf: helps to find the important words that can
# provide specific document context

head(sp_words)
sp_words %>% 
  bind_tf_idf(word, speech, n) %>% 
  arrange(-tf_idf)-> sp_words
sp_words

# highest tf-idf plot

sp_words %>% 
  group_by(speech) %>% 
  top_n(10, wt = tf_idf) %>% 
  filter(speech < 5) %>% 
  ungroup() %>% 
  ggplot(aes(x = reorder(word,tf_idf), y = tf_idf, fill = speech)) +
  geom_bar(stat = "identity") +
 facet_wrap( ~ speech, scales = "free") +
  coord_flip()


