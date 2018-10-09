# package loading

load_pk <- function()
{
  library(tidyverse)
  library(tidytext)
}

load_pk()

#data import
df <- data.table::fread(
  "C:\\Users\\Ashish\\Documents\\R\\Text_Mining\\Womens Clothing E-Commerce Reviews.csv",
  data.table = FALSE
) %>% 
  rename(ID = V1) %>% 
  select(-Title) %>% 
  mutate(Age = as.integer(Age))

glimpse(df)

df %>% 
  select(`Review Text`) %>% 
  unnest_tokens(word, `Review Text`) %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) %>% 
  ggplot(aes(n))+
  geom_histogram()+
  scale_x_log10()

# Uninformative words
df %>% 
  unnest_tokens(word, `Review Text`) %>% 
  anti_join(stop_words) %>% 
  count(word) %>% 
  arrange(n)

# removing unnecessary words

df %>% 
  unnest_tokens(word, `Review Text`) %>% 
  anti_join(stop_words) %>% 
  filter(
    !str_detect(word, pattern = "[[:digit:]]"),
    !str_detect(word, pattern = "[[:punct:]]"),
    !str_detect(word, pattern = "(.)\\1{2,}"), # removes any words with 3 or more repeated letters
    !str_detect(word, pattern = "\\b(.)\\b") # single letter words
  ) %>% 
  mutate(word = corpus::text_tokens(word, stemmer = "en") %>% unlist()) %>% 
  count(word) %>% 
  mutate(word = if_else(n < 10, "infrequent", word)) %>% 
  group_by(word) %>% 
  summarize(n = sum(n)) %>% 
  arrange(n)

df %>% 
  unnest_tokens(word, `Review Text`) %>% 
  anti_join(stop_words) %>% 
  filter(
    !str_detect(word, pattern = "[[:digit:]]"),
    !str_detect(word, pattern = "[[:punct:]]"),
    !str_detect(word, pattern = "(.)\\1{2,}"), # removes any words with 3 or more repeated letters
    !str_detect(word, pattern = "\\b(.)\\b") # single letter words
  ) %>% 
  count(word) %>% 
  filter(n >= 10) %>% 
  pull(word) -> word_list

bow_features <- df %>% 
  unnest_tokens(word, `Review Text`) %>% 
  anti_join(stop_words) %>% 
  filter(word %in% word_list) %>% 
  count(ID, word) %>% 
  spread(word, n) %>% 
  map_df(replace_na, 0)

dim(bow_features)

df %>% 
  inner_join(bow_features, by = "ID") %>% 
  select(-`Review Text`) -> df_bow

# Bigram creation

df %>% 
  unnest_tokens(bigram, `Review Text`, token = "ngrams",
                n = 2) %>% 
  separate(bigram, c("word1","word2"), sep = " ") %>% 
  filter(
    !word1 %in% stop_words$word,
    !word2 %in% stop_words$word,
    !str_detect(word1, pattern = "[[:digit:]]"),
    !str_detect(word2, pattern = "[[:digit:]]"),
    !str_detect(word1, pattern = "[[:punct:]]"),
    !str_detect(word2, pattern = "[[:punct:]]"),
    !str_detect(word1, pattern = "(.)\\1{2,}"),
    !str_detect(word2, pattern = "(.)\\1{2,}"),
    !str_detect(word1, pattern = "\\b(.)\\b"),
    !str_detect(word2, pattern = "\\b(.)\\b")
  ) %>% 
  unite("bigram", c(word1, word2), sep = " ") %>% 
  count(bigram) %>% 
  filter(n >= 10) %>% 
  pull(bigram) -> ngram_list

bigrams <- df %>% 
  select(`Review Text`) %>% 
  unnest_tokens(bigram, `Review Text`, token = "ngrams", n = 2) %>% 
  filter(bigram %in% ngram_list) %>% 
  separate(bigram, c("word1", "word2"), sep = " ")


count_w1 <- bigrams %>% 
  count(word1)

count_w2 <- bigrams %>% 
  count(word2)

count_w12 <- bigrams %>% 
  count(word1, word2)

N <- nrow(bigrams)

# log-likelihood

LL_test <- count_w12 %>% 
  left_join(count_w1, by = "word1") %>% 
  left_join(count_w2, by = "word2") %>% 
  rename(c_w1 = n.y, c_w2 = n, c_w12 = n.x) %>% 
  mutate(
    p = c_w2 / N,
    p1 = c_w12 / c_w1,
    p2 = (c_w2 - c_w12) / (N - c_w1),
    LL = log((pbinom(c_w12, c_w1, p)* pbinom(c_w2 - c_w12, N - c_w1, p)) / (pbinom(c_w12, c_w1, p1) * pbinom(c_w2 - c_w12, N - c_w1, p)))
  )

head(LL_test)

unique_bigrams <- LL_test %>% 
  mutate(
    Chi_value = -2 * LL,
    pvalue = pchisq(LL, df = 1)
  ) %>% 
  filter(pvalue < 0.05) %>% 
  select(word1, word2) %>% 
  unite(bigram, word1, word2, sep = " ")

head(unique_bigrams)

# informative words

bow <- df %>% 
  select(`Review Text`) %>% 
  unnest_tokens(word, `Review Text`) %>% 
  anti_join(stop_words) %>% 
  filter(word %in% word_list)

bow_pos <- mutate(bow,
                  pos = RDRPOSTagger::rdr_pos(tagger, x = word)$pos)


