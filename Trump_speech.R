
rm(list = ls())

# loading the required libraries, if not install

library(tidyverse)
library(stringr)
library(tidytext)
library(data.table)
library(ggplot2)

pck <- c("tidytext")

for (p in pck){
  if(!(p %in% rownames(installed.packages())))
  {install.packages(p)}
  require(p)
}
rm(pck,p)

# loading the text file

df <- read_file("E:/Study/R Projects/Common files/full_speech.txt")   #keep folder name short
df

# tibble dataframe conversion

text <- tibble(text = df)
text %>%
  unnest_tokens(output = word,
                input = text,
                token = "words",
                to_lower = TRUE) -> txt_unnest
rm(text)                                               # free memory

### Word Frequency

txt_unnest %>% 
  count(word, sort = TRUE)

# Most of the common words are 'Stop words', removing them
  
txt_unnest %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE)  %>%                        # make sure to give column name as 'word' for join to work
  top_n(10) %>% 
  ggplot(aes(x= reorder(word,n), y= n, fill = word)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = n), hjust = -0.2) +
  labs(title = "Top 10 common words",
       x = "Words", y = "Frequency") +
  coord_flip() +
  theme_minimal()

# Interesting  !!!!!

# percent of frequency
txt_unnest %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE)  %>%
  mutate(percent = n/sum(n)) %>% 
  top_n(10) %>% 
  ggplot(aes(x= reorder(word,percent), y= percent, fill = word)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = scales::percent(percent)), hjust = -0.1) +
  labs(title = "Top 10 common words by percentage",
       x = "Words", y = "% Frequency") +
  coord_flip() +
  theme_minimal()


## Different speeches

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
  unnest_tokens(output = word,
                input = text,
                token = "words",
                to_lower = TRUE) -> df_all_un
rm(df_all)

## word count

df_all_un %>% 
  count(word, sort = TRUE)      # remove stopwords

df_all_un %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) %>% 
  top_n(10) %>% 
  ggplot(aes(x=reorder(word,n), y = n, fill = word)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 10 commom words (all speeches)", x = "words", y = "count") +
  geom_text(aes(label = n), hjust = -0.2) +
  coord_flip() +
  #scale_fill_brewer(palette = 'YlOrRd') + 
  theme_minimal(base_size = 11)

## Speech wise analysis

df_all_un %>% 
  anti_join(stop_words) %>% 
  group_by(speech) %>% 
  count(word) %>% 
  filter(speech < 7) %>% 
  arrange(speech,-n) %>% 
  top_n(10) %>% 
  ungroup() %>% 
  ggplot(aes(x=reorder(word,n), y = n, fill = word)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ speech, scales = "free_y") +                              # free_y to to avoid non-zero in group display
  labs(title = "Top 10 commom words (6 speeches)", x = "Freq", y = "") +
  geom_text(aes(label = n), hjust = -0.2) +
  coord_flip() +
  #scale_fill_brewer(palette = 'YlOrRd') + 
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")


# percent of words across all speeches

all_pct <- df_all_un %>% 
           anti_join(stop_words) %>%
           group_by(word) %>% 
           summarise(cnt = n()) %>% 
           transmute(word,all_wrd = cnt/sum(cnt))
# percent of words by speeches

freq <- df_all_un %>% 
        anti_join(stop_words) %>% 
        group_by(word,speech) %>% 
        summarise(cnt_sp = n()) %>% 
        mutate(sp_word = cnt_sp/sum(cnt_sp)) %>% 
        left_join(all_pct) %>% 
        arrange(-sp_word) %>% 
        ungroup()
freq <- freq %>% 
  filter(speech <5)

ggplot(freq, aes(x=all_wrd, y = sp_word)) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3)+
  geom_text(aes(label = word, check_overlap = TRUE)) +
  scale_x_log10(labels = scales::percent_format()) +
  scale_y_log10(labels = scales::percent_format()) +
  facet_wrap(~ speech, ncol = 2)

## Correlation check

freq %>%
  group_by(speech) %>% 
  summarise(corr = cor(all_wrd,sp_word),
            p_value = cor.test(all_wrd, sp_word)$p.value)
## negative correlation: word frequencies are same across speeches.
# right botton corner shows most common words among all speeches






