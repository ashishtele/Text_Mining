
library(caret)
library(text2vec)
data("movie_review")

setDT(movie_review)
setkey(movie_review, id)
head(movie_review,1)

all_ids <- movie_review$id

train_ids <- sample(all_ids, 4000)
test_ids <- setdiff(all_ids, train_ids)

train <- movie_review[J(train_ids)]
test <- movie_review[J(test_ids)]

# preprocessing and tokenization 

pre_lower <- tolower
tok_fun <- word_tokenizer

it_train <- itoken(train$review,                # iterator to create vocabulary
                   preprocessor = pre_lower,
                   tokenizer = tok_fun,
                   ids = train$id,
                   progressbar = FALSE)
vocab <- create_vocabulary(it_train, 
                           ngram = c(1L,2L),
                           stopwords = c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours"))            # collecting unique terms from all docs and marking each with a unique ID

pruned <- prune_vocabulary(vocab,
                           term_count_min = 10,
                           doc_proportion_min = 0.001,
                           doc_proportion_max = 0.5)

vectorizer <- vocab_vectorizer(pruned)           # tokens into vector space
h_vect <- hash_vectorizer(hash_size = 2 ^ 14, ngram = c(1L,2L))
dtm_train <- create_dtm(it_train, vectorizer)

dim(dtm_train)


dtm_l1_norm = normalize(dtm_train)         # transformation of rows of DTM, adjust the values from diff to common values
                                           # sum of row values be equal to 1
d <- as.matrix(dtm_l1_norm)
View(d[1:25,1:25])

# define tfidf model

tfidf <- TfIdf$new()
lsa <- LSA$new(n_topics = 10)

# transform with fitted model

dtm_tfidf <- fit_transform(dtm_train, tfidf)
dtm_tfidf <- fit_transform(dtm_tfidf, lsa)

dim(dtm_tfidf)
















