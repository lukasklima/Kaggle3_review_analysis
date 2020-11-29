library(tidyverse)
library(tidytext)
library(glmnet)
library(ROCR)


# Data import -------------------------------------------------------------

amazon <-  read_csv("bda18-amazon/amazon_baby.csv")
head(amazon)

# get index of training data (the ones that are non-NA values)
trainidx <-  !is.na(amazon$rating)
table(trainidx)


# Preprocessing -----------------------------------------------------------

# extract bigrams
reviews_2ngram <- amazon %>%
  mutate(id = row_number()) %>%
  unnest_tokens(token, review, token = "ngrams", n = 2) %>%
  count(id, name, rating, token)

head(reviews_2ngram)

# extract all features 
features_2ngrams <- 
  reviews_2ngram %>%
  group_by(id) %>%
  mutate(nwords = sum(n)) %>% # the number of tokens per document
  group_by(token) %>%
  mutate(
    docfreq = n(), # number of documents that contain the token
    idf = log(nrow(amazon) / docfreq), # inverse document frequency ('surprise')
    tf = n / nwords, # relative frequency of token within the given document
    tf_idf = tf*idf
  ) %>%
  ungroup()

# put into sparse matrix
dtm_2ngrams <-
  filter(features_2ngrams, docfreq > 10) %>%
  cast_sparse(row=id, column=token, value = tf_idf)

dim(dtm_2ngrams)

# checking which observations may have been ommiited 
used_rows = as.integer(rownames(dtm_2ngrams))
used_amazon = amazon[used_rows, ]
trainidx = trainidx[used_rows]
table(trainidx)


# Making the response variable --------------------------------------------

# extract ratings
y = used_amazon$rating

# make logical
y_train <- ifelse(y[trainidx] > 3, "pos", "neg")

# factorize
y_train <- factor(y_train)


# Fitting models ----------------------------------------------------------

# fit a cross-validated cv.glmnet 
fit_lasso_cv <- cv.glmnet(dtm_2ngrams[trainidx, ], y_train, 
                             family = "binomial", type.measure = 'auc', alpha = 1)

# plot lambdas against AUC
plot(fit_lasso_cv, xvar = "lambda")

# extract the optimal lambda and optimal lambda within 1SE
best_l_lasso <- fit_lasso_cv$lambda.min

best_l_lasso_1se <- fit_lasso_cv$lambda.1se


# Predictions on training set ---------------------------------------------

# making predictions based on the best lambda and lasso model 
pred_lasso_best_prob = tibble(id = which(trainidx), rating = y_train) %>%
  mutate(pred = predict(fit_lasso_cv, dtm_2ngrams[trainidx, ], s = best_l_lasso, type = "response")) # add predictions

#plotting a quick ROC curve just to check 
pred_best <- prediction(pred_lasso_best_prob$pred, pred_lasso_best_prob$rating)
perf <- performance(pred_best,"tpr","fpr")
performance(pred_best, "auc") # shows calculated AUC for model
plot(perf, colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )


# making predictions based on the best 1SE lambda and lasso model 
pred_lasso_1se_prob = tibble(id = which(trainidx), rating = y_train) %>%
  mutate(pred = predict(fit_lasso_cv, dtm_2ngrams[trainidx, ], s = best_l_lasso_1se, type = "response")) # add predictions

# plotting an ROC curve
pred_se <- prediction(pred_lasso_1se_prob$pred, pred_lasso_1se_prob$rating)
perf <- performance(pred_se,"tpr","fpr")
performance(pred_se,"auc") # shows calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )


# Predictions on test set for submission ----------------------------------

sample_submission = read_csv("bda18-amazon/amazon_baby_testset_sample.csv", 
                             col_types = cols(col_character(), col_double()))

# used_rows computed earlier contains all the indices of reviews used in dtm
all_test_reviews = which(is.na(amazon$rating))
missing_test_ids = setdiff(all_test_reviews, used_rows)

# best prediction if no review features are available
mean(y[trainidx])
best_default_prediction = 1
cat("best baseline prediction:", best_default_prediction,"\n")

# make prediction data frame
dtm_test_predictions <- 
  tibble(Id = as.character(used_rows[!trainidx]),
         pred=predict(fit_lasso_cv, dtm_2ngrams[!trainidx, ], s = best_l_lasso, type = "response")[,1]
  )

pred_df = sample_submission %>%
  left_join(dtm_test_predictions) %>%
  mutate(Prediction = ifelse(Id %in% missing_test_ids, 1, pred)) # add predictions

# make submission file
pred_df %>%
  transmute(Id=Id, Prediction = Prediction) %>%
  write_csv("my_submission_4n.csv")
file.show("my_submission_4n.csv")

