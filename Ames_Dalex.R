###################################################################
############ Descriptive machine learning explanations#############
###################################################################

# Loading the required packages

rm(list = ls())

load_lb <- function()
{
  suppressPackageStartupMessages(library(doParallel))
  suppressPackageStartupMessages(library(rsample))
  suppressPackageStartupMessages(library(tidyr))
  suppressPackageStartupMessages(library(dplyr))
  suppressPackageStartupMessages(library(h2o))
}

load_lb()

# Registering the multi-clusters
cl <- makeCluster(detectCores())
registerDoParallel(cl)

##stopCluster(cl)

if(!require(DALEX)) install.packages("DALEX") 
library(DALEX)

# intitialize h2o session
h2o.no_progress()
h2o.init()


# DALEX function, categorical predictor to be factors
# h2o does not support ordered categorical variables


## ordered columns
rsample::attrition %>% 
  select_if(is.ordered) %>% 
  names()

df <- rsample::attrition %>% 
  mutate_if(is.ordered,factor, ordered = FALSE) %>% 
  mutate(Attrition = recode(Attrition, "Yes" = "1", "No" = "0") %>% factor(levels = c("1","0")))

# h2o object

df.h2o <- as.h2o(df)

# train, valid and test

set.seed(931992)
splits <- h2o.splitFrame(df.h2o, ratios = c(0.7,0.15))
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]


# response variable names

y <- "Attrition"
x <- setdiff(names(df),y)

# elastic net model

glm <- h2o.glm(
  x = x,
  y = y,
  training_frame = train,
  validation_frame = valid,
  family = "binomial",
  seed = 931992
)

# random forest

rf <- h2o.randomForest(
  x = x,
  y = y,
  training_frame = train,
  validation_frame = valid,
  ntrees = 1000,
  stopping_rounds = 10,
  stopping_metric = "AUC",
  stopping_tolerance = 0.005,
  seed = 234
)

# gradient boosting

gbm <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train,
  validation_frame = valid,
  ntrees = 1000,
  stopping_rounds = 10,
  stopping_metric = "AUC",
  stopping_tolerance = 0.005,
  seed = 234
)

# model performances
h2o.auc(glm, valid = TRUE)
h2o.auc(rf, valid = TRUE)
h2o.auc(gbm, valid = TRUE)

# Dalex requirements

# x_valid: original data format
# y_valid: numeric vector (classification response to 0/1)
# pred: custom function to return numeric vector.(classification: probability of response)

# feature data to non-h2o obj

x_valid <- as.data.frame(valid[,x])

# response as numeric binary vector

y_valid <- as.vector(as.numeric(as.character(valid$Attrition)))
head(y_valid)

# custom predict function

pred <- function(model, newdata)
{
  results <- as.data.frame(h2o.predict(model, as.h2o(newdata)))
  #a <- names(results)     # predict, p0, p1
  return(results[[3L]])
}

pred(rf, x_valid) %>% head()


# explainers

exp_glm <- explain(
  model = glm,
  data = x_valid,
  y = y_valid,
  predict_function = pred,
  label = "h2o glm"
)

exp_rf <- explain(
  model = rf,
  data = x_valid,
  y = y_valid,
  predict_function = pred,
  label = "h2o rf"
)

exp_gbm <- explain(
  model = gbm,
  data = x_valid,
  y = y_valid,
  predict_function = pred,
  label = "h2o gbm"
)

# explainer obj
class(exp_gbm)
summary(exp_gbm)

## Residual diagnostic
# residual quantiles
# plot: absolute residual values

# predictions and residuals

resi_glm <- model_performance(exp_glm)
resi_rf <- model_performance(exp_rf)
resi_gbm <- model_performance(exp_gbm)

# quantiles
resi_gbm
resi_glm
resi_rf

# comparison plots for each model

p1 <- plot(resi_gbm, resi_glm, resi_rf)
p2 <- plot(resi_gbm, resi_glm, resi_rf, geom = "boxplot")

gridExtra::grid.arrange(p1,p2, nrow = 1)


# variable importance (performace based)

vip_glm <- variable_importance(exp_glm, 
                               n_sample = -1,
                               loss_function = loss_root_mean_square)
vip_rf <- variable_importance(exp_rf, 
                               n_sample = -1,
                               loss_function = loss_root_mean_square)
vip_gbm <- variable_importance(exp_gbm, 
                               n_sample = -1,
                               loss_function = loss_root_mean_square)
plot(vip_glm, vip_rf, vip_gbm)


# predictor-response relationship (PDP Package)

pdp_glm <- variable_response(exp_glm, variable = "Age", type = "pdp")
pdp_rf <- variable_response(exp_rf, variable = "Age", type = "pdp")
pdp_gbm <- variable_response(exp_gbm, variable = "Age", type = "pdp")

plot(variable_response(exp_rf, variable = "MonthlyIncome", type = "pdp"))

plot(pdp_gbm, pdp_glm, pdp_rf)

# categorical variable

cat_glm <- variable_response(exp_glm, variable = "EnvironmentSatisfaction", type = "factor")
cat_rf <- variable_response(exp_rf, variable = "EnvironmentSatisfaction", type = "factor")
cat_gbm <- variable_response(exp_gbm, variable = "EnvironmentSatisfaction", type = "factor")

plot(cat_gbm, cat_glm, cat_rf)

### Local interpretation

# single obs.

new_c <- valid[1,] %>% 
  as.data.frame()

# breakdown distance

new_c_glm <- prediction_breakdown(exp_glm, observation = new_c)
new_c_rf <- prediction_breakdown(exp_rf, observation = new_c)
new_c_gbm <- prediction_breakdown(exp_gbm, observation = new_c)

class(new_c_glm)

# top 10 influencial variables (ICE: individual conditional expectation)

new_c_glm[1:10, 1:5]
plot(new_c_glm)

library(ggplot2)
new_c_glm %>% 
  ggplot(aes(contribution, reorder(variable, contribution)))+
  geom_point()+
  geom_vline(xintercept = 0, size = 1, color = "black")+
  labs(title = "GLM contribution")















