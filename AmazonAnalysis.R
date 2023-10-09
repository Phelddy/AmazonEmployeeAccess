library(tidymodels)
library(embed)
library(vroom)

amazon_tr <- vroom("./AmazonEmployeeAccess/AmazonEmployeeAccess/train.csv")
amazon_te <- vroom("./AmazonEmployeeAccess/AmazonEmployeeAccess/test.csv")

library(ggmosaic)


library(DataExplorer)
library(GGally)
library(patchwork)

glimpse(amazon_tr)
plot_correlation(amazon_tr)
plot_histogram(amazon_tr)
plot_missing(amazon_tr)

amazon_EDA <- amazon_tr %>% mutate(RESOURCE = as.factor(RESOURCE), ACTION = as.factor(ACTION))
ggplot(data = amazon_EDA) + geom_mosaic(aes(x = product(RESOURCE), fill = ACTION))

my_recipe <- recipe(ACTION ~ ., data = amazon_tr) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors())
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazon_tr)
baked <- bake(prep, new_data = amazon_te)

logistic_mod <- logistic_red() %>%
  set_engine("glm")

logistic_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logistic_mod) %>%
  fit(data = amazon_tr)

amazon_predictions <- predict(logistic_workflow,
                              new_data = amazon_te,
                              type = "prob")
