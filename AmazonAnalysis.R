library(tidymodels)
library(embed)
library(vroom)

amazon_tr <- vroom("./AmazonEmployeeAccess/AmazonEmployeeAccess/train.csv")
amazon_te <- vroom("./AmazonEmployeeAccess/AmazonEmployeeAccess/test.csv")

library(ggmosaic)

ggplot(data = amazon_tr) + geom_mosaic(aes(x = product(RESOURCE), fill = ACTION))

library(DataExplorer)
library(GGally)
library(patchwork)

glimpse(amazon_tr)
plot_correlation(amazon_tr)
plot_histogram(amazon_tr)
plot_missing(amazon_tr)

amazon_EDA <- amazon_tr %>% mutate(RESOURCE = as.factor(RESOURCE), ACTION = as.factor(ACTION))
ggplot(data = amazon_EDA) + geom_mosaic(aes(x = product(RESOURCE), fill = ACTION))

my_recipe <- recipe(rFormula, data=myDataset) %>%4
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  step_other(var2, threshold = .05) %>% # combines categorical values that occur <5% into an "other" value6
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding7
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(target_var)) #target encoding8
# also step_lencode_glm() and step_lencode_bayes()9
10
11
# NOTE: some of these step functions are not appropriate to use together12
13
# apply the recipe to your data14
prep <- prep(my_recipe)15
baked <- bake(prep, new_data = NULL)

my_recipe <- recipe(ACTION ~ ., data = amazon_tr) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors())
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazon_tr)
