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

amazon_tr <- amazon_tr %>% mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION ~ ., data = amazon_tr) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazon_tr)
baked <- bake(prep, new_data = amazon_te)

logistic_mod <- logistic_reg() %>%
  set_engine("glm")

logistic_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logistic_mod) %>%
  fit(data = amazon_tr)

amazon_predictions <- predict(logistic_workflow,
                              new_data = amazon_te,
                              type = "prob")

#formats submissions properly
submission <- bind_cols(amazon_te %>% select(id), amazon_predictions$.pred_1)
submission <- submission %>% rename("Id" = "id", "Action" = "...2")
#writes onto a csv
vroom_write(submission, "./AmazonEmployeeAccess/AmazonEmployeeAccess/submission.csv", col_names = TRUE, delim = ", ")


### 10-11-23 Penalized Logistic Regression
my_recipe <- recipe(ACTION ~ ., data = amazon_tr) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazon_tr)
baked <- bake(prep, new_data = amazon_te)

plog_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(plog_mod)

##Tuning Grid

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(amazon_tr, v = 5, repeats = 1)

CV_results <- amazon_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_tr)

amazon_predictions <- final_wf %>% predict(new_data = amazon_te, type = "prob")

#formats submissions properly
submission <- bind_cols(amazon_te %>% select(id), amazon_predictions$.pred_1)
submission <- submission %>% rename("Id" = "id", "Action" = "...2")
#writes onto a csv
vroom_write(submission, "./AmazonEmployeeAccess/AmazonEmployeeAccess/submission.csv", col_names = TRUE, delim = ", ")

##10/16/2023
#Random Forests!!!
tree_mod <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


##Workflow
forest_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_mod)


tuning_grid <- grid_regular(mtry(range = c(1, 15)),
                            min_n(),
                            levels = 2)

folds <- vfold_cv(amazon_tr, v = 5, repeats = 1)

CV_results <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- forest_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_tr)

amazon_predictions <- final_wf %>% predict(new_data = amazon_te, type = "prob")

#formats submissions properly
submission <- bind_cols(amazon_te %>% select(id), amazon_predictions$.pred_1)
submission <- submission %>% rename("Id" = "id", "Action" = "...2")
#writes onto a csv
vroom_write(submission, "./AmazonEmployeeAccess/AmazonEmployeeAccess/submission.csv", col_names = TRUE, delim = ", ")

##Naive BAYES
library(tidymodels)

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

library(discrim)


nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

folds <- vfold_cv(amazon_tr, v = 5, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_tr)

amazon_predictions <- final_wf %>% predict(new_data = amazon_te, type = "prob")

#formats submissions properly
submission <- bind_cols(amazon_te %>% select(id), amazon_predictions$.pred_1)
submission <- submission %>% rename("Id" = "id", "Action" = "...2")
#writes onto a csv
vroom_write(submission, "./AmazonEmployeeAccess/AmazonEmployeeAccess/submission.csv", col_names = TRUE, delim = ", ")

###PCA Dimension Reduction
library(tidymodels)
library(embed)
library(vroom)
library(ggmosaic)
library(DataExplorer)
library(GGally)
library(patchwork)

amazon_tr <- vroom("./AmazonEmployeeAccess/AmazonEmployeeAccess/train.csv")
amazon_te <- vroom("./AmazonEmployeeAccess/AmazonEmployeeAccess/test.csv")


amazon_tr <- amazon_tr %>% mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION ~ ., data = amazon_tr) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .9)
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazon_tr)
baked <- bake(prep, new_data = amazon_te)

#KNN
knn <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")



knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn)

tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

folds <- vfold_cv(amazon_tr, v = 5, repeats = 1)

CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_tr)

amazon_predictions <- final_wf %>% predict(new_data = amazon_te, type = "prob")

#formats submissions properly
submission <- bind_cols(amazon_te %>% select(id), amazon_predictions$.pred_1)
submission <- submission %>% rename("Id" = "id", "Action" = "...2")
#writes onto a csv
vroom_write(submission, "./AmazonEmployeeAccess/AmazonEmployeeAccess/submission.csv", col_names = TRUE, delim = ", ")


#### SVM 
library(tidymodels)
library(embed)
library(vroom)
library(ggmosaic)
library(DataExplorer)
library(GGally)
library(patchwork)

amazon_tr <- vroom("./AmazonEmployeeAccess/AmazonEmployeeAccess/train.csv")
amazon_te <- vroom("./AmazonEmployeeAccess/AmazonEmployeeAccess/test.csv")

amazon_tr <- amazon_tr %>% mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION ~ ., data = amazon_tr) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .8)
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazon_tr)
baked <- bake(prep, new_data = amazon_te)

svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 2)

folds <- vfold_cv(amazon_tr, v = 4, repeats = 1)

CV_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_tr)

amazon_predictions <- final_wf %>% predict(new_data = amazon_te, type = "prob")

#formats submissions properly
submission <- bind_cols(amazon_te %>% select(id), amazon_predictions$.pred_1)
submission <- submission %>% rename("Id" = "id", "Action" = "...2")
#writes onto a csv
vroom_write(submission, "./AmazonEmployeeAccess/AmazonEmployeeAccess/submission.csv", col_names = TRUE, delim = ", ")



