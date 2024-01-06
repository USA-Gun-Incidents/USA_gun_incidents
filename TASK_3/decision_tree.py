# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
#
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
#
# # Decision Tree Classifier
#
# We import the libraries and define constants and settings of the notebook:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import pydotplus
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from time import time
from classification_utils import *

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RANDOM_STATE = 42
RESULTS_DIR = '../data/classification_results'
clf_name = 'DecisionTreeClassifier'

# %% [markdown]
# We load the data:

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

# load the names of the features to use for the classification task
features_for_clf = json.loads(open('../data/clf_indicators_names_rule_based.json').read())

# project on the features to use
indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# %% [markdown]
# We display the features names we will use:

# %%
print(features_for_clf)
print(f'Number of features: {len(features_for_clf)}')

# %% [markdown]
# We define a list of the categorical features:

# %%
categorical_features = [
    'day', 'day_of_week', 'month', 'year',
    'democrat', 'gun_law_rank',
    'aggression', 'accidental', 'defensive', 'suicide',
    'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]

# %% [markdown]
# Parameters explored:
# - criterion: The function to measure the quality of a split. We will try both 'centropy' and 'gini'.
# - class_weight: Weights associated with classes. We will try both 'balanced' (classes are assigned weights that are inversely proportional to their frequencies) and not.
# - min_samples_split: The minimum number of samples required to split an internal node. We will try 2 (the minimum possible) and different fractions of the training set.
# - min_samples_leaf: The minimum number of samples required to be at a leaf node. We will try 1 (the minimum possible) and different fractions of the training set.
#
# Fixed parameters:
# - splitter: The strategy used to choose the split at each node. We won't try random beacuse we will experiment a similar approach using Random Forests.
# - max_depth: The maximum depth of the tree. We won't limit the depth of the tree through this parameter. We will control the depth of the tree using the 'min_samples_split' and 'min_samples_leaf', parameter.
# - max_leaft_nodes: The maximum number of leaf nodes. We won't limit the number of leaf nodes through this parameter. We will control the number of leaf nodes using the 'min_samples_leaf', parameter.
# - min_impurity_decreas: The minimum impurity decrease to split a node. We won't use this parameter, we will use instead the 'min_samples_split' and 'min_samples_leaf' parameters to control the growth of the tree.
# - ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning. We will later prune the best tree using this algorithm.
# - min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. We won't weight samples. To address class imbalance we will use the 'class_weight' parameter.
# - max_features: The number of features to consider when looking for the best split. We will experiment a similar approach using Random Forests.

# %%
param_grid = {
    # values to try
    'criterion': ['entropy', 'gini'],
    'class_weight': ['balanced', None],
    'min_samples_split': [2, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    'min_samples_leaf': [1, 0.01, 0.025, 0.05, 0.1],
    # default values
    #'splitter': ['best'],
    #'max_depth': [None],
    #'max_leaf_nodes': [None],
    #'min_impurity_decrease': [0],
    #'ccp_alpha': [0],
    #'min_weight_fraction_leaf': [0],
    #'max_features': [None],
    'random_state': [RANDOM_STATE]
}

gs = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5, # uses a stratified 5-fold cv to validate the models
    refit=False
)
gs.fit(indicators_train_df, true_labels_train)

# %% [markdown]
# We display the grid search results:

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
cv_results_df.head()

# %% [markdown]
# We visualize the interaction of hyperparameters through a heatmap:

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# replace the minimum values for param_min_samples_split and param_min_samples_leaf with 0 to compare it with the other percentages
cv_results_df['param_min_samples_split'] = cv_results_df['param_min_samples_split'].replace(2, 0)
cv_results_df['param_min_samples_leaf'] = cv_results_df['param_min_samples_leaf'].replace(1, 0)
pvt_balanced = pd.pivot_table(
    cv_results_df[(cv_results_df['param_class_weight'] == 'balanced')],
    values='mean_test_score',
    index=['param_min_samples_leaf'],
    columns=['param_min_samples_split', 'param_criterion']
)
pvt_non_balanced = pd.pivot_table(
    cv_results_df[(cv_results_df['param_class_weight'].isna())],
    values='mean_test_score',
    index=['param_min_samples_leaf'],
    columns=['param_min_samples_split', 'param_criterion']
)
min_score = cv_results_df['mean_test_score'].min()
max_score = cv_results_df['mean_test_score'].max()
sns.heatmap(pvt_non_balanced, cmap='Blues', ax=axs[0], vmin=min_score, vmax=max_score)
axs[0].set_title('param_class_weight = None');
sns.heatmap(pvt_balanced, cmap='Blues', ax=axs[1], vmin=min_score, vmax=max_score)
axs[1].set_title('param_class_weight = balanced');

# %% [markdown]
# We plot again the heatmap with 'balanced' class weights:

# %%
cv_results_best = cv_results_df[
    (cv_results_df['param_class_weight'] == 'balanced') &
    (cv_results_df['param_criterion'] == 'entropy')
]
fig, axs = plt.subplots(1, 1, figsize=(8, 5))
pvt = pd.pivot_table(
    cv_results_best,
    values='mean_test_score',
    index=['param_min_samples_leaf'],
    columns=['param_min_samples_split']
)
sns.heatmap(pvt, cmap='Blues', ax=axs)
axs.set_title('param_class_weight = balanced; param_criterion = entropy');

# %% [markdown]
# We disaply the performance of the top 10 models:

# %%
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %% [markdown]
# We refit the best model on the whole training set:

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model = DecisionTreeClassifier(**best_model_params)

# fit the model on all the training data
fit_start = time()
best_model.fit(indicators_train_df, true_labels_train)
fit_time = time()-fit_start

# get the predictions on the training data
train_score_start = time()
pred_labels_train = best_model.predict(indicators_train_df)
train_score_time = time()-train_score_start
pred_probas_train = best_model.predict_proba(indicators_train_df)

# get the predictions on the test data
test_score_start = time()
pred_labels_test = best_model.predict(indicators_test_df)
test_score_time = time()-test_score_start
pred_probas_test = best_model.predict_proba(indicators_test_df)

# save the predictions
pd.DataFrame(
    {'labels': pred_labels_test, 'probs': pred_probas_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_preds.csv')

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}.pkl', 'wb')
pickle.dump(obj=best_model, file=file)
file.close()

# save the cv results
best_model_cv_results = pd.DataFrame(cv_results_df.iloc[best_index]).T
best_model_cv_results.index = [clf_name]
best_model_cv_results.to_csv(f'{RESULTS_DIR}/{clf_name}_train_cv_scores.csv')

# %% [markdown]
# We display traning and test scores:

# %%
compute_clf_scores(
    y_true=true_labels_train,
    y_pred=pred_labels_train,
    train_time=fit_time,
    score_time=train_score_time,
    params=best_model_params,
    prob_pred=pred_probas_train,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_train_scores.csv'
)

# %%
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    train_time=fit_time,
    score_time=test_score_time,
    params=best_model_params,
    prob_pred=pred_probas_test,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_test_scores.csv'
)
test_scores

# %% [markdown]
# We load the dataset randomly oversampled:

# %%
indicators_over_train_df = pd.read_csv('../data/clf_indicators_train_over.csv', index_col=0)
indicators_over_train_df = indicators_over_train_df[features_for_clf]
true_labels_over_train = pd.read_csv('../data/clf_y_train_over.csv', index_col=0).values.ravel()

# %% [markdown]
# We fit and test the best model on the oversampled dataset:

# %%
# fit the model on all the training data
best_model_over = DecisionTreeClassifier(**best_model_params)
fit_start = time()
best_model_over.fit(indicators_over_train_df, true_labels_over_train)
fit_over_time = time()-fit_start

# get the predictions on the training data
train_score_start = time()
pred_labels_over_train = best_model_over.predict(indicators_over_train_df)
train_score_over_time = time()-train_score_start
pred_probas_over_train = best_model_over.predict_proba(indicators_over_train_df)

# get the predictions on the test data
test_score_start = time()
pred_labels_over_test = best_model_over.predict(indicators_test_df)
test_score_over_time = time()-test_score_start
pred_probas_over_test = best_model_over.predict_proba(indicators_test_df)

# save the predictions
pd.DataFrame(
    {'labels': pred_labels_over_test, 'probs': pred_probas_over_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_oversample_preds.csv')

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}_oversample.pkl', 'wb')
pickle.dump(obj=best_model_over, file=file)
file.close()

# %% [markdown]
# We load the dataset oversampled with SMOTE:

# %%
indicators_smote_train_df = pd.read_csv('../data/clf_indicators_train_smote.csv', index_col=0)
indicators_smote_train_df = indicators_smote_train_df[features_for_clf]
true_labels_smote_train = pd.read_csv('../data/clf_y_train_smote.csv', index_col=0).values.ravel()

# %% [markdown]
# We train and test the best model on the SMOTE dataset:

# %%
# fit the model on all the training data
best_model_smote = DecisionTreeClassifier(**best_model_params)
fit_start = time()
best_model_smote.fit(indicators_smote_train_df, true_labels_smote_train)
fit_smote_time = time()-fit_start

# get the predictions on the training data
train_score_start = time()
pred_labels_smote_train = best_model_smote.predict(indicators_smote_train_df)
train_score_smote_time = time()-train_score_start
pred_probas_smote_train = best_model_smote.predict_proba(indicators_smote_train_df)

# get the predictions on the test data
test_score_start = time()
pred_labels_smote_test = best_model_smote.predict(indicators_test_df)
test_score_smote_time = time()-test_score_start
pred_probas_smote_test = best_model_smote.predict_proba(indicators_test_df)

# save the predictions
pd.DataFrame(
    {'labels': pred_labels_smote_test, 'probs': pred_probas_smote_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_smote_preds.csv')

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}_smote.pkl', 'wb')
pickle.dump(obj=best_model_smote, file=file)
file.close()

# %% [markdown]
# We compare the performance of the best model on the three datasets:

# %%
test_over_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_over_test,
    train_time=fit_over_time,
    score_time=test_score_over_time,
    params=best_model_params,
    prob_pred=pred_probas_over_test,
    clf_name=clf_name+' over',
    path=f'{RESULTS_DIR}/{clf_name}_over_test_scores.csv'
)

test_smote_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_smote_test,
    train_time=fit_smote_time,
    score_time=test_score_smote_time,
    params=best_model_params,
    prob_pred=pred_probas_smote_test,
    clf_name=clf_name+' SMOTE',
    path=f'{RESULTS_DIR}/{clf_name}_smote_test_scores.csv'
)

pd.concat([test_scores, test_over_scores, test_smote_scores])

# %% [markdown]
# We load the dataset with missing values:

# %%
# load the training data with nan values
incidents_train_nan_df = pd.read_csv('../data/clf_indicators_train_nan.csv', index_col=0)
true_labels_train_nan_df = pd.read_csv('../data/clf_y_train_nan.csv', index_col=0)
true_labels_train_nan = true_labels_train_nan_df.values.ravel()

# project on the features_to_use
indicators_train_nan_df = incidents_train_nan_df[features_for_clf]
best_model_nan = DecisionTreeClassifier(**best_model_params)

# fit the model on all the training data
fit_start_nan = time()
best_model_nan.fit(indicators_train_nan_df, true_labels_train_nan)
fit_time_nan = time()-fit_start_nan

# get the predictions on the training data
train_score_start_nan = time()
pred_labels_train_nan = best_model_nan.predict(indicators_train_nan_df)
train_score_time_nan = time()-train_score_start_nan
pred_probas_train_nan = best_model_nan.predict_proba(indicators_train_nan_df)

# get the predictions on the test data
test_score_start_nan = time()
pred_labels_test_nan = best_model_nan.predict(indicators_test_df)
test_score_time_nan = time()-test_score_start_nan
pred_probas_test_nan = best_model_nan.predict_proba(indicators_test_df)

# save the predictions
pd.DataFrame(
    {'labels': pred_labels_test_nan, 'probs_True': pred_probas_test_nan[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_nan_preds.csv')

# %% [markdown]
# We compare the performance of the best model:

# %%
test_scores_nan = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test_nan,
    train_time=fit_time_nan,
    score_time=test_score_time_nan,
    params=best_model_params,
    prob_pred=pred_probas_test_nan,
    clf_name=clf_name + ' nan',
    path=f'{RESULTS_DIR}/{clf_name}_nan_test_scores.csv'
)
pd.concat([test_scores, test_scores_nan])

# %% [markdown]
# We display the decision tree:

# %%
dot_data = export_graphviz(
    best_model,
    out_file=None, 
    feature_names=list(indicators_train_df.columns),
    filled=True,
    rounded=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# %% [markdown]
# We display the feature importances:

# %%
fig, axs = plt.subplots(1, 1, figsize=(5, 8))
display_feature_importances(
    feature_names=best_model.feature_names_in_,
    feature_importances=best_model.feature_importances_,
    axs=axs,
    title=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_feature_importances.csv'
)

# %% [markdown]
# We display confusion matrices:

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title=clf_name
)

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_smote_test,
    title=clf_name + ' SMOTE'
)

# %% [markdown]
# We plot incidents (actual class and predicted class) in different feature spaces:

# %%
plot_predictions_in_features_space(
    df=incidents_test_df,
    features=['n_males_prop', 'n_child_prop', 'n_teen_prop', 'n_participants', 'poverty_perc'],
    true_labels=true_labels_test,
    pred_labels=pred_labels_test,
    figsize=(15, 50)
)

# %% [markdown]
# We plot the ROC curve:

# %%
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

# %% [markdown]
# We plot the decision boundaries:

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_PCA_decision_boundary(
  train_set=indicators_train_df,
  features=[col for col in indicators_train_df.columns if col not in categorical_features],
  train_label=true_labels_train,
  classifier=best_model,
  classifier_name=clf_name,
  axs=axs,
  scale=True,
  pca=True
)

# %% [markdown]
# We plot the learning curve:

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_learning_curve(
    classifier=best_model,
    classifier_name=clf_name,
    train_set=indicators_train_df,
    labels=true_labels_train,
    ax=axs,
    train_sizes=np.linspace(0.1, 1.0, 5),
    metric='f1'
)

# %% [markdown]
# We plot the performance of the best model varying the complexity parameters:

# %%
param_of_interest = 'min_samples_split'
fixed_params = best_model_params.copy()
if fixed_params['min_samples_leaf']==1:
    fixed_params['min_samples_leaf'] = 0
fixed_params.pop(param_of_interest)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_scores_varying_params(
    cv_results_df,
    param_of_interest,
    fixed_params,
    'F1',
    axs,
    title=clf_name
)

# %%
param_of_interest = 'min_samples_leaf'
fixed_params = best_model_params.copy()
if fixed_params['min_samples_split']==2:
    fixed_params['min_samples_split'] = 0
fixed_params.pop(param_of_interest)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_scores_varying_params(
    cv_results_df,
    param_of_interest,
    fixed_params,
    'F1',
    axs,
    title=clf_name
)

# %% [markdown]
# We plot the distribution of the features for misclassified incidents:

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'n_killed',
    'bar',
    title='n_killed distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'suicide',
    'pie',
    title='suicide distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'incident_characteristics1',
    'pie',
    title='incident_characteristics1 distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'incident_characteristics2',
    'pie',
    pie_perc_threshold=2,
    figsize=(20, 5),
    title='incident_characteristics2 distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'location_imp',
    'hist',
    bins=5,
    title='location_imp distribution'
)


