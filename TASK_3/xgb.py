# %%
import pandas as pd
import numpy as np
import json
from classification_utils import *
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier
from xgboost import plot_tree
import pickle
RESULTS_DIR = '../data/classification_results'
RANDOM_STATE = 42
clf_name = 'XGBClassifier'

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
# Parameters explored:
# - min_child_weight: Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be (default=1).
# - subsample: Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration (default=1).
# - scale_pos_weight: Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances) (default=1).
# - colsample_bytree: is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed (default=1).
# - colsample_bylevel: is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree (default=1).
# - colsample_bynode: is the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level (default=1).
# - max_depth: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth (default=6).
# 
# Fixed parameters (many other parameters can be tuned, here we report only the most important one):
# - eta: Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative (default=0.3).

# %%
# TODO: decidere quali parametri esplorare (https://xgboost.readthedocs.io/en/stable/parameter.html)
cv_train_size = (4/5)*(indicators_train_df.shape[0])
num_pos_inst = np.unique(true_labels_train, return_counts=True)[1][1]
num_neg_inst = np.unique(true_labels_train, return_counts=True)[1][0]
param_grid = {
    'eta': [0.3], # 0.5'
    'min_child_weight': [1, int(0.01*cv_train_size), int(0.025*cv_train_size), int(0.05*cv_train_size), int(0.1*cv_train_size)],
    'subsample': [0.5, 1],
    'scale_pos_weight': [1, num_neg_inst / num_pos_inst],
    'colsample_bytree': [1],
    'colsample_bylevel': [1],
    'colsample_bynode': [1, np.sqrt(len(features_for_clf)/len(features_for_clf))],
    'max_depth': [4, 6, 8]
    # TODO: cosa fare per categorical data https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
}

gs = GridSearchCV(
    XGBClassifier(),
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5, # uses a stratified 5-fold cv to validate the models
    refit=False
)
gs.fit(indicators_train_df, true_labels_train)

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
cv_results_df.head()

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
pvt_balanced = pd.pivot_table(
    cv_results_df[(cv_results_df['param_scale_pos_weight'] == 1)],
    values='mean_test_score',
    index=['param_min_child_weight', 'param_subsample'],
    columns=['param_colsample_bynode', 'param_max_depth']
)
pvt_non_balanced = pd.pivot_table(
    cv_results_df[(cv_results_df['param_scale_pos_weight'] != 1)],
    values='mean_test_score',
    index=['param_min_child_weight', 'param_subsample'],
    columns=['param_colsample_bynode', 'param_max_depth']
)
min_score = cv_results_df['mean_test_score'].min()
max_score = cv_results_df['mean_test_score'].max()
sns.heatmap(pvt_non_balanced, cmap='Blues', ax=axs[0], vmin=min_score, vmax=max_score)
axs[0].set_title('param_scale_pos_weight = 1');
sns.heatmap(pvt_balanced, cmap='Blues', ax=axs[1], vmin=min_score, vmax=max_score)
axs[1].set_title('param_scale_pos_weight = num_neg_inst / num_pos_inst');

# %%
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model = XGBClassifier(**best_model_params)

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
compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    train_time=fit_time,
    score_time=test_score_time,
    params=best_model_params,
    prob_pred=pred_probas_test,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_test_scores.csv'
)

# %%
fig, axs = plt.subplots(figsize=(20, 50))
plot_tree(best_model, num_trees=0, ax=axs)

# %%
fig, axs = plt.subplots(figsize=(20, 40))
plot_tree(best_model, num_trees=1, ax=axs)

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title=clf_name
)

# %%
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

# %%
plot_predictions_in_features_space(
    df=incidents_test_df,
    features=['n_males_prop', 'n_child_prop', 'n_participants'], # TODO: farlo con features significativve
    true_labels=true_labels_test,
    pred_labels=pred_labels_test,
    figsize=(15, 15)
)

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_PCA_decision_boundary(
  train_set=indicators_train_df,
  features=indicators_train_df.columns, # TODO: eventualmente usare solo le numeriche, togliere x, y
  train_label=true_labels_train,
  classifier=best_model,
  classifier_name=clf_name,
  axs=axs
)

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

# %%
# TODO: plot al variare di parametri di complessità

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
display_feature_importances(
    feature_names=indicators_train_df.columns,
    feature_importances=best_model.feature_importances_,
    axs=axs,
    title=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_feature_importances.csv'
)

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


