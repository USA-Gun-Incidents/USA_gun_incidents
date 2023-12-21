# %%
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
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

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

# load the names of the features to use for the classification task
features_for_clf = json.loads(open('../data/clf_indicators_subset.json').read())

# project on the features_to_use
indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# %%
clf_name = 'DecisionTreeClassifier'
param_grid = {
    "criterion": ["entropy", "gini", "log_loss"], # TODO: esplorarne altri
    "max_depth": [2, 3, 5, 6, 7, 10, 12]
}
# TODO: capire se ci vuole troppe e in caso usare randomized
grid = GridSearchCV( # uses a stratified 5-fold cv to validate the models
    DecisionTreeClassifier(),
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5,
    refit=False
)
grid.fit(indicators_train_df, true_labels_train_df)

# %%
# TODO: scegliere il miglior modello in modo più accurato, e.g.
# - differisce tanto dagli altri? scikit learn offre qualcosa per test statistici?
# - è in overfitting?

# %%
cv_results_df = pd.DataFrame(grid.cv_results_)
best_index = grid.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model_train_score = cv_results_df.loc[best_index]['mean_test_score']
best_model = DecisionTreeClassifier(**best_model_params)
# fit the model on all the training data
fit_start = time()
best_model.fit(indicators_train_df, true_labels_train)
fit_time = time()-fit_start
# get the predictions on the test data
test_score_start = time()
pred_labels_test = best_model.predict(indicators_test_df)
test_score_time = time()-test_score_start
pred_probas_test = best_model.predict_proba(indicators_test_df)
# save the predictions
pd.DataFrame(
    {'labels': pred_labels_test, 'probs_True': pred_probas_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_preds.csv')

# %%
compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    train_score=best_model_train_score,
    train_score_name='f1_score_train', # TODO: migliorare formatting
    train_time=fit_time,
    score_time=test_score_time,
    params=best_model_params,
    prob_pred=pred_probas_test,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_scores.csv'
)

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

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title=clf_name
)

# %%
plot_predictions_in_features_space(
    df=incidents_test_df,
    features=['n_males_prop', 'n_child_prop', 'n_participants'], # TODO: farlo con features significativve
    true_labels=true_labels_test,
    pred_labels=pred_labels_test,
    figsize=(15, 15)
)

# %%
plot_roc(y_trues=[true_labels_test], y_probs=[pred_probas_test[:,1]], names=[clf_name])

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_PCA_decision_boundary(
  train_set=indicators_train_df,
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
param_of_interest = 'max_depth'
fixed_params = best_model_params.copy()
fixed_params.pop(param_of_interest)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_scores_varying_params(
    grid.cv_results_,
    param_of_interest,
    fixed_params,
    'F1', # TODO: specificare l'averaging che usa
    axs,
    title=clf_name
)

# %%
plot_distribution_missclassifications(true_labels_test, pred_labels_test, incidents_test_df, 'n_killed', 'bar')

# %%
plot_distribution_missclassifications(true_labels_test, pred_labels_test, incidents_test_df, 'suicide', 'pie')

# %%
plot_distribution_missclassifications(true_labels_test, pred_labels_test, incidents_test_df, 'incident_characteristics1', 'pie')

# %%
plot_distribution_missclassifications(true_labels_test, pred_labels_test, incidents_test_df, 'incident_characteristics2', 'pie', pie_perc_threshold=2, figsize=(20, 5))

# %%
plot_distribution_missclassifications(true_labels_test, pred_labels_test, incidents_test_df, 'location_imp', 'hist', bins=5)


