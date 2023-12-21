# %%
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score
from classification_utils import *
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RANDOM_STATE = 42

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
dt = tree.DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=10,
    min_samples_split=3,
    min_samples_leaf=4
)
dt = dt.fit(indicators_train_df, true_labels_train_df)

# %%
dot_data = tree.export_graphviz(
    dt,
    out_file=None, 
    feature_names=list(indicators_train_df.columns),
    filled=True,
    rounded=True
)
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

# %%
pred_labels_train = dt.predict(indicators_train_df)
pred_probas_train = dt.predict_proba(indicators_train_df)
pred_labels_test = dt.predict(indicators_test_df)
pred_probas_test = dt.predict_proba(indicators_test_df)

# %%
print(classification_report(y_true=true_labels_train_df, y_pred=pred_labels_train))

# %%
print(classification_report(y_true=true_labels_test_df, y_pred=pred_labels_test))

# %%
from scipy.stats import randint

dt_param_grid = {
    "criterion": ["entropy", "gini", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [2, 3, 5, 6, 7, 10, 12, None], # If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    "min_samples_split": randint(2, 51), # also float
    "min_samples_leaf": randint(1, 51), # also float
    # min_weight_fraction_leaf
    # "max_features": [None, 2, 3, 4, 5]
    #'random_state': RANDOM_STATE # ??
    #'max_leaf_nodes'
    #'min_impurity_decrease'
    #'class_weight'
    #'ccp_alpha'
    }

n_iter_search = 500 # number of random combinations to try
n_jobs_search = -1 # use all cores
n_folds = 5 # number of folds for cross validation
best_score = make_scorer(f1_score) # score to use for selecting the best model

dt_grid = RandomizedSearchCV(
    tree.DecisionTreeClassifier(),
    param_distributions=dt_param_grid,
    n_iter=n_iter_search,
    n_jobs=n_jobs_search,
    scoring=best_score,
    verbose=10,
    cv=n_folds
)
dt_grid.fit(indicators_train_df, true_labels_train_df)
print(dt_grid.best_params_)

# %%
dt = tree.DecisionTreeClassifier(**dt_grid.best_params_).fit(indicators_train_df, true_labels_train_df)

# %%
dot_data = tree.export_graphviz(
    dt,
    out_file=None, 
    feature_names=list(indicators_train_df.columns),
    filled=True,
    rounded=True
)
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

# %%
plot_confusion_matrix(y_true=true_labels_test, y_pred=pred_labels_test)

# %%
plot_predictions_in_features_space(
    incidents_test_df,
    ['n_males_prop', 'n_child_prop', 'n_participants'],
    true_labels_test,
    pred_labels_test,
    figsize=(15, 15)
)

# %%
y_trues = [true_labels_test, true_labels_test]
y_probs = [pred_probas_test[:,1], pred_probas_test[:,1]]
names = ['DT 1', 'DT 2']
plot_roc(y_trues, y_probs, names)

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plot_PCA_decision_boundary(indicators_train_df, true_labels_train, dt, 'DT', axs=axs[0])
plot_PCA_decision_boundary(indicators_train_df, true_labels_train, dt, 'DT', axs=axs[1])

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plot_learning_curve(dt, 'DT', indicators_train_df, true_labels_train, axs[0])
plot_learning_curve(dt, 'DT', indicators_train_df, true_labels_train, axs[1])

# %%
dt_param_grid = {
    "criterion": ["entropy", "gini", "log_loss"],
    "max_depth": [2, 3, 5, 6, 7, 10, 12]
}

dt_grid = GridSearchCV(
    tree.DecisionTreeClassifier(),
    param_grid=dt_param_grid,
    n_jobs=-1, # default cv = stratified 5-fold
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5
)
dt_grid.fit(indicators_train_df, true_labels_train_df)

# %%
param_of_interest = 'max_depth'
fixed_params = dt_grid.best_params_
fixed_params.pop(param_of_interest)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_scores_varying_params(dt_grid.cv_results_, param_of_interest, fixed_params, 'F1 Score', axs)
plot_scores_varying_params(dt_grid.cv_results_, param_of_interest, {'criterion': 'entropy'}, 'F1 Score', axs, title=f'F1 score varying {param_of_interest}')

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


