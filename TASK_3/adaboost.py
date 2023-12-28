# %% [markdown]
# # AdaBoost
# 
# AdaBoost is a bosting algorithm.
# 
# Boosting is a general method for improving the accuracy of any given learning algorith
# Boosting refers to a general and provably effective method of producing a very accurate prediction rule by combining rough and moderately inaccurate rules of thumb in a manner similar to
# that suggested above. 
# 
# AdaBoost algorithm takes as input a training set belongs to some domain and label in some label set. 
# The algorithm originally presented works for two calss classification, but it can be extended to the multiclass case. 
# AdaBoost calls a given weak or base learning algorithm repeatedly in a series of rounds. One of the main ideas of the algorithm is to maintain a distribution or set of weights over the training set.
# Initially, all weights are set equally, but on each round, the weights of incorrectly classified examples are increased so that the weak learner is forced to focus on the hard examples in the training set.
# The weak learner’s job is to find a weak hypothesis h appropriate for the distribution.

# %% [markdown]
# We import libraries

# %%
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from classification_utils import compute_clf_scores, plot_confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

# %% [markdown]
# We load data

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

# names of the features to use for the classification task
features_for_clf = [
    'location_imp', 'latitude', 'longitude', 'state_code', 'congressional_district', 
    'age_range', 'avg_age', 'n_child_prop', 'n_teen_prop', 'n_males_prop', 'n_participants', 
    'day', 'day_of_week', 'month', 'poverty_perc', 'democrat', 
    'aggression', 'accidental', 'defensive', 'suicide', 'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]

# project on the features_to_use
indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# %% [markdown]
# ## Grid search

# %% [markdown]
# 'SAMME.R' real boosting algorithm. estimator must support calculation of class probabilities. 
# 'SAMME' then use the SAMME discrete boosting algorithm. 
# 
# The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
# 
# Both algorithms were presented in ''Multi-class AdaBoost' by Ji Zhu, Saharon Rosset, Hui Zou, Trevor Hastie.

# %%
RANDOM_STATE = 42

param_grid = {
    'estimator': [None], # defualt: DecisionTreeClassifier,
    'n_estimators': [500, 1000],
    'learning_rate': [1, 0.9, 0.8],
    'algorithm': ['SAMME', 'SAMME.R'],
    'random_state': [RANDOM_STATE]
}

gs = GridSearchCV(
    AdaBoostClassifier(),
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5, # uses a stratified 5-fold cv to validate the models
    refit=False
)
gs.fit(indicators_train_df, true_labels_train)

# %%
best_index = gs.best_index_
cv_results_df = pd.DataFrame(gs.cv_results_)

best_model_params = cv_results_df.loc[best_index]['params']
best_model_params 

# %% [markdown]
# ## Classification

# %% [markdown]
# Make prediction using best model

# %%
best_model = AdaBoostClassifier(**best_model_params)
best_model.fit(indicators_train_df, true_labels_train)

# %%
pred_labels_test = best_model.predict(indicators_test_df)

# %%
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    params=best_model_params,
    train_time=None,
    score_time=None,
    prob_pred=None,
    clf_name='AdaBoost',
)
test_scores

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title='AdaBoost'
)


