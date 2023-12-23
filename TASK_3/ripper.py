# %%
import pandas as pd
import json
import wittgenstein as lw
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from time import time
import pickle
from classification_utils import *
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RESULTS_DIR = '../data/classification_results'
clf_name = 'RipperClassifier'

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
ripper = lw.RIPPER()
param_grid = { # TODO: explore
    'prune_size':[0.5],
    'k': [1]
    #'prune_size': [0.5, 0.6], # the fraction of rules to prune
    #'k': [1, 3, 5] # the number of optimization runs, maybe 1,2,4?, "prune_size": [0.33, 0.5, 0.8], "k": [1, 2]}?
}
# considers replacing each rule with a completely new grown-and-pruned replacement, as well as a grown-and-pruned revision of the original
gs = GridSearchCV(
    estimator=ripper,
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5,
    refit=True
)
gs.fit(indicators_train_df, true_labels_train)

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
cv_results_df.head()

# %%
best_index = gs.best_index_
ripper = gs.best_estimator_
ripper_params = gs.best_params_
fit_time = gs.refit_time_

# %%
ripper.out_model()

# %%
# get the predictions on the training data
train_score_start = time()
pred_labels_train = ripper.predict(indicators_train_df)
train_score_time = time()-train_score_start
pred_probas_train = ripper.predict_proba(indicators_train_df)

# get the predictions on the test data
test_score_start = time()
pred_labels_rules_test = ripper.predict(indicators_test_df,  give_reasons=True)
test_score_time = time()-test_score_start
pred_labels_test = pred_labels_rules_test[0]
pred_rules_test = pred_labels_rules_test[1]
rule_list_str = [str(rule) for rule in pred_rules_test]
pd.DataFrame(rule_list_str).to_csv(f'{RESULTS_DIR}/{clf_name}_rules.csv')
pred_probas_test = ripper.predict_proba(indicators_test_df,  give_reasons=False)

# save the predictions
pd.DataFrame(
    {'labels': pred_labels_test, 'probs': pred_probas_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_preds.csv')

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}.pkl', 'wb')
pickle.dump(obj=ripper, file=file)
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
    params=ripper_params,
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
    params=ripper_params,
    prob_pred=pred_probas_test,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_test_scores.csv'
)

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
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

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


