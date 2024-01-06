# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
# 
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
# 
# # Explainable Boosting Machine (EBM)
# 
# We import the libraries and define constants and settings of the notebook:

# %%
import pandas as pd
import numpy as np
import json
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from interpret.perf import ROC
from scipy.stats import rankdata
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
import pickle
import os, sys
sys.path.append(os.path.abspath('../TASK_3/'))
from classification_utils import *
from explanation_utils import *
RESULTS_DIR = '../data/classification_results'
RANDOM_STATE = 42
clf_name = 'ExplainableBoostingMachineClassifier'

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

# %%
params = {
    'feature_names' : features_for_clf,
    'feature_types': None,
    'max_bins': 256,
    'max_interaction_bins': 32,
    'interactions': 10,
    'exclude': [],
    'validation_size': 0.15,
    'outer_bags': 8,
    'inner_bags': 0,
    'learning_rate': 0.01,
    'greediness': 0.0,
    'smoothing_rounds': 0,
    'max_rounds': 5000,
    'early_stopping_rounds': 50,
    'early_stopping_tolerance': 0.0001,
    'min_samples_leaf': 2,
    'max_leaves': 3,
    'objective': 'log_loss',
    'n_jobs': - 2, 
    'random_state': 42
}

ebm = ExplainableBoostingClassifier(**params)

# fit the model on all the training data
fit_start = time()
ebm.fit(indicators_train_df, true_labels_train)
fit_time = time()-fit_start

# get the predictions on the training data
train_score_start = time()
pred_labels_train = ebm.predict(indicators_train_df)
train_score_time = time()-train_score_start
pred_probas_train = ebm.predict_proba(indicators_train_df)

# get the predictions on the test data
test_score_start = time()
pred_labels_test = ebm.predict(indicators_test_df)
test_score_time = time()-test_score_start
pred_probas_test = ebm.predict_proba(indicators_test_df)

# save the predictions
pd.DataFrame(
    {'labels': pred_labels_test, 'probs': pred_probas_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_preds.csv')

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}.pkl', 'wb')
pickle.dump(obj=ebm, file=file)
file.close()

# %% [markdown]
# We display traning and test scores:

# %%
compute_clf_scores(
    y_true=true_labels_train,
    y_pred=pred_labels_train,
    train_time=fit_time,
    score_time=train_score_time,
    params=params,
    prob_pred=pred_probas_train,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_train_scores.csv'
)

# %%
ebm_test_scores_df = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    train_time=fit_time,
    score_time=test_score_time,
    params=params,
    prob_pred=pred_probas_test,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_test_scores.csv'
)
xgb_test_scores_df = pd.read_csv(f'{RESULTS_DIR}/ExtremeGradientBoostingClassifier_test_scores.csv', index_col=0)
pd.concat([ebm_test_scores_df, xgb_test_scores_df])

# %% [markdown]
# We display confusion matrices:

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title=clf_name
)

# %% [markdown]
# We plot the ROC curve:

# %%
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

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

# %% [markdown]
# ## Global Interpretation

# %%
ebm_global = ebm.explain_global(name='EBM Global')
show(ebm_global)

# %%
# save feature importance in csv
feature_names = ebm_global.data()['names']
feature_importances = ebm_global.data()['scores']
sorted_idx = np.flip(np.argsort(feature_importances))
sorted_feature_names = [feature_names[i] for i in sorted_idx]
sorted_feature_imp = [feature_importances[i] for i in sorted_idx]
pd.DataFrame({
    'features': sorted_feature_names,
    'importances': sorted_feature_imp,
    'rank': rankdata([-imp for imp in sorted_feature_imp], method='dense')
}).to_csv(f'{RESULTS_DIR}/{clf_name}_feature_importances.csv')

# %% [markdown]
# ## Local Interpretation

# %%
selected_records_to_explain_df = pd.read_csv('../data/explanation_results/selected_records_to_explain.csv', index_col=0)
attempted_suicide_pos = selected_records_to_explain_df[selected_records_to_explain_df['instance names']=='Attempted Suicide']['positions'][0]

# %%
ebm_local = ebm.explain_local(
    indicators_test_df.iloc[attempted_suicide_pos].to_frame().T,
    true_labels_test[attempted_suicide_pos].reshape(1, -1)
)
show(ebm_local)

# %%
mass_shooting_pos = selected_records_to_explain_df[selected_records_to_explain_df['instance names']=='Mass shooting']['positions']

# %%
ebm_local = ebm.explain_local(
    indicators_test_df.iloc[mass_shooting_pos],
    true_labels_test[mass_shooting_pos].reshape(1, -1)
)
show(ebm_local)

# %% [markdown]
# # Evaluation

# %%
non_fatal_rb_default = pd.read_csv(RESULTS_DIR+'/non_fatal_rb_default_features.csv').to_numpy()[0]
fatal_rb_default = pd.read_csv(RESULTS_DIR+'/fatal_rb_default_features.csv').to_numpy()[0]

# %%
positions_to_explain = selected_records_to_explain_df['positions'].to_list()
instance_names_to_explain = selected_records_to_explain_df['instance names'].to_list()
true_labels_to_explain = selected_records_to_explain_df['true labels'].to_list()

instances = indicators_test_df.iloc[positions_to_explain].values
metrics_selected_records = {}
for i in range(instances.shape[0]):
    prediction = ebm.predict(instances[i].reshape(1,-1))
    explanation = ebm.explain_local(instances[i], true_labels_test[i].reshape(1, -1))
    feature_importances = np.array(explanation._internal_obj['specific'][0]['scores'][:-10])
    feature_default = non_fatal_rb_default if true_labels_test[i] == 1 else fatal_rb_default
    sample_metric = evaluate_explanation(ebm, instances[i], feature_importances, feature_default)
    metrics_selected_records[instance_names_to_explain[i]] = sample_metric

metrics_selected_records_df = pd.DataFrame(metrics_selected_records).T
metrics_selected_records_df.to_csv('../data/explanation_results/ebm_metrics_selected_records.csv')
metrics_selected_records_df

# %%
random_records_to_explain_df = pd.read_csv('../data/explanation_results/random_records_to_explain.csv', index_col=0)
positions_to_explain = random_records_to_explain_df['positions'].to_list()
true_labels_to_explain = random_records_to_explain_df['true labels'].to_list()

instances = indicators_test_df.iloc[positions_to_explain].values
faithfulness = []
for i in range(instances.shape[0]):
    prediction = ebm.predict(instances[i].reshape(1,-1))
    explanation = ebm.explain_local(instances[i], true_labels_test[i].reshape(1, -1))
    feature_importances = np.array(explanation._internal_obj['specific'][0]['scores'][:-10])
    feature_default = non_fatal_rb_default if true_labels_test[i] == 1 else fatal_rb_default
    sample_metric = evaluate_explanation(ebm, instances[i], feature_importances, feature_default)
    faithfulness.append(sample_metric['faithfulness'])

metrics_random_records = {}
metrics_random_records['mean faithfulness'] = np.nanmean(faithfulness)
metrics_random_records['std faithfulness'] = np.nanstd(faithfulness)
metrics_random_records_df = pd.DataFrame(metrics_random_records, index=[clf_name])
metrics_random_records_df.to_csv('../data/explanation_results/ebm_metrics_random_records.csv')
metrics_random_records_df


