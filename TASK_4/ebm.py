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
param_grid = { # TODO: scegliere (per ora default, se non li mette non funziona)
    'feature_names' : [features_for_clf],
    'feature_types': [None],
    'max_bins': [256],
    'max_interaction_bins': [32],
    'interactions': [10],
    'exclude': [[]],
    'validation_size': [0.15],
    'outer_bags': [8],
    'inner_bags': [0],
    'learning_rate': [0.01],
    'greediness': [0.0],
    'smoothing_rounds': [0],
    'max_rounds': [5000],
    'early_stopping_rounds': [50],
    'early_stopping_tolerance': [0.0001],
    'min_samples_leaf': [2],
    'max_leaves': [3],
    'objective': ['log_loss'],
    'n_jobs': [- 2], 
    'random_state': [42]
}

gs = GridSearchCV(
    ExplainableBoostingClassifier(),
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
# TODO: heatmap

# %%
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model = ExplainableBoostingClassifier(**best_model_params)

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
ebm_test_scores_df = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    train_time=fit_time,
    score_time=test_score_time,
    params=best_model_params,
    prob_pred=pred_probas_test,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_test_scores.csv'
)
xgb_test_scores_df = pd.read_csv(f'{RESULTS_DIR}/XGBClassifier_test_scores.csv', index_col=0)
pd.concat([ebm_test_scores_df, xgb_test_scores_df])

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title=clf_name
)

# %%
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

# %%
# plot_predictions_in_features_space(
#     df=incidents_test_df,
#     features=['n_males_prop', 'n_child_prop', 'n_participants'], # TODO: farlo con features significativve
#     true_labels=true_labels_test,
#     pred_labels=pred_labels_test,
#     figsize=(15, 15)
# )

# %%
# fig, axs = plt.subplots(1, 1, figsize=(10, 5))
# plot_PCA_decision_boundary(
#   train_set=indicators_train_df,
#   features=indicators_train_df.columns, # TODO: eventualmente usare solo le numeriche, togliere x, y
#   train_label=true_labels_train,
#   classifier=best_model,
#   classifier_name=clf_name,
#   axs=axs
# )

# %%
# fig, axs = plt.subplots(1, 1, figsize=(10, 5))
# plot_learning_curve(
#     classifier=best_model,
#     classifier_name=clf_name,
#     train_set=indicators_train_df,
#     labels=true_labels_train,
#     ax=axs,
#     train_sizes=np.linspace(0.1, 1.0, 5),
#     metric='f1'
# )

# %%
# TODO: plot al variare di parametri di complessità

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
ebm_global = best_model.explain_global(name='EBM Global')
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
ebm_local = best_model.explain_local(
    indicators_test_df.iloc[attempted_suicide_pos].to_frame().T,
    true_labels_test[attempted_suicide_pos].reshape(1, -1)
)
show(ebm_local)

# %%
# TODO: mass shooting

# %%
ebm_perf = ROC(best_model.predict_proba).explain_perf(incidents_test_df, true_labels_test, name='EBM')
show(ebm_perf)

# %%
non_fatal_rb_default = pd.read_csv(RESULTS_DIR+'/non_fatal_rb_default_features.csv').to_numpy()[0]
fatal_rb_default = pd.read_csv(RESULTS_DIR+'/fatal_rb_default_features.csv').to_numpy()[0]

# %%
positions_to_explain = selected_records_to_explain_df['positions'].to_list()
instance_names_to_explain = selected_records_to_explain_df['instance names'].to_list()
true_labels_to_explain = selected_records_to_explain_df['true labels'].to_list()

samples = indicators_test_df.iloc[positions_to_explain].values
metrics_selected_records = {}
for i in range(samples.shape[0]):
    prediction = best_model.predict(samples[i].reshape(1,-1))
    explanation = best_model.explain_local(samples[i], true_labels_test[i].reshape(1, -1))
    feature_importances = np.array(explanation._internal_obj['specific'][0]['scores'][:-10]) # TODO: le ultime feature sono combinazioni delle altre (?)
    feature_default = non_fatal_rb_default if true_labels_test[i] == 1 else fatal_rb_default
    sample_metric = evaluate_explanation(best_model, samples[i], feature_importances, feature_default)
    metrics_selected_records[instance_names_to_explain[i]] = sample_metric

metrics_selected_records_df = pd.DataFrame(metrics_selected_records).T
metrics_selected_records_df.to_csv('../data/explanation_results/ebm_metrics_selected_records.csv')
metrics_selected_records_df

# %%
random_records_to_explain_df = pd.read_csv('../data/explanation_results/random_records_to_explain.csv', index_col=0)
positions_to_explain = random_records_to_explain_df['positions'].to_list()
true_labels_to_explain = random_records_to_explain_df['true labels'].to_list()

samples = indicators_test_df.iloc[positions_to_explain].values
faithfulness = []
for i in range(samples.shape[0]):
    prediction = best_model.predict(samples[i].reshape(1,-1))
    explanation = best_model.explain_local(samples[i], true_labels_test[i].reshape(1, -1))
    feature_importances = np.array(explanation._internal_obj['specific'][0]['scores'][:-10]) # TODO: le ultime feature sono combinazioni delle altre (?)
    feature_default = non_fatal_rb_default if true_labels_test[i] == 1 else fatal_rb_default
    sample_metric = evaluate_explanation(best_model, samples[i], feature_importances, feature_default)
    faithfulness.append(sample_metric['faithfulness'])

metrics_random_records = {}
metrics_random_records['mean faithfulness'] = np.nanmean(faithfulness)
metrics_random_records['std faithfulness'] = np.nanstd(faithfulness)
metrics_random_records_df = pd.DataFrame(metrics_random_records, index=[clf_name])
metrics_random_records_df.to_csv('../data/explanation_results/ebm_metrics_random_records.csv')
metrics_random_records_df


