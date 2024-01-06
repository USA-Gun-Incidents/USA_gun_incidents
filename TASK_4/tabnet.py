# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
# 
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
# 
# # TabNet
# 
# We import the libraries and define constants and settings of the notebook:

# %%
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.augmentations import ClassificationSMOTE
from sklearn.model_selection import train_test_split
import os, sys
sys.path.append(os.path.abspath('../TASK_3/'))
from classification_utils import *
from time import time
from explanation_utils import *
RESULTS_DIR = '../data/classification_results'
clf_name = 'TabNetClassifier'

# %% [markdown]
# We load the data:

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
incidents_test_df = pd.read_csv('../data/clf_scaled_indicators_test.csv', index_col=0)

true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

# load the names of the features to use for the classification task
features_for_clf = json.loads(open('../data/clf_indicators_names_distance_based.json').read())

# project on the features to use
indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# scale the data
scaler = MinMaxScaler()
indicators_train_scaled = scaler.fit_transform(indicators_train_df)
# split the data into train and validation sets
train_set, val_set, train_labels, val_labels = train_test_split(
    indicators_train_scaled,
    true_labels_train,
    test_size=0.2
)

# %%
tabnet = TabNetClassifier()
fit_start = time()
tabnet.fit(
  train_set, train_labels,
  eval_set=[(train_set, train_labels), (val_set, val_labels)],
  eval_name=['train', 'val'],
  eval_metric=['balanced_accuracy', 'logloss'],
  max_epochs=50,
  augmentations= ClassificationSMOTE(p=0.2),
  weights=1,
)
fit_time = time()-fit_start

# %%
plt.plot(tabnet.history['loss'], label='Train')
plt.xlabel('Epochs');
plt.ylabel('Training Loss');

# %%
plt.plot(tabnet.history['train_logloss'], label='Train')
plt.plot(tabnet.history['val_logloss'], label='Validation')
plt.xlabel('Epochs');
plt.ylabel('Training Loss');

# %%
plt.plot(tabnet.history['train_balanced_accuracy'], label='Train')
plt.plot(tabnet.history['val_balanced_accuracy'], label='Validation')
plt.xlabel('Epochs');
plt.ylabel('Validation Accuracy');

# %%
# get the predictions on the training data
train_score_start = time()
pred_labels_train = tabnet.predict(indicators_train_df.values)
train_score_time = time()-train_score_start
pred_probas_train = tabnet.predict_proba(indicators_train_df.values)

# get the predictions on the test data
test_score_start = time()
pred_labels_test = tabnet.predict(indicators_test_df.values)
test_score_time = time()-test_score_start
pred_probas_test = tabnet.predict_proba(indicators_test_df.values)

# save the predictions
pd.DataFrame(
   {'labels': pred_labels_test, 'probs': pred_probas_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_preds.csv')

# save the model
tabnet.save_model(f'{RESULTS_DIR}/{clf_name}.pkl')

# %%
compute_clf_scores(
    y_true=true_labels_train,
    y_pred=pred_labels_train,
    train_time=fit_time,
    score_time=train_score_time,
    params=tabnet.get_params(),
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
    params=tabnet.get_params(),
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
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

# %%
plot_predictions_in_features_space(
    df=incidents_test_df,
    features=['n_males_prop', 'n_child_prop', 'n_participants'],
    true_labels=true_labels_test,
    pred_labels=pred_labels_test,
    figsize=(15, 15)
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'n_killed',
    'bar'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'suicide',
    'pie'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'incident_characteristics1',
    'pie'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'incident_characteristics2',
    'pie',
    pie_perc_threshold=2,
    figsize=(20, 5)
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'location_imp',
    'hist',
    bins=5
)

# %% [markdown]
# ## Global interpretation

# %%
fig, axs = plt.subplots(1, 1, figsize=(7,6))
display_feature_importances(
    features_for_clf,
    tabnet.feature_importances_,
    axs,
    title='TabNetClassifier feature importances',
    path=f'{RESULTS_DIR}/{clf_name}_feature_importances.csv'
)

# %% [markdown]
# # Local interpretation
# 
# ## Attempted Suicide

# %%
selected_records_to_explain_df = pd.read_csv('../data/explanation_results/selected_records_to_explain.csv', index_col=0)
attempted_suicide_pos = selected_records_to_explain_df[selected_records_to_explain_df['instance names']=='Attempted Suicide']['positions'][0]

# %%
explanation, mask = tabnet.explain(indicators_test_df.iloc[attempted_suicide_pos].values.reshape(1,-1), normalize=False)

# %%
fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharex=True)
mask_sum = np.zeros_like(mask[0][0])
for i in range(3):
    mask_sum += mask[i][0]
    sorted_idx = np.argsort(mask[i][0])
    sorted_features_names = [features_for_clf[j] for j in sorted_idx]
    sorted_features_imp = [mask[i][0][j] for j in sorted_idx]
    axs[i].barh(y=sorted_features_names, width=sorted_features_imp)
    axs[i].set_title(f"mask {i}")
    for tick in axs[i].get_xticklabels():
        tick.set_rotation(90);
    axs[i].set_ylabel('feature importance')
fig.tight_layout()

# %%
fig, axs = plt.subplots(1, figsize=(7, 6))
sorted_idx = np.argsort(mask_sum)
sorted_features_names = [features_for_clf[j] for j in sorted_idx]
sorted_features_imp = [mask_sum[j] for j in sorted_idx]
plt.barh(y=sorted_features_names, width=sorted_features_imp)
axs.set_xlabel('feature importance (mask 1 + mask 2 + mask 3)')

# %%
sorted_idx = np.argsort(explanation[0])
sorted_features_names = [features_for_clf[j] for j in sorted_idx]
sorted_features_imp = [explanation[0][j] for j in sorted_idx]
plt.barh(y=sorted_features_names, width=sorted_features_imp)
axs.set_xlabel('feature importance (explanation)')

# %% [markdown]
# ## Mass shooting

# %%
mass_shooting_pos = selected_records_to_explain_df[selected_records_to_explain_df['instance names']=='Mass shooting']['positions']

# %%
explanation, mask = tabnet.explain(indicators_test_df.iloc[mass_shooting_pos].values.reshape(1,-1), normalize=False)

# %%
fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharex=True)
mask_sum = np.zeros_like(mask[0][0])
for i in range(3):
    mask_sum += mask[i][0]
    sorted_idx = np.argsort(mask[i][0])
    sorted_features_names = [features_for_clf[j] for j in sorted_idx]
    sorted_features_imp = [mask[i][0][j] for j in sorted_idx]
    axs[i].barh(y=sorted_features_names, width=sorted_features_imp)
    axs[i].set_title(f"mask {i}")
    for tick in axs[i].get_xticklabels():
        tick.set_rotation(90);
    axs[i].set_ylabel('feature importance')
fig.tight_layout()

# %%
fig, axs = plt.subplots(1, figsize=(7, 6))
sorted_idx = np.argsort(mask_sum)
sorted_features_names = [features_for_clf[j] for j in sorted_idx]
sorted_features_imp = [mask_sum[j] for j in sorted_idx]
plt.barh(y=sorted_features_names, width=sorted_features_imp)
axs.set_xlabel('feature importance (mask 1 + mask 2 + mask 3)')

# %%
sorted_idx = np.argsort(explanation[0])
sorted_features_names = [features_for_clf[j] for j in sorted_idx]
sorted_features_imp = [explanation[0][j] for j in sorted_idx]
plt.barh(y=sorted_features_names, width=sorted_features_imp)
axs.set_xlabel('feature importance (explanation)')

# %% [markdown]
# # Evaluation of explanations

# %%
non_fatal_default = pd.read_csv(RESULTS_DIR+'/non_fatal_db_default_features.csv').to_numpy()[0]
fatal_default = pd.read_csv(RESULTS_DIR+'/fatal_db_default_features.csv').to_numpy()[0]

# %%
positions_to_explain = selected_records_to_explain_df['positions'].to_list()
instance_names_to_explain = selected_records_to_explain_df['instance names'].to_list()
true_labels_to_explain = selected_records_to_explain_df['true labels'].to_list()

instances = indicators_test_df.iloc[positions_to_explain].values
metrics_selected_records = {}
for i in range(instances.shape[0]):
    prediction = tabnet.predict(instances[i].reshape(1,-1))
    feature_importances, _ = tabnet.explain(instances[i].reshape(1,-1), normalize=False)
    feature_default = non_fatal_default if true_labels_test[i] == 1 else fatal_default
    sample_metric = evaluate_explanation(tabnet, instances[i], feature_importances, feature_default)
    metrics_selected_records[instance_names_to_explain[i]] = sample_metric

metrics_selected_records_df = pd.DataFrame(metrics_selected_records).T
metrics_selected_records_df.to_csv('../data/explanation_results/tabnet_metrics_selected_records.csv')
metrics_selected_records_df

# %%
random_records_to_explain_df = pd.read_csv('../data/explanation_results/random_records_to_explain.csv', index_col=0)
positions_to_explain = random_records_to_explain_df['positions'].to_list()
true_labels_to_explain = random_records_to_explain_df['true labels'].to_list()

instances = indicators_test_df.iloc[positions_to_explain].values
faithfulness = []
for i in range(instances.shape[0]):
    prediction = tabnet.predict(instances[i].reshape(1,-1))
    feature_importances, _ = tabnet.explain(instances[i].reshape(1,-1), normalize=False)
    feature_default = non_fatal_default if true_labels_test[i] == 1 else fatal_default
    sample_metric = evaluate_explanation(tabnet, instances[i], feature_importances, feature_default)
    faithfulness.append(sample_metric['faithfulness'])

metrics_random_records = {}
metrics_random_records['mean faithfulness'] = np.nanmean(faithfulness)
metrics_random_records['std faithfulness'] = np.nanstd(faithfulness)
metrics_random_records_df = pd.DataFrame(metrics_random_records, index=[clf_name])
metrics_random_records_df.to_csv('../data/explanation_results/tabnet_metrics_random_records.csv')
metrics_random_records_df


