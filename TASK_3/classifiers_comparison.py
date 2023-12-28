# %%
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from classification_utils import *
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier
import os
import sys
sys.path.append(os.path.abspath('..'))
from plot_utils import *
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %%
RESULTS_DIR = '../data/classification_results/'
clf_names = ['DecisionTreeClassifier', 'RandomForestClassifier']

test_true_labels = pd.read_csv('../data/clf_y_test.csv')
test_data = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)

cv_train_scores = pd.DataFrame()
train_scores = pd.DataFrame()
test_scores = pd.DataFrame()
test_preds = {}
for clf in clf_names:
    clf_cv_train_scores = pd.read_csv(RESULTS_DIR+clf+'_train_cv_scores.csv', index_col=0)
    cv_train_scores = pd.concat([cv_train_scores, clf_cv_train_scores], axis=0)
    clf_train_scores = pd.read_csv(RESULTS_DIR+clf+'_train_scores.csv', index_col=0)
    train_scores = pd.concat([train_scores, clf_train_scores], axis=0)
    clf_test_scores = pd.read_csv(RESULTS_DIR+clf+'_test_scores.csv', index_col=0)
    test_scores = pd.concat([test_scores, clf_test_scores], axis=0)
    clf_test_preds = pd.read_csv(RESULTS_DIR+clf+'_preds.csv')
    test_preds[clf] = {}
    test_preds[clf]['labels'] = clf_test_preds['labels']
    test_preds[clf]['probs'] = clf_test_preds['probs']
    clf_feature_imp = pd.read_csv(RESULTS_DIR+clf+'_feature_importances.csv')
    test_preds[clf]['features_ranked'] = clf_feature_imp['features']
    test_preds[clf]['features_importance'] = clf_feature_imp['importances']

# %%
# TODO: ricordare di commentare che i tempi sono influenzati anche dal fatto che con alcuni classificatori
# è stato necessario fare one-hot encoding delle feature e la complessità cresce al crescere delle feature

# %% [markdown]
# Train scores comparison:

# %%
cv_train_scores[['std_test_score', 'mean_test_score']].style.background_gradient(cmap='Blues', axis=0)

# %%
train_scores.style.background_gradient(cmap='Blues', axis=0)

# %% [markdown]
# Test scores comparison:

# %%
test_scores.style.background_gradient(cmap='Blues', axis=0)

# %% [markdown]
# Cross validation performance and standard deviation:

# %%
cv_train_scores.plot.bar(figsize=(10, 5), y='mean_test_score', yerr='std_test_score')
plt.xlabel('Classifier');
plt.ylabel('F1');

# %% [markdown]
# Training time comparison:

# %%
cv_train_scores.plot.bar(figsize=(10, 5), y='mean_fit_time', yerr='std_fit_time')
plt.xlabel('Classifier');
plt.ylabel('Fit time (s)');

# %% [markdown]
# Score time comparison:

# %%
cv_train_scores.plot.bar(figsize=(10, 5), y='mean_score_time', yerr='std_score_time')
plt.xlabel('Classifier');
plt.ylabel('Score time (s)');

# %% [markdown]
# Train - Test scores comparison:

# %%
metric = 'f1-score-macro avg'
train_metric = train_scores[metric].reset_index().rename(columns={'index': 'clf', metric: 'train '+metric})
test_metric = test_scores[metric].reset_index().rename(columns={'index': 'clf', metric: 'test '+metric})
metrics = pd.merge(train_metric, test_metric, on='clf')
metrics.plot.bar(x='clf', figsize=(10, 5), title=metric)
plt.xlabel('Classifier')
plt.ylabel(metric)

# %% [markdown]
# AUROC comparison:

# %%
y_probs = [test_preds[clf]['probs'] for clf in clf_names]
plot_roc(test_true_labels['death'], y_probs, clf_names)

# %% [markdown]
# Confusion matrix comparison:

# %%
ncols = 2
nplots = len(clf_names)
nrows = int(nplots / ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,4), squeeze=False)
min = test_true_labels['death'].shape[0]
max = 0
# compute min and max to use same colorbar
cms = []
for clf in clf_names:
    cm = confusion_matrix(y_true=test_true_labels['death'], y_pred=test_preds[clf]['labels'], labels=[0,1])
    cms.append(cm)
    clf_min = cm.min()
    if clf_min < min:
        min = clf_min
    clf_max = cm.max()
    if clf_max > max:
        max = clf_max
# plot confusion matrix
for i, clf in enumerate(clf_names):
    sns.heatmap(cms[i], annot=True, fmt='g', ax=axs[int(i/ncols)][i%ncols], cmap='Blues', vmin=min, vmax=max, yticklabels=['Non-Fatal', 'Fatal'])
    axs[int(i/ncols)][i%ncols].set_title(clf);

# %% [markdown]
# Missclassification comparison:

# %%
labels = [test_preds[clf]['labels'] for clf in clf_names]
labels.append(test_true_labels['death'])
sankey_plot(
    labels=labels,
    labels_titles=clf_names+['True labels'],
    title='Classication comparison'
)

# %% [markdown]
# Feature importance comparison:

# %%
ncols = 2
nplots = len(clf_names)
nrows = int(nplots / ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,10), squeeze=False)
for i, clf in enumerate(clf_names):
    axs[int(i/ncols)][i%ncols].bar(
        x=test_preds[clf]['features_ranked'],
        height=test_preds[clf]['features_importance'],
    )
    axs[int(i/ncols)][i%ncols].set_title(clf)
    axs[int(i/ncols)][i%ncols].set_xlabel('Features')
    axs[int(i/ncols)][i%ncols].set_ylabel('Importance')
    for tick in axs[int(i/ncols)][i%ncols].get_xticklabels():
        tick.set_rotation(90);

# %% [markdown]
# Confidence comparison:

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
for i in range(len(clf_names)):
    CalibrationDisplay.from_predictions(test_true_labels['death'], y_probs[i], name=clf_names[i], ax=axs)
fig.suptitle('Calibration curves')

# %% [markdown]
# Rules comparison:

# %%
rule_based_clf_names = ['DecisionTreeClassifier', 'RandomForestClassifier']
for clf in rule_based_clf_names:
    print(clf)
    print("Incidents predicted as 'Fatal' with highest confidence")
    fatal_idx = test_preds[clf]['probs'][test_preds[clf]['labels']==1].sort_values(ascending=False).head(5).index.values
    display(test_data.iloc[fatal_idx])
    print("Incidents predicted as 'Non-Fatal' with highest confidence")
    nonfatal_idx = test_preds[clf]['probs'][test_preds[clf]['labels']==0].sort_values(ascending=True).head(5).index.values
    display(test_data.iloc[nonfatal_idx])

# %% [markdown]
# All the incidents displayed above are correctly classified by the models.
# 
# - The incidents classified as 'Fatal' with highest confidence by the Decision Tree model are all murder-suicides.
# - The incidents classifies as 'Non-Fatal' with highest confidence by the Decision Tree model are non-shooting incidents involving a male in which the tag 'illegal_holding' or the tag 'drug_alchool' is True.
# - The incidents classified as 'Fatal' with highest confidence by the Random Forest model are all murder-suicides (notes are not available but we know they are suicides and the number of killed people is grater than 1)
# - The incidents classified as 'Non-Fatal' with highest confidence by the Random Forest model are incidents involving a male in which the tag 'illegal_holding' or the tag 'drug_alchool' is True.

# %%
# TODO: estrai regole e confrontale anche con RIPPER?
with open(RESULTS_DIR+'DecisionTreeClassifier.pkl', 'rb') as file:
    dt = pickle.load(file)
with open(RESULTS_DIR+'RandomForestClassifier.pkl', 'rb') as file:
    rf = pickle.load(file)

incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
features_for_clf = json.loads(open('../data/clf_indicators_subset.json').read())
indicators_test_df = incidents_test_df[features_for_clf]

# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

# TODO: confrontare regole che usano


