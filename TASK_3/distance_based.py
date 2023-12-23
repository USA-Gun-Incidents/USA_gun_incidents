# %% [markdown]
# ### Distance based classifiers

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import statistics
import joblib
from classification_utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, f1_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
from sklearn.inspection import DecisionBoundaryDisplay
from itertools import product
from time import time

DATA_FOLDER = '../data/'
SEED = 42

# %%
incidents_df = pd.read_csv(DATA_FOLDER + 'clf_incidents_indicators.csv', index_col=0)
indicators_df = pd.read_csv(DATA_FOLDER + 'clf_indicators.csv', index_col=0)

# %%
# create is_killed column
label_name = 'is_killed'
incidents_df[label_name] = incidents_df['n_killed'].apply(lambda x: 1 if x >= 1 else 0)
indicators_df[label_name] = incidents_df[label_name]

# %%
indicators_df.isna().sum()

# %%
indicators_df.dropna(inplace=True)

indicators_df

# %%
# drop columns with categorical data since we're using distance
categorical_features = ['month', 'day_of_week']
indicators_df.drop(columns=categorical_features, inplace=True)

# %%
# if needed apply normalization
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(indicators_df.values)

# %%
X_minmax_df = pd.DataFrame(X_minmax, columns=indicators_df.columns, index=indicators_df.index) # normalized dataframe

X_minmax_df

# %%
# classes distrubution
X_minmax_df[label_name].value_counts(normalize=True)

# %%
# scatterplot

features = indicators_df.columns
features = features.drop(label_name)
scatter_by_label(
        indicators_df,
        ["age_range", "avg_age", "n_child_prop", "n_teen_prop", "n_males_prop"],
        label_name,
        figsize=(35, 60)
        )

# %%
# split dataset
label = X_minmax_df.pop(label_name)
# we apply stratification since we have unbalanced data
train_set, test_set, train_label, test_label = train_test_split(X_minmax_df, label, stratify=label, test_size=0.30, random_state=SEED)

# %%
def grid_search(clf_name, param_grid, classifier, train_set, train_labels):
    grid = GridSearchCV( # uses a stratified 5-fold cv to validate the models
        classifier,
        param_grid=param_grid,
        n_jobs=-1,
        scoring=make_scorer(f1_score),
        verbose=10,
        cv=5,
        refit=False
    )
    grid.fit(train_set, train_labels)

    return grid

# %% [markdown]
# ## KNN

# %%
knn_param_grid = {
    "n_neighbors": [1, 5, 10, 20, 50, 100],
    "algorithm": ['auto'],
    "metric": ['minkowski'],
    "p": [1, 2]
}

knn_grid = grid_search('KNN', knn_param_grid, KNeighborsClassifier(), train_set, train_label)

# %%
knn_cv_results_df = pd.DataFrame(knn_grid.cv_results_)
knn_best_index = knn_grid.best_index_
knn_best_model_params = knn_cv_results_df.loc[knn_best_index]['params']
knn_best_model_train_score = knn_cv_results_df.loc[knn_best_index]['mean_test_score']
knn_best_model = KNeighborsClassifier(**knn_best_model_params)

# %%
RESULTS_DIR = '../data/classification_results'

# fit the model on all the training data
fit_start = time()
knn_best_model.fit(train_set, train_label)
knn_fit_time = time() - fit_start

# get the predictions on the test data
test_score_start = time()
knn_pred_labels_test = knn_best_model.predict(test_set)
knn_test_score_time = time() - test_score_start
knn_pred_probas_test = knn_best_model.predict_proba(test_set)

# save the predictions
pd.DataFrame({'labels': knn_pred_labels_test, 'probs_True': knn_pred_probas_test[:,1]}).to_csv(RESULTS_DIR + '/knn_preds.csv')

# %%
knn_best_model_params

# %%
BEST_MODELS_DIR = "./best_models"

# save model
joblib.dump(knn_best_model, BEST_MODELS_DIR + '/knn.pkl')

# %%
knn_cv_results_df = pd.DataFrame(knn_grid.cv_results_)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
pvt_manhattan = pd.pivot_table(
    knn_cv_results_df[(knn_cv_results_df['param_p'] == 1)],
    values='mean_test_score',
    index=['param_n_neighbors']
)
pvt_euclidean = pd.pivot_table(
    knn_cv_results_df[(knn_cv_results_df['param_p'] == 2)],
    values='mean_test_score',
    index=['param_n_neighbors']
)
min_score = knn_cv_results_df['mean_test_score'].min()
max_score = knn_cv_results_df['mean_test_score'].max()
axs[0].set_title('metric distance = Manhattan')
axs[1].set_title('metric distance = Euclidean')
sns.heatmap(pvt_manhattan, cmap='Blues', ax=axs[0], vmin=min_score, vmax=max_score)
sns.heatmap(pvt_euclidean, cmap='Blues', ax=axs[1], vmin=min_score, vmax=max_score)

# %%
SCORES_DIR = '../data/classification_scores'

compute_clf_scores(
    test_label,
    y_pred=knn_pred_labels_test,
    train_score=knn_best_model_train_score,
    train_score_name='f1_score_train',
    train_time=knn_fit_time,
    score_time=knn_test_score_time,
    params=knn_best_model_params,
    prob_pred=knn_pred_probas_test,
    clf_name='KNN',
    path=SCORES_DIR + '/knn_scores.csv'
)

# %% [markdown]
# ## SVM

# %%
svm_param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'C': [0.001, 0.01, 0.1, 1.0]
}

svm_grid = grid_search('SVM', svm_param_grid, SVC(), train_set, train_label)

# %%
svm_cv_results_df = pd.DataFrame(svm_grid.cv_results_)
svm_best_index = svm_grid.best_index_
svm_best_model_params = svm_cv_results_df.loc[svm_best_index]['params']
svm_best_model_train_score = svm_cv_results_df.loc[svm_best_index]['mean_test_score']

# we create two models for different purposes:
svm_best_model = SVC(**svm_best_model_params) # to measure the training time
svm_best_model_proba = SVC(probability=True, **svm_best_model_params) # to get probabilities for ROC

# %%
fit_start = time()
svm_best_model.fit(train_set, train_label)
svm_fit_time = time() - fit_start
svm_best_model_proba.fit(train_set, train_label)

test_score_start = time()
svm_pred_labels_test = svm_best_model.predict(test_set)
svm_test_score_time = time() - test_score_start
svm_pred_probas_test = svm_best_model_proba.predict_proba(test_set)

pd.DataFrame({'labels': svm_pred_labels_test, 'probs_True': svm_pred_probas_test[:,1]}).to_csv(RESULTS_DIR + '/svm_preds.csv')

# %%
svm_best_model_params

# %%
joblib.dump(svm_best_model, BEST_MODELS_DIR + '/svm.pkl')

# %%
svm_cv_results_df = pd.DataFrame(svm_grid.cv_results_)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
pvt_scale = pd.pivot_table(
    svm_cv_results_df[(svm_cv_results_df['param_gamma'] == 'scale')],
    values='mean_test_score',
    index=['param_C'],
    columns=['param_kernel']
)
pvt_auto = pd.pivot_table(
    svm_cv_results_df[(svm_cv_results_df['param_gamma'] == 'auto')],
    values='mean_test_score',
    index=['param_C'],
    columns=['param_kernel']
)
min_score = svm_cv_results_df['mean_test_score'].min()
max_score = svm_cv_results_df['mean_test_score'].max()
axs[0].set_title('gamma selected with \'scale\'')
axs[1].set_title('gamma selected with \'auto\'')
sns.heatmap(pvt_scale, cmap='Blues', ax=axs[0], vmin=min_score, vmax=max_score)
sns.heatmap(pvt_auto, cmap='Blues', ax=axs[1], vmin=min_score, vmax=max_score)

# %%
compute_clf_scores(
    test_label,
    y_pred=svm_pred_labels_test,
    train_score=svm_best_model_train_score,
    train_score_name='f1_score_train',
    train_time=svm_fit_time,
    score_time=svm_test_score_time,
    params=svm_best_model_params,
    prob_pred=svm_pred_probas_test,
    clf_name='SVM',
    path=SCORES_DIR + '/svm_scores.csv'
)

# %% [markdown]
# ## Nearest Centroid

# %%
nc_param_grid = {
    'metric': ['euclidean', 'manhattan']
}

nc_grid = grid_search('Nearest Centroid', nc_param_grid, NearestCentroid(), train_set, train_label)

# %%
nc_cv_results_df = pd.DataFrame(nc_grid.cv_results_)
nc_best_index = nc_grid.best_index_
nc_best_model_params = nc_cv_results_df.loc[nc_best_index]['params']
nc_best_model_train_score = nc_cv_results_df.loc[nc_best_index]['mean_test_score']
nc_best_model = NearestCentroid(**nc_best_model_params)

# %%
fit_start = time()
nc_best_model.fit(train_set, train_label)
nc_fit_time = time() - fit_start

test_score_start = time()
nc_pred_labels_test = nc_best_model.predict(test_set)
nc_test_score_time = time() - test_score_start

pd.DataFrame({'labels': nc_pred_labels_test}).to_csv(RESULTS_DIR + '/nc_preds.csv')

# %%
nc_best_model_params

# %%
joblib.dump(nc_best_model, BEST_MODELS_DIR + '/nc.pkl')

# %%
nc_cv_results_df = pd.DataFrame(nc_grid.cv_results_)

fig, axs = plt.subplots(1, 1, figsize=(7, 5))
pvt_nc = pd.pivot_table(
    nc_cv_results_df,
    values='mean_test_score',
    index=['param_metric']
)
axs.set_title('Nearest Neighbors')
sns.heatmap(pvt_nc, cmap='Blues', ax=axs)

# %%
SCORES_DIR = '../data/classification_scores'
compute_clf_scores(
    test_label,
    y_pred=nc_pred_labels_test,
    train_score=nc_best_model_train_score,
    train_score_name='f1_score_train',
    train_time=nc_fit_time,
    score_time=nc_test_score_time,
    params=nc_best_model_params,
    clf_name='Nearest Centroid',
    path=SCORES_DIR + '/nc_scores.csv'
)

# %% [markdown]
# ## Analisys and plots

# %%
# load trained models
knn_best_model = joblib.load(BEST_MODELS_DIR + "/knn.pkl")
svm_best_model = joblib.load(BEST_MODELS_DIR + "/svm.pkl")
nc_best_model = joblib.load(BEST_MODELS_DIR + "/nc.pkl")

# %%
y_trues = [test_label, test_label]
y_probs = [knn_pred_probas_test[:,1], knn_pred_probas_test[:,1]]
names = ['KNN']
plot_roc(y_trues, y_probs, names)

y_probs = [svm_pred_probas_test[:,1], svm_pred_probas_test[:,1]]
names = ['SVM']
plot_roc(y_trues, y_probs, names)

# %%
skplt.metrics.plot_roc(test_label.values, knn_pred_probas_test)
skplt.metrics.plot_roc(test_label.values, svm_pred_probas_test)
plt.show()

# %%
# compute confusion matrix

plot_confusion_matrix(y_true=test_label, y_pred=knn_pred_labels_test)
plot_confusion_matrix(y_true=test_label, y_pred=svm_pred_labels_test)
plot_confusion_matrix(y_true=test_label, y_pred=nc_pred_labels_test)

# %%
plot_predictions_in_features_space(
    test_set,
    ['n_males_prop', 'n_child_prop', 'n_participants'],
    test_label,
    knn_pred_labels_test,
    figsize=(15, 15)
)

plot_predictions_in_features_space(
    test_set,
    ['n_males_prop', 'n_child_prop', 'n_participants'],
    test_label,
    svm_pred_labels_test,
    figsize=(15, 15)
)

plot_predictions_in_features_space(
    test_set,
    ['n_males_prop', 'n_child_prop', 'n_participants'],
    test_label,
    nc_pred_labels_test,
    figsize=(15, 15)
)

# %%
fig, axs = plt.subplots(1, figsize=(10, 5))
plot_PCA_decision_boundary(train_set, train_label, knn_best_model, 'KNN', axs=axs)

# %%
fig, axs = plt.subplots(1, figsize=(10, 5))
plot_PCA_decision_boundary(train_set, train_label, svm_best_model, 'SVM', axs=axs)

# %%
fig, axs = plt.subplots(1, figsize=(10, 5))
plot_PCA_decision_boundary(train_set, train_label, nc_best_model, 'Nearest Centroid', axs=axs)

# %%
fig, axs = plt.subplots(1, figsize=(10, 5))
plot_learning_curve(knn_best_model, 'KNN', train_set, train_label, axs)

# %%
fig, axs = plt.subplots(1, figsize=(10, 5))
plot_learning_curve(svm_best_model, 'SVM', train_set, train_label, axs)

# %%
fig, axs = plt.subplots(1, figsize=(10, 5))
plot_learning_curve(nc_best_model, 'Nearest Centroid', train_set, train_label, axs)

# %%
plot_distribution_missclassifications(
    test_label,
    knn_pred_labels_test,
    test_set,
    'poverty_perc',
    'hist'
    )

# %%
plot_distribution_missclassifications(
    test_label,
    svm_pred_labels_test,
    test_set,
    'avg_age',
    'hist'
    )

# %%
plot_distribution_missclassifications(
    test_label,
    nc_pred_labels_test,
    test_set,
    'poverty_perc',
    'hist'
    )

# %%
param_of_interest = 'n_neighbors'
fixed_params = knn_best_model_params.copy() # best params
fixed_params.pop(param_of_interest)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_scores_varying_params(knn_grid.cv_results_, param_of_interest, fixed_params, 'F1 Score', axs, title=f'KNN - F1 score varying {param_of_interest}')

# %%
param_of_interest = 'C'
fixed_params = svm_best_model_params.copy() # best params
fixed_params.pop(param_of_interest)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.set_xscale('log')
plot_scores_varying_params(svm_grid.cv_results_, param_of_interest, fixed_params, 'F1 Score', axs, title=f'SVM - F1 score varying {param_of_interest}')


