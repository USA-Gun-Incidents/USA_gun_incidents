# %%
import pandas as pd
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from time import time
from classification_utils import *
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RESULTS_DIR = '../data/classification_results'
clf_name = 'KNearestNeighborsClassifier'

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

# %% [markdown]
# We display the features names we will use:

# %%
print(features_for_clf)
print(f'Number of features: {len(features_for_clf)}')

# %%
scaler = MinMaxScaler()
knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('scaler', scaler), ('knn', knn)])

param_grid = [
    {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 21, 31, 51], # odd values to avoid ties
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['brute'], # gli altri cambiano in base al train (rischiamo di usare euristiche su certi fold e brute force su altri)
        'knn__metric': ['minkowski'],
        'knn__p': [1, 2]
    },
    {
        'knn__n_neighbors': [1],
        'knn__weights': ['uniform'], # to reduce the number of combinations
        'knn__algorithm': ['brute'],
        'knn__metric': ['minkowski'],
        'knn__p': [1, 2]
    }
]

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5,
    refit=False
)
gs.fit(indicators_train_df, true_labels_train)

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
cv_results_df.columns = [col.replace('knn__', '') for col in cv_results_df.columns]
cv_results_df.head()

# %%
fig, axs = plt.subplots(1, figsize=(8, 5))
pvt_manhattan = pd.pivot_table(
    cv_results_df,
    values='mean_test_score',
    columns=['param_p', 'param_weights'],
    index=['param_n_neighbors']
)
sns.heatmap(pvt_manhattan, cmap='Blues')

# %%
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model_params = {k.replace('knn__', ''): v for k, v in best_model_params.items()}
best_model = KNeighborsClassifier(**best_model_params)

# scale all the data
minmax_scaler = MinMaxScaler()
indicators_train_scaled = minmax_scaler.fit_transform(indicators_train_df)

# fit the model on all the training data
fit_start = time()
best_model.fit(indicators_train_scaled, true_labels_train)
fit_time = time()-fit_start

# get the predictions on the training data
train_score_start = time()
pred_labels_train = best_model.predict(indicators_train_scaled)
train_score_time = time()-train_score_start
pred_probas_train = best_model.predict_proba(indicators_train_scaled)

# get the predictions on the test data
test_score_start = time()
pred_labels_test = best_model.predict(indicators_test_df.values)
test_score_time = time()-test_score_start
pred_probas_test = best_model.predict_proba(indicators_test_df.values)

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
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_PCA_decision_boundary(
  train_set=indicators_train_scaled,
  features=indicators_train_df.columns, # TODO: eventualmente usare solo le numeriche
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
    train_set=indicators_train_scaled,
    labels=true_labels_train,
    ax=axs,
    train_sizes=np.linspace(0.1, 1.0, 5),
    metric='f1'
)

# %%
param_of_interest = 'n_neighbors'
fixed_params = best_model_params.copy()
fixed_params.pop(param_of_interest)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_scores_varying_params(
    cv_results_df,
    param_of_interest,
    fixed_params,
    'F1',
    axs,
    title=clf_name
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


