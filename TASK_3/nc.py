# %% [markdown]
# # Nearest Centroid

# %% [markdown]
# Nearest Centroid is a classification model which use the mean of the points in the dataset (centroid) to assign labels to observation. Each sample is assigned to the label of the nearest centroid in the feature space.

# %%
import pandas as pd
import json
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, f1_score
from time import time
from classification_utils import *
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RESULTS_DIR = '../data/classification_results'
SEED=42
clf_name = 'NearestCentroidClassifier'

# %% [markdown]
# We load the dataset and the features used for classification.

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
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
# We display the features names we will use.

# %%
print(features_for_clf)
print(f'Number of features: {len(features_for_clf)}')

# %%
# TODO: dire che con feature categoriche non è indicato
# (comunque anche se il centroide assume valori in [0,1] il ragionamento utilizzato per classificare ha senso,
# i.e. se in una classe la metà degli esempi ha una feature a 0 e l'altra metà a 1,
# il centroide sta nel mezzo => la feature non è rilevante)

# %% [markdown]
# We make a 5-fold cross validation in which we check which of the two distance metrics (euclidean and manhattan) has better performance. <br>
# The values for training are scaled to be between 0 and 1 so that bigger values in module, and so bigger distances, don't affect the classification.

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scaler = MinMaxScaler()
nc = NearestCentroid()
pipe = Pipeline(steps=[('scaler', scaler), ('nc', nc)])

param_grid = {
    'nc__metric': ['euclidean', 'manhattan']
}

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=cv,
    refit=False
)
gs.fit(indicators_train_df, true_labels_train)

# %% [markdown]
# We display some metrics derived from cross validation fitting.

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
cv_results_df.columns = [col.replace('nc__', '') for col in cv_results_df.columns]
cv_results_df.head()

# %% [markdown]
# We display the mean and standard deviation of the score of the models fitted in the cross validation. These values are calculated on the test set.

# %%
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %% [markdown]
# We make a random oversampling on the positive class such that fatal incidents become 40% of the total number of records. Doing oversampling we can help classifier make better predictions on test set, but we don't reach a 50-50 ratio since it may be unrealistic.

# %%
# minority oversampling
indicators_oversampled_train_df = pd.read_csv('../data/clf_indicators_train_over.csv', index_col=0)
indicators_oversampled_train_df = indicators_oversampled_train_df[features_for_clf]
true_oversampled_labels_train = pd.read_csv('../data/clf_y_train_over.csv', index_col=0).values.ravel()

# SMOTE oversampling
indicators_smote_train_df = pd.read_csv('../data/clf_indicators_train_smote.csv', index_col=0)
indicators_smote_train_df = indicators_smote_train_df[features_for_clf]
true_labels_smote_train = pd.read_csv('../data/clf_y_train_smote.csv', index_col=0).values.ravel()

# %% [markdown]
# We fit the best model we got from cross validation, then we make and save the predictions on test set. Finally we save the models so that we can reuse it without performing the training.

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model_params = {k.replace('nc__', ''): v for k, v in best_model_params.items()}
best_model = NearestCentroid(**best_model_params)
best_model_oversampled = NearestCentroid(**best_model_params)
best_model_smote = NearestCentroid(**best_model_params)

# scale all the data
minmax_scaler = MinMaxScaler()
indicators_train_scaled = minmax_scaler.fit_transform(indicators_train_df)
indicators_oversampled_train_scaled = minmax_scaler.fit_transform(indicators_oversampled_train_df)
indicators_smote_train_scaled = minmax_scaler.fit_transform(indicators_smote_train_df)
indicators_test_scaled = minmax_scaler.fit_transform(indicators_test_df)


# fit the model on all the training data
fit_start = time()
best_model.fit(indicators_train_scaled, true_labels_train)
fit_time = time()-fit_start

# fit the model on all the training data (oversampled)
fit_start = time()
best_model_oversampled.fit(indicators_oversampled_train_scaled, true_oversampled_labels_train)
fit_time_oversampled = time()-fit_start

# fit the model on all the training data (smote)
fit_start = time()
best_model_smote.fit(indicators_smote_train_scaled, true_labels_smote_train)
fit_time_smote = time()-fit_start


# get the predictions on the training data
train_score_start = time()
pred_labels_train = best_model.predict(indicators_train_scaled)
train_score_time = time()-train_score_start

# get the predictions on the training data (oversampled)
train_score_start = time()
pred_labels_oversampled_train = best_model_oversampled.predict(indicators_oversampled_train_scaled)
train_score_time_oversampled = time()-train_score_start

# get the predictions on the training data (smote)
train_score_start = time()
pred_labels_smote_train = best_model_smote.predict(indicators_smote_train_scaled)
train_score_time_smote = time()-train_score_start


# get the predictions on the test data
test_score_start = time()
pred_labels_test = best_model.predict(indicators_test_scaled)
test_score_time = time()-test_score_start

# get the predictions on the test data (oversampled)
test_score_start = time()
pred_labels_oversampled_test = best_model_oversampled.predict(indicators_test_scaled)
test_score_time_oversampled = time()-test_score_start

# get the predictions on the test data (smote)
test_score_start = time()
pred_labels_smote_test = best_model_smote.predict(indicators_test_scaled)
test_score_time_smote = time()-test_score_start


# save the predictions
pd.DataFrame(
    {'labels': pred_labels_test}
).to_csv(f'{RESULTS_DIR}/{clf_name}_preds.csv')

# save the predictions (oversampled)
pd.DataFrame(
    {'labels': pred_labels_oversampled_test}
).to_csv(f'{RESULTS_DIR}/{clf_name}_oversampled_preds.csv')

# save the predictions (smote)
pd.DataFrame(
    {'labels': pred_labels_smote_test}
).to_csv(f'{RESULTS_DIR}/{clf_name}_smote_preds.csv')


# save the model
file = open(f'{RESULTS_DIR}/{clf_name}.pkl', 'wb')
pickle.dump(obj=best_model, file=file)
file.close()

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}_oversampled.pkl', 'wb')
pickle.dump(obj=best_model_oversampled, file=file)
file.close()

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}_smote.pkl', 'wb')
pickle.dump(obj=best_model_smote, file=file)
file.close()

# save the cv results
best_model_cv_results = pd.DataFrame(cv_results_df.iloc[best_index]).T
best_model_cv_results.index = [clf_name]
best_model_cv_results.to_csv(f'{RESULTS_DIR}/{clf_name}_train_cv_scores.csv')

# %% [markdown]
# We display some classification scores in order to have some metrics useful to make comparisions with other models. These scores are refearing to predictions on training and test set including oversampled and smote.

# %%
compute_clf_scores(
    y_true=true_labels_train,
    y_pred=pred_labels_train,
    train_time=fit_time,
    score_time=train_score_time,
    params=best_model_params,
    prob_pred=None,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_train_scores.csv'
)

# %%
compute_clf_scores(
    y_true=true_oversampled_labels_train,
    y_pred=pred_labels_oversampled_train,
    train_time=fit_time_oversampled,
    score_time=train_score_time_oversampled,
    params=best_model_params,
    prob_pred=None,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_oversampled_train_scores.csv'
)

# %%
compute_clf_scores(
    y_true=true_labels_smote_train,
    y_pred=pred_labels_smote_train,
    train_time=fit_time_smote,
    score_time=train_score_time_smote,
    params=best_model_params,
    prob_pred=None,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_smote_train_scores.csv'
)

# %%
compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    train_time=fit_time,
    score_time=test_score_time,
    params=best_model_params,
    prob_pred=None,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_test_scores.csv'
)

# %%
compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_oversampled_test,
    train_time=fit_time_oversampled,
    score_time=train_score_time_oversampled,
    params=best_model_params,
    prob_pred=None,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_oversampled_test_scores.csv'
)

# %%
compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_smote_test,
    train_time=fit_time_smote,
    score_time=train_score_time_smote,
    params=best_model_params,
    prob_pred=None,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_smote_test_scores.csv'
)

# %% [markdown]
# Since the model with better performance (better f1-score macro) is the one trained on oversampled data, we will use it for further analisys.

# %% [markdown]
# We plot the diffusion matrix in orther to know what is the balancing between false positves, false negatives, true positives and true negatives.

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_oversampled_test,
    title=clf_name
)

# %% [markdown]
# We plot the classification labels in the bidimensional feature spaces obtained pairing 3 features: **aggression**, **drug_alcohol** and **organized**.

# %%
plot_predictions_in_features_space(
    df=incidents_test_df,
    features=['aggression', 'drug_alcohol', 'gun_law_rank', 'n_males'], # TODO: farlo con features significativve
    true_labels=true_labels_test,
    pred_labels=pred_labels_oversampled_test,
    figsize=(30, 30)
)

# %% [markdown]
# Given the plot, although we can't deduce anything on the separation of clusters, we can notice that when **aggression** and **drug_alcohol** are set to 1, the incidents are not fatal, and the model tends to predict them well.

# %% [markdown]
# We permorm PCA and we plot the decision boundary of the first two components.

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_PCA_decision_boundary(
  train_set=indicators_oversampled_train_scaled,
  features=indicators_train_df.columns, # TODO: eventualmente usare solo le numeriche
  train_label=true_oversampled_labels_train,
  classifier=best_model_oversampled,
  classifier_name=clf_name,
  axs=axs
)

# %% [markdown]
# As we expected we got a very simple decision boundary, since the two regions are separated by a straight line.

# %% [markdown]
# We plot the learning curve of the model. Each time we perform training in a subsample of the dataset and see how the results vary.

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_learning_curve(
    classifier=best_model_oversampled,
    classifier_name=clf_name,
    train_set=indicators_oversampled_train_scaled,
    labels=true_oversampled_labels_train,
    ax=axs,
    train_sizes=np.linspace(0.1, 1.0, 5),
    metric='f1'
)

# %% [markdown]
# We notice that the model has a better performance on test than on train. The curve of the test score is pretty stable around 0.6 and has a small standard deviation, while the training curve has some big improvements, both on F1-score and standard deviation, as the training set increase. 

# %% [markdown]
# We display with different types of plot the distribution of the values of the miscassified incidents of **n_killed**, **suicide**, **aggression**, **road** and **location_imp**.

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_oversampled_test,
    incidents_test_df,
    'n_killed',
    'bar',
    title='n_killed distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_oversampled_test,
    incidents_test_df,
    'suicide',
    'pie',
    title='suicide distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_oversampled_test,
    incidents_test_df,
    'aggression',
    'pie',
    title='aggression distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_oversampled_test,
    incidents_test_df,
    'road',
    'pie',
    pie_perc_threshold=2,
    figsize=(20, 5),
    title='road distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_oversampled_test,
    incidents_test_df,
    'location_imp',
    'hist',
    bins=5,
    title='location_imp distribution'
)

# %% [markdown]
# We can notice that for **location_imp** and **n_killed** we have similar distribuition shapes. For **road**, **aggression**  and **suicide** we have that the percentage of misclassified incidents with the tag set to True is less than the one on the entire dataset, in particular there are very few miscassified incidents with **suicide** tag set to true.

# %% [markdown]
# ## Final considerations
# The advantage of Nearest Centroid is that is one of the simplest distance based model for classification, on the other hand it's difficult to have good performance on non-trivial dataset like the one of our case study. So, better performance are possible with more complex models.


