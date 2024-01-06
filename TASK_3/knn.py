# %% [markdown]
# # K-Nearest Neighbors
# 
# KNN is a distance based algorithm useful for classification tasks. IT work by calculating the centroids of the samples belonging to the same class; the label is predicted by looking at wich is the most common class of the K points nearest to the one we want to classify.

# %%
import pandas as pd
import json
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, f1_score
from time import time
from classification_utils import *
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RESULTS_DIR = '../data/classification_results'
SEED=42
clf_name = 'KNearestNeighborsClassifier'

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

# %% [markdown]
# We make a random oversampling on the positive class such that fatal incidents become 40% of the total number of records. Doing oversampling we can help classifier make better predictions on test set, but we don't reach a 50-50 ratio since it may be unrealistic

# %%
# minority oversampling
oversampler = RandomOverSampler(sampling_strategy=0.67, random_state=SEED)
indicators_oversampled_train_df, true_oversampled_labels_train = oversampler.fit_resample(indicators_train_df, true_labels_train)

# %% [markdown]
# We make a 5-fold cross validation in which we check which is the best value for each hyperparameter (*n_neighbors*, *weights*, *algorithm*, *metric*, *p*). The parameters we select in this section are chosen on the basis of previous test, looking at the ones which performed better. <br>
# The values for training are scaled to be between 0 and 1 so that bigger values in module, and so bigger distances, don't affect the classification.

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scaler = MinMaxScaler()
knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('scaler', scaler), ('knn', knn)])

param_grid = [
    {
        'knn__n_neighbors': [251, 265, 287, 303, 321, 335], # odd values to avoid ties
        'knn__weights': ['uniform', 'distance'],
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
    cv=cv,
    refit=False
)
gs.fit(indicators_oversampled_train_df, true_oversampled_labels_train)

# %% [markdown]
# We display some metrics derived from cross validation fitting.

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
cv_results_df.columns = [col.replace('knn__', '') for col in cv_results_df.columns]
cv_results_df.head()

# %% [markdown]
# We plot the heatmap of the scores, on the test set, archieved by each classifier with different paramenters. It gives an idea of the importance of each parameter and the variation of the performance in changing their values.

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
fig, axs = plt.subplots(1, figsize=(8, 5))
pvt_manhattan = pd.pivot_table(
    cv_results_df,
    values='mean_test_score',
    columns=['param_p'],
    index=['param_n_neighbors']
)
sns.heatmap(pvt_manhattan, cmap='Blues')

# %% [markdown]
# From the heatmaps we can see that is foundamental to use Manhattan distance (p=1) and weighted distances in order to archieve better performance. As we expected, there is just a slight variation in the score changing the number of neighbors; in fact those are some of the best parameters chosen in a wider range. <br>
# In these settings, it turns out that the most influent hyperparameters are *weights*, *metric* and *p*.

# %% [markdown]
# We display the mean and standard deviation of the score of the models fitted in the cross validation. These values are calculated on the test set.

# %%
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %% [markdown]
# We fit the best model we got from cross validation, then we make and save the predictions on test set. Finally we save the models so that we can reuse it without performing the training.

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model_params = {k.replace('knn__', ''): v for k, v in best_model_params.items()}
best_model = KNeighborsClassifier(**best_model_params)

# scale all the data
minmax_scaler = MinMaxScaler()
indicators_train_scaled = minmax_scaler.fit_transform(indicators_train_df)
indicators_oversampled_train_scaled = minmax_scaler.fit_transform(indicators_oversampled_train_df)

# fit the model on all the training data
fit_start = time()
best_model.fit(indicators_oversampled_train_df, true_oversampled_labels_train)
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

# %% [markdown]
# We display some classification scores in order to have some metrics useful to make comparisions with other models. These scores are refearing to predictions on training and test set

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

# %% [markdown]
# We plot the diffusion matrix in orther to know what is the balancing between false positves, false negatives, true positives and true negatives.

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title=clf_name
)

# %% [markdown]
# We can see that more than half of the fatal incidents are not detected, and the ratio between TP and FP is about 0.5.

# %% [markdown]
# We plot the classification labels in the bidimensional feature spaces obtained pairing 3 features: **aggression**, **drug_alcohol** and **organized**.

# %%
plot_predictions_in_features_space(
    df=incidents_test_df,
    features=['aggression', 'drug_alcohol', 'gun_law_rank', 'n_males'], # TODO: farlo con features significativve
    true_labels=true_labels_test,
    pred_labels=pred_labels_test,
    figsize=(30, 30)
)

# %% [markdown]
# As we can see there are a lot of miscalssified points so we can't assert anything about a possible underlying structure of the classes.

# %% [markdown]
# We plot the ROC curve of the model basing on its prediction on test set.

# %%
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

# %% [markdown]
# We permorm PCA and we plot the decision boundary of the first two components.

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

# %% [markdown]
# In the feature space of the first two components we have that the boundary confine the space of fatal incidents in a small zone.

# %% [markdown]
# We plot the learning curve of the model. Each time we perform training in a subsample of the dataset and see how the results vary.

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

# %% [markdown]
# While training error is always 1, we have F1-score which has small improvements with the increasing of training samples. Standard deviation of the score is generally small, but it slightly increase with very few or very much samples.

# %% [markdown]
# We plot the curve of the F1 score for different values of k.

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

# %% [markdown]
# As already said, with the parameters used in cross validation, the means of the scores are pretty muche the same, as well as the standard deviations.

# %% [markdown]
# We display with different types of plot the distribution of the values of the miscassified incidents of **n_killed**, **suicide**, **aggression**, **road** and **location_imp**.

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
    'aggression',
    'pie',
    title='aggression distribution'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
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
    pred_labels_test,
    incidents_test_df,
    'location_imp',
    'hist',
    bins=5,
    title='location_imp distribution'
)

# %% [markdown]
# We can notice that for **location_imp** we have similar distribuition shapes. For **road** and **aggression** we have that the percentage of misclassified incidents with tag set to True is slightly than the one on the entire dataset; for **suicide** instead, we have the opposite situation. For **n_killed** we have a something different, where the two distributions are quite similar, except that there are very few misclassified incidents with n_killed equals to 0 in proportion to the how many are they in the whole dataset.

# %% [markdown]
# ## Final considerations
# 
# KNN is a good algorithm for classification since it's easy and quite fast to execute. In our case, we found the best hyperparameter values and reached pretty decent results. Despite this, we think that other more colpex classifiers will have better performance on our dataset.

# %% [markdown]
# Qua sotto abbiamo qualche grid search in cui controlliamo valori meno performanti. andrebbero tolte però per ora le lascio per farvele vedere.

# %%
param_grid_f = [
    #{
    #    'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 21, 31, 51], # odd values to avoid ties
    #    'knn__weights': ['uniform', 'distance'],
    #    'knn__algorithm': ['brute'], # gli altri cambiano in base al train (rischiamo di usare euristiche su certi fold e brute force su altri)
    #    'knn__metric': ['minkowski'],
    #    'knn__p': [1, 2]
    #},
    #{
        #'knn__n_neighbors': [13, 21, 35, 51], # odd values to avoid ties
        #'knn__n_neighbors': [67, 95, 121, 151], # odd values to avoid ties
        #'knn__n_neighbors': [201, 251, 315, 401], # odd values to avoid ties
        #'knn__n_neighbors': [271, 297, 315, 335], # odd values to avoid ties
        #'knn__n_neighbors': [201, 251, 301, 335], # odd values to avoid ties
        #'knn__weights': ['distance'],
        #'knn__algorithm': ['brute'], # gli altri cambiano in base al train (rischiamo di usare euristiche su certi fold e brute force su altri)
        #'knn__metric': ['minkowski'],
        #'knn__p': [1, 2]
    #}
    #,{
    #    'knn__n_neighbors': [1],
    #    'knn__weights': ['uniform', 'distance'], # to reduce the number of combinations
    #    'knn__algorithm': ['brute'],
    #    'knn__metric': ['minkowski'],
    #    'knn__p': [1, 2]
    #}ù
]

gs_f = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_f,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=cv,
    refit=False
)
gs_f.fit(indicators_oversampled_train_df, true_oversampled_labels_train)

# %%
cv_results_df_f = pd.DataFrame(gs_f.cv_results_)
cv_results_df_f.columns = [col.replace('knn__', '') for col in cv_results_df_f.columns]
cv_results_df_f.head()

# %%
params = [col for col in cv_results_df_f.columns if 'param_' in col and 'random' not in col]
cv_results_df_f.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
params = [col for col in cv_results_df_f.columns if 'param_' in col and 'random' not in col]
cv_results_df_f.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
params = [col for col in cv_results_df_f.columns if 'param_' in col and 'random' not in col]
cv_results_df_f.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
params = [col for col in cv_results_df_f.columns if 'param_' in col and 'random' not in col]
cv_results_df_f.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
params = [col for col in cv_results_df_f.columns if 'param_' in col and 'random' not in col]
cv_results_df_f.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')


