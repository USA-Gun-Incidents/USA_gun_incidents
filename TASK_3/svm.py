# %%
import pandas as pd
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,  StratifiedShuffleSplit
from sklearn.metrics import make_scorer, f1_score
from sklearn.inspection import permutation_importance
from time import time
from classification_utils import *
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RESULTS_DIR = '../data/classification_results'
clf_name = 'SupportVectorMachineClassifier'

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
svc = SVC()
pipe = Pipeline(steps=[('scaler', scaler), ('svc', svc)])

param_grid = [
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.001, 0.01, 0.1, 1]
    },
    {
        'svc__kernel': ['poly', 'rbf', 'sigmoid'],
        'svc__C': [0.001, 0.01, 0.1, 1],
        'svc__gamma': ['scale', 'auto'],
    } # poly degree default = 3; class_weight='balanced'? default = None
]

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=StratifiedShuffleSplit(n_splits=2, test_size=1/3), # TODO: con altri classificatori usiamo stratified 5-fold, qui ci vuole troppo
    refit=False
)
gs.fit(indicators_train_df, true_labels_train)

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
cv_results_df.columns = [col.replace('svc__', '') for col in cv_results_df.columns]
cv_results_df.head()

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
pvt_scale = pd.pivot_table(
    cv_results_df[(cv_results_df['param_gamma'] == 'scale')],
    values='mean_test_score',
    index=['param_C'],
    columns=['param_kernel']
)
pvt_auto = pd.pivot_table(
    cv_results_df[(cv_results_df['param_gamma'] == 'auto')],
    values='mean_test_score',
    index=['param_C'],
    columns=['param_kernel']
)
min_score = cv_results_df['mean_test_score'].min()
max_score = cv_results_df['mean_test_score'].max()
axs[0].set_title("param_gamma='scale'")
axs[1].set_title("param_gamma='auto'")
sns.heatmap(pvt_scale, cmap='Blues', ax=axs[0], vmin=min_score, vmax=max_score)
sns.heatmap(pvt_auto, cmap='Blues', ax=axs[1], vmin=min_score, vmax=max_score)

# %%
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model_params = {k.replace('svc__', ''): v for k, v in best_model_params.items()}
best_model_params['probability'] = True
best_model = SVC(**best_model_params)

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
param_of_interest = 'C'
fixed_params = best_model_params.copy()
fixed_params.pop(param_of_interest)
fixed_params.pop('probability')
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
# TODO: eventualmente spostare ciò che segue in explainability

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
svc =  SVC(kernel='linear')
svc.fit(indicators_train_scaled, true_labels_train)
axs.set_yscale('log')
display_feature_importances(
    feature_names=indicators_train_df.columns,
    feature_importances=np.abs(svc.coef_[0]),
    axs=axs,
    title='SVM - linear kernel',
    path=f'{RESULTS_DIR}/svm_linear_feature_importances.csv'
)

# %%
# train SVC with rbf kernel
svc =  SVC(kernel='rbf', gamma='scale', C=0.1)
svc.fit(indicators_train_scaled, true_labels_train)
# get features importances
perm_importance = permutation_importance(svc, indicators_train_scaled, true_labels_train, random_state = 42)
# plot features importances
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
display_feature_importances(
    feature_names=indicators_train_df.columns,
    feature_importances=perm_importance['importances_mean'],
    axs=axs,
    title='SVM - rbf kernel',
    path=f'{RESULTS_DIR}/svm_rbf_feature_importances.csv'
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


