# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
# 
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
# 
# # Neural Network Classifier
# 
# We import the libraries and define constants and settings of the notebook:

# %%
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from time import time
from classification_utils import *

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RESULTS_DIR = '../data/classification_results'
RANDOM_STATE = 42
clf_name = 'NeuralNetworkClassifier'

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

# %% [markdown]
# We display the features names we will use:

# %%
print(features_for_clf)
print(f'Number of features: {len(features_for_clf)}')

# %% [markdown]
# We define a list of the categorical features:

# %%
categorical_features = [
    'day_x', 'day_y', 'day_of_week_x', 'day_of_week_y', 'month_x', 'month_y', 'year',
    'democrat', 'gun_law_rank',
    'aggression', 'accidental', 'defensive', 'suicide',
    'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]

# %%
# Di seguito uso un wrapper di Keras per usare scikit learn
# fino a poco tempo fa era parte di tensorflow, poi hanno smesso di mantenerlo e infine l'hanno rimosso dalle nuove versioni :(
# https://adriangb.com/scikeras/stable/notebooks/Basic_Usage.html#7.2-Performing-a-grid-search

def get_clf(meta, hidden_layer_sizes, dropouts, activation_functions, last_activation_function):
    n_features_in_ = meta["n_features_in_"]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size, activation_function, dropout in zip(hidden_layer_sizes, activation_functions, dropouts):
        model.add(tf.keras.layers.Dense(hidden_layer_size, activation=activation_function))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation=last_activation_function))
    return model

best_model = KerasClassifier(
    get_clf,
    metrics=['accuracy'],
    validation_split=0.2, # FIXME: sul validation usiamo i dati scalati (ma forse è okay, lo usiamo solo per vedere la curva o fare eventualmente early stopping)
    model__hidden_layer_sizes=None,
    model__activation_functions=None,
    model__dropouts=None,
    model__last_activation_function=None
)

scaler = MinMaxScaler()
pipe = Pipeline(steps=[("scaler", scaler), ("nn", best_model)])

# TODO: questi valori li ho messi a caso per testarlo
# TODO: usare loss pesata per bilanciare le classi
param_grid = [
    {
        'nn__model__hidden_layer_sizes': [(256, 256,)],
        'nn__model__activation_functions': [('sigmoid', 'sigmoid',)],
        'nn__model__last_activation_function': ['sigmoid'],
        'nn__model__dropouts': [(0.1, 0.1,)],
        'nn__optimizer': ['adamax'],
        'nn__optimizer__learning_rate': [0.001], # default of adamax
        #'nn__optimizer__decay': [1e-5],
        'nn__loss': ['mean_squared_error'],
        'nn__batch_size': [256],
        'nn__epochs': [10]
    },
] # lista di dizionari (e.g. con 3 layer) per evitare combinazioni non valide

gs = GridSearchCV( # RandomizedSearchCV?
    estimator=pipe,
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=StratifiedShuffleSplit(n_splits=2, test_size=1/3, random_state=RANDOM_STATE), # TODO: con altri classificatori usiamo stratified 5-fold, qui ci vuole troppo
    refit=False
)
gs.fit(indicators_train_df, true_labels_train)

# %%
val_results_df = pd.DataFrame(gs.cv_results_)
val_results_df.head()

# %%
# TODO: heatmap per studiare influenza iperparametri (vedi altri notebook)

# %%
params = [col for col in val_results_df.columns if 'param_' in col and 'random' not in col]
val_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
best_index = gs.best_index_
best_model_params = val_results_df.iloc[best_index]['params']
# remove from the params the prefix 'nn__'
best_model_params = {k.replace('nn__', ''): v for k, v in best_model_params.items()}
best_model = KerasClassifier(
    get_clf,
    metrics=['accuracy'],
    validation_split=0.2,
    **best_model_params
)
print(best_model_params)

# %%
# scale all the data
minmax_scaler = MinMaxScaler()
indicators_train_scaled = minmax_scaler.fit_transform(indicators_train_df)

# fit the model on all the training data
fit_start = time()
best_model.fit(indicators_train_scaled, true_labels_train)
fit_time = time()-fit_start

# get the predictions on the training data
train_score_start = time()
pred_labels_train = best_model.predict(indicators_train_scaled) # TODO: assicurarsi che il 'flattening' automatico sia come vogliamo (tanh è centrata in 0, sigmoid in 0.5)
train_score_time = time()-train_score_start
pred_probas_train = best_model.predict_proba(indicators_train_scaled)

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
best_model.model_.save(f'{RESULTS_DIR}/{clf_name}.h5')

# save the val results
best_model_val_results = pd.DataFrame(val_results_df.iloc[best_index]).T
best_model_val_results.index = [clf_name]
best_model_val_results.to_csv(f'{RESULTS_DIR}/{clf_name}_train_cv_scores.csv')

# %%
best_model.model_.summary()

# %%
def plot_learning_curve(history, metric_name):
    metric = history[metric_name]
    val_metric = history['val_'+metric_name]
    epochs = range(1, len(metric) + 1)
    plt.plot(epochs, metric, label='Training')
    plt.plot(epochs, val_metric, label='Validation')
    plt.title('Training and validation '+metric_name)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

# %%
plot_learning_curve(best_model.history_, 'accuracy')

# %%
plot_learning_curve(best_model.history_, 'loss')

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
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    train_time=fit_time,
    score_time=test_score_time,
    params=best_model_params,
    prob_pred=pred_probas_test,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_test_scores.csv'
)
test_scores

# %%
indicators_over_train_df = pd.read_csv('../data/clf_indicators_train_over.csv', index_col=0)
indicators_over_train_df = indicators_over_train_df[features_for_clf]
indicators_train_over_scaled = minmax_scaler.fit_transform(indicators_over_train_df)
true_labels_over_train = pd.read_csv('../data/clf_y_train_over.csv', index_col=0).values.ravel()

# %%
# fit the model on all the training data
best_model_over = KerasClassifier(
    get_clf,
    metrics=['accuracy'],
    validation_split=0.2,
    **best_model_params
)

fit_start = time()
best_model_over.fit(indicators_train_over_scaled, true_labels_over_train)
fit_over_time = time()-fit_start

# get the predictions on the training data
train_score_start = time()
pred_labels_over_train = best_model_over.predict(indicators_train_over_scaled)
train_score_over_time = time()-train_score_start
pred_probas_over_train = best_model_over.predict_proba(indicators_train_over_scaled)

# get the predictions on the test data
test_score_start = time()
pred_labels_over_test = best_model_over.predict(indicators_test_df.values)
test_score_over_time = time()-test_score_start
pred_probas_over_test = best_model_over.predict_proba(indicators_test_df.values)

# save the predictions
pd.DataFrame(
    {'labels': pred_labels_over_test, 'probs': pred_probas_over_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_oversample_preds.csv')

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}_oversample.pkl', 'wb')
pickle.dump(obj=best_model_over, file=file)
file.close()

# %%
indicators_smote_train_df = pd.read_csv('../data/clf_indicators_train_smote.csv', index_col=0)
indicators_smote_train_df = indicators_smote_train_df[features_for_clf]
indicators_smote_train_scaled = minmax_scaler.fit_transform(indicators_smote_train_df)
true_labels_smote_train = pd.read_csv('../data/clf_y_train_smote.csv', index_col=0).values.ravel()

# %%
# fit the model on all the training data
best_model_smote = KerasClassifier(
    get_clf,
    metrics=['accuracy'],
    validation_split=0.2,
    **best_model_params
)

fit_start = time()
best_model_smote.fit(indicators_smote_train_scaled, true_labels_smote_train)
fit_smote_time = time()-fit_start

# get the predictions on the training data
train_score_start = time()
pred_labels_smote_train = best_model_smote.predict(indicators_smote_train_scaled)
train_score_smote_time = time()-train_score_start
pred_probas_smote_train = best_model_smote.predict_proba(indicators_smote_train_scaled)

# get the predictions on the test data
test_score_start = time()
pred_labels_smote_test = best_model_smote.predict(indicators_test_df.values)
test_score_smote_time = time()-test_score_start
pred_probas_smote_test = best_model_smote.predict_proba(indicators_test_df.values)

# save the predictions
pd.DataFrame(
    {'labels': pred_labels_smote_test, 'probs': pred_probas_smote_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_smote_preds.csv')

# save the model
file = open(f'{RESULTS_DIR}/{clf_name}_smote.pkl', 'wb')
pickle.dump(obj=best_model_smote, file=file)
file.close()

# %%
test_over_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_over_test,
    train_time=fit_over_time,
    score_time=test_score_over_time,
    params=best_model_params,
    prob_pred=pred_probas_over_test,
    clf_name=clf_name+' over',
    path=f'{RESULTS_DIR}/{clf_name}_over_test_scores.csv'
)

test_smote_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_smote_test,
    train_time=fit_smote_time,
    score_time=test_score_smote_time,
    params=best_model_params,
    prob_pred=pred_probas_smote_test,
    clf_name=clf_name+' SMOTE',
    path=f'{RESULTS_DIR}/{clf_name}_smote_test_scores.csv'
)

pd.concat([test_scores, test_over_scores, test_smote_scores])

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title=clf_name
)

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_smote_test,
    title=clf_name + ' SMOTE'
)

# %%
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

# %%
plot_predictions_in_features_space(
    df=incidents_test_df,
    features=['n_males_prop', 'n_child_prop', 'n_teen_prop', 'n_participants', 'poverty_perc'],
    true_labels=true_labels_test,
    pred_labels=pred_labels_test,
    figsize=(15, 50)
)

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_PCA_decision_boundary(
  train_set=indicators_train_df,
  features=[col for col in indicators_train_df.columns if col not in categorical_features],
  train_label=true_labels_train,
  classifier=best_model,
  classifier_name=clf_name,
  axs=axs,
  scale=True,
  pca=True
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


