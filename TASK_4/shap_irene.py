# -*- coding: utf-8 -*-
# %%
# TODO:
# provare diversi metodi SHAP

# %%
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import shap
from IPython.display import HTML
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from keras.models import load_model
from explanation_utils import *

# %%
def show_plot(plot):
    display(HTML(f"<div style='background-color:white;'>{plot.html()}</div>"))

# %%
RANDOM_STATE = 42

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# neural network model to later load
def nn_model(meta, hidden_layer_sizes, dropouts, activation_functions, last_activation_function):
    n_features_in_ = meta["n_features_in_"]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size, activation_function, dropout in zip(hidden_layer_sizes, activation_functions, dropouts):
        model.add(tf.keras.layers.Dense(hidden_layer_size, activation=activation_function))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation=last_activation_function))
    return model

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()

incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

# load the names of the features
features_db = json.loads(open('../data/clf_indicators_names_distance_based.json').read())
features_rb = json.loads(open('../data/clf_indicators_names_rule_based.json').read())

# project on the used features
indicators_train_db_df = incidents_train_df[features_db]
indicators_train_rb_df = incidents_train_df[features_rb]
indicators_test_db_df = incidents_test_df[features_db]
indicators_test_rb_df = incidents_test_df[features_rb]

# features to analyze
features_to_explore = [
    'date', 'day_of_week', 'days_from_first_incident',
    'state', 'address', 'city',  'min_age', 'max_age',
    'n_child', 'n_teen', 'n_adult', 'n_males', 'n_females',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants', 'notes', 'incident_characteristics1',
    'incident_characteristics2', 'democrat', 'poverty_perc',
    'gun_law_rank', 'aggression', 'accidental', 'defensive',
    'suicide', 'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers',
    'organized', 'social_reasons', 'abduction'
]

# load models and predictions
DT = 'DecisionTreeClassifier'
RF = 'RandomForestClassifier'
XGB = 'XGBClassifier'
NC = 'NearestCentroidClassifier'
KNN = 'KNearestNeighborsClassifier'
SVM = 'SupportVectorMachineClassifier'
NN = 'NeuralNetworkClassifier'
TN = 'TabNetClassifier'
clf_names = [DT, RF, XGB, NC, KNN, SVM, NN, TN]
rb_clf_names = [DT, RF, XGB]

data_dir = '../data/classification_results/'
positions_to_evaluate = []
names_to_evaluate = []
true_labels_to_evaluate = []
preds = {}
models = {}

for clf_name in clf_names:
    preds[clf_name] = {}
    # load predictions
    clf_preds = pd.read_csv(data_dir+clf_name+'_preds.csv')
    preds[clf_name]['labels'] = clf_preds['labels']
    # load probabilities
    if clf_name != NC and clf_name != KNN:
        preds[clf_name]['probs'] = clf_preds['probs']
    # load the model
    if clf_name == NN:
        nn = KerasClassifier(
            nn_model,
            metrics=['accuracy'],
            validation_split=0.2,
            model__hidden_layer_sizes=None,
            model__activation_functions=None,
            model__dropouts=None,
            model__last_activation_function=None
        )
        nn.model = load_model(data_dir+clf_name+'.h5')
        models[clf_name] = nn
    elif clf_name == TN:
        tn = TabNetClassifier()
        tn.load_model(data_dir+TN+'.pkl.zip')
        models[clf_name] = tn
    else:
        with open(data_dir+clf_name+'.pkl', 'rb') as file:
            models[clf_name] = pickle.load(file)

# %% [markdown]
# # Global explanation

# %%
shap_values = {}
shap_interaction_values = {}

# %% [markdown]
# ## Decision Tree

# %%
shap_values[DT] = shap.TreeExplainer(models[DT]).shap_values(indicators_test_rb_df)

# %%
shap.summary_plot(shap_values[DT], indicators_test_rb_df,plot_size=(10,10))
# in classificazione binaria le altezze delle barre sono uguali
# alto abs(shap) => grande importanza

# %%
shap.dependence_plot("avg_age", shap_values[DT][1], indicators_test_rb_df) # Fatal (shap_values_tree[0] is symmetric, only the sign changes)
# democrat è la feature che interagisce di più con avg_age
# età>60 => fatale
# età<20 => non fatale
# sotto i 20 anni lo shap value cambia tra -0.2 e 0.05, dunque dipende da altre features (quella da cui dipende di più è democrat)
# quando è nagativo lo stato è democratico => in stati democratici la mortalità in giovani è minore

# %%
# TODO: farlo per altri attributi

# %%
#shap_interaction_values[DT] = shap.TreeExplainer(models[DT]).shap_interaction_values(indicators_test_rb_df)

# %%
#shap.summary_plot(shap_interaction_values[DT][1], indicators_test_rb_df, max_display=31) # Fatal
# il colore è il valore assunto dalla feature sulle righe (road, school, etc sono per lp più blu sulle righe)
# la posizione lungo y non ha una scala, mostra solo quanti punti ci sono
# interaction value negativo => 

# %%
#shap.summary_plot(shap_interaction_values[DT][1], indicators_test_rb_df, max_display=7)
# età-eta
# tanti punti con interazione negativa; il valore dell'età è basso (blu)
# aggression-age

# %%
# shap.dependence_plot(
#     ("avg_age", "aggression"),
#     shap_interaction_values[DT][1][:500], indicators_test_rb_df[:500]
# )
# quando non c'è aggressione, avg age assume un po' tutti i valori e non c'è interazione
# sotto i 30 anni l'interazione è sia positiva che negativa
# sopra i 40 anni l'interazione è nulla o negativa
# punti in cui l'età è 20 e aggression 1 hanno interazioni positiva e nagativa?
# TODO: cosa significa il valore di interazione? non è una correlazione...

# %%
shap.initjs()
force_plot = shap.force_plot(shap.TreeExplainer(models[DT]).expected_value[0], shap_values[DT][0][:500], indicators_test_rb_df[:500])
show = show_plot(force_plot)
# gli esempi sono ordinati su asse x
# le feature blue determinerebbero non mortale, rosse mortale
# tanti force plot locali stacked verticalmente
# TODO:
# f(x) è?
# si può scegliere lo shapeley value di una feature a scelta

# %% [markdown]
# ## Random Forest

# %%
#shap_values[RF] = shap.TreeExplainer(models[RF]).shap_values(indicators_test_rb_df)

# %% [markdown]
# # Attempted suicide
#
# Attempted suicides involving a single person:

# %%
attempted_suicide_index = incidents_test_df[
    (incidents_test_df['suicide']==1) &
    (true_labels_test_df['death']==0) &
    (incidents_test_df['n_participants']==1)
].index[0]
attempted_suicide_pos = incidents_test_df.index.get_loc(attempted_suicide_index)
positions_to_evaluate.append(attempted_suicide_pos)
names_to_evaluate.append('Attempted Suicide')
true_labels_to_evaluate.append(true_labels_test[attempted_suicide_pos])
attempted_suicide_db_inst = indicators_test_db_df.iloc[attempted_suicide_pos].values
attempted_suicide_rb_inst = indicators_test_rb_df.iloc[attempted_suicide_pos].values
incidents_test_df.iloc[attempted_suicide_pos].to_frame().T

# %% [markdown]
# ## Decision Tree

# %%
pos = attempted_suicide_pos
shap.initjs()
attempted_suicide_expected_value = shap.TreeExplainer(models[DT]).expected_value[0] # prior (base value)
attempted_suicide_shap_values = shap_values[DT][0][pos]
attempted_suicide_sample = indicators_test_rb_df.iloc[pos,:]
shap.force_plot(attempted_suicide_expected_value, attempted_suicide_shap_values, attempted_suicide_sample, matplotlib=matplotlib)
# il valore nero è la probabilità predetta
# base value è the average model output over the training dataset (the value that would be predicted if we did not know any features)
# l'ampiezza delle feature è quanto impattano; le ampiezze più grandi sono collocate al centro

# %% [markdown]
# ## Random Forest

# %% [markdown]
# ## Extreme gradient boosting

# %% [markdown]
# ## K Nearest Neighbors

# %%
if models[KNN].get_params()['n_neighbors'] < 3:
    n_neighbors = 3
else:
    n_neighbors = models[KNN].get_params()['n_neighbors']
distances, indeces = models[KNN].kneighbors(
    X=indicators_test_db_df.iloc[attempted_suicide_pos].values.reshape(-1,len(features_db)),
    n_neighbors=n_neighbors,
    return_distance=True
)

# %%
neighbors_df = incidents_train_df.iloc[indeces[0]].copy()
neighbors_df['distance'] = distances[0]
# put distance column first
neighbors_df = neighbors_df[['distance'] + [col for col in neighbors_df.columns if col != 'distance']]
neighbors_df.style.background_gradient(cmap='Blues', subset='distance')

# %% [markdown]
# ## Support Vector Machine

# %%
hyperplane_dists = models[SVM].decision_function(
    X=pd.concat([
        indicators_test_db_df.iloc[attempted_suicide_pos].to_frame().T.reset_index(),
        indicators_train_db_df.iloc[indeces[0]].reset_index() # neighbors (could be both mortal or not)
    ]).drop(columns=['index'])
)
probas = models[SVM].predict_proba(
    X=pd.concat([
        indicators_test_db_df.iloc[attempted_suicide_pos].to_frame().T.reset_index(),
        indicators_train_db_df.iloc[indeces[0]].reset_index() # neighbors (could be both mortal or not)
    ]).drop(columns=['index'])
)
dist_probs = {
    'distance_from_hyperplane': hyperplane_dists,
    'fatal_probability': probas[:,1]
}
pd.DataFrame(dist_probs)

# %% [markdown]
# ## Feed Forward Neural Networks

# %% [markdown]
# ## TabNet

# %% [markdown]
# # Mass shooting

# %% [markdown]
# Involving many killed people:

# %%
max_killed = incidents_test_df['n_killed'].max()
mass_shooting_index = incidents_test_df[incidents_test_df['n_killed'] == max_killed].index[0]
mass_shooting_pos = incidents_test_df.index.get_loc(mass_shooting_index)
positions_to_evaluate.append(mass_shooting_pos)
names_to_evaluate.append('Mass shooting')
true_labels_to_evaluate.append(true_labels_test[mass_shooting_pos])
mass_shooting_db_inst = indicators_test_db_df.iloc[mass_shooting_pos].values
mass_shooting_rb_inst = indicators_test_rb_df.iloc[mass_shooting_pos].values
incidents_test_df.iloc[mass_shooting_pos].to_frame().T

# %%
# TODO: fare come sopra

# %% [markdown]
# # Incidents predicted as Fatal with highest probability

# %%
indeces_max_prob_death = []
for clf_name in clf_names:
    if clf_name != NC and clf_name != KNN:
        pos = preds[clf_name]['probs'].idxmax()
        indeces_max_prob_death.append(pos)
        positions_to_evaluate.append(pos)
        names_to_evaluate.append(f'Fatal with highest confidence by {clf_name}')
        true_labels_to_evaluate.append(true_labels_test[pos])

max_prob_death_table = {}
for index in indeces_max_prob_death:
    max_prob_death_table[index] = {}
    max_prob_death_table[index]['True_label'] = true_labels_test[index]
    for clf_name in clf_names:
        if clf_name != NC and clf_name != KNN:
            max_prob_death_table[index][clf_name+'_pos_prob'] = preds[clf_name]['probs'][index]
max_prob_death_table = pd.DataFrame(max_prob_death_table).T
max_prob_death_table.style.background_gradient(cmap='Blues', axis=1)

# %%
pd.concat([
    max_prob_death_table.reset_index(),
    incidents_test_df.iloc[indeces_max_prob_death].reset_index()[features_to_explore]],
    axis=1
)

# %% [markdown]
# # Incidents predicted as Non-Fatal with highest probability

# %%
indeces_min_prob_death = []
for clf_name in clf_names:
    if clf_name != NC and clf_name != KNN:
        pos = preds[clf_name]['probs'].idxmin()
        indeces_min_prob_death.append(pos)
        positions_to_evaluate.append(pos)
        names_to_evaluate.append(f'Non-Fatal with highest confidence by {clf_name}')
        true_labels_to_evaluate.append(true_labels_test[pos])

min_prob_death_table = {}
for index in indeces_min_prob_death:
    min_prob_death_table[index] = {}
    min_prob_death_table[index]['True_label'] = true_labels_test[index]
    for clf_name in clf_names:
        if clf_name != NC and clf_name != KNN:
            min_prob_death_table[index][clf_name+'_pos_prob'] = preds[clf_name]['probs'][index]
min_prob_death_table = pd.DataFrame(min_prob_death_table).T
min_prob_death_table.style.background_gradient(cmap='Blues', axis=1)

# %%
pd.concat([
    min_prob_death_table.reset_index(),
    incidents_test_df.iloc[indeces_min_prob_death].reset_index()[features_to_explore]],
    axis=1
)

# %% [markdown]
# ## Incidents with the highest uncertainty in the predicted outcomes

# %%
# indeces_unknown_death = []
# for clf_name in clf_names:
#     if clf_name != NC and clf_name != KNN:
#         indeces_unknown_death.append(np.abs(preds[clf_name]['probs']-0.5).idxmin())

# unknown_death_table = {}
# for index in indeces_unknown_death:
#     unknown_death_table[index] = {}
#     unknown_death_table[index]['True_label'] = true_labels_test_df.iloc[index]['death']
#     for clf_name in clf_names:
#         if clf_name != NC and clf_name != KNN:
#             unknown_death_table[index][clf_name+'_pos_prob'] = preds[clf_name]['probs'][index]
# unknown_death_table = pd.DataFrame(unknown_death_table).T
# unknown_death_table.style.background_gradient(cmap='Blues', axis=1)

# %%
# pd.concat([
#     unknown_death_table.reset_index(),
#     incidents_test_df.iloc[indeces_unknown_death].reset_index()[features_to_explore]],
#     axis=1
# )

# %% [markdown]
# ## Evaluation

# %%
# load the already computed default values for features
non_fatal_db_default = pd.read_csv(data_dir+'non_fatal_db_default_features.csv').to_numpy()[0]
fatal_db_default = pd.read_csv(data_dir+'fatal_db_default_features.csv').to_numpy()[0]
non_fatal_rb_default = pd.read_csv(data_dir+'non_fatal_rb_default_features.csv').to_numpy()[0]
fatal_rb_default = pd.read_csv(data_dir+'fatal_rb_default_features.csv').to_numpy()[0]

# %%
# TODO: assicurarsi di accedere a quelli giusti!
clf_names = [DT] # TODO: da togliere quando li abbiamo tutti

clfs_metrics = []
for clf_name in clf_names:
    model = models[clf_name]
    if clf_name in rb_clf_names:
        samples = indicators_test_rb_df.iloc[positions_to_evaluate].values
    else:
        samples = indicators_test_db_df.iloc[positions_to_evaluate].values

    if clf_name != XGB:
        models[clf_name].feature_names_in_ = None # to silence warnings

    clf_metrics = {}
    for i in range(samples.shape[0]):
        prediction = model.predict(samples[i].reshape(1,-1))[0]
        feature_importances = shap_values[clf_name][prediction][positions_to_evaluate[i]]
        if clf_name in rb_clf_names:
            if prediction == 0:
                feature_defaults = non_fatal_rb_default
            else:
                feature_defaults = fatal_rb_default
        else:
            if prediction == 0:
                feature_defaults = non_fatal_db_default
            else:
                feature_defaults = fatal_db_default
        sample_metric = evaluate_explanation(model, samples[i], feature_importances, feature_defaults)
        clf_metrics[names_to_evaluate[i]] = sample_metric

    clf_metrics_df = pd.DataFrame(clf_metrics).T
    clf_metrics_df.columns = pd.MultiIndex.from_product([clf_metrics_df.columns, [clf_name]])
    clfs_metrics.append(clf_metrics_df)

clfs_metrics = clfs_metrics[0].join(clfs_metrics[1:]).sort_index(level=0, axis=1)
clfs_metrics['True Label'] = true_labels_to_evaluate

# save faithfulness
faithfulness = clfs_metrics[['faithfulness']]
faithfulness.columns = faithfulness.columns.droplevel()
faithfulness.to_csv('../data/eplanation_results/lime_faithfulness.csv')

# save monotonity
monotonity = clfs_metrics[['monotonity']]
monotonity.columns = monotonity.columns.droplevel()
monotonity.to_csv('../data/eplanation_results/lime_monotonity.csv')

clfs_metrics

# %%
positions_to_evaluate = np.arange(0,51) # TODO: decidere quanti prenderne e se prenderli a caso o con un criterio
clf_names = [DT] # TODO: da togliere quando li abbiamo tutti

clfs_metrics = {}
for clf_name in clf_names:
    model = models[clf_name]
    if clf_name in rb_clf_names:
        samples = indicators_test_rb_df.iloc[positions_to_evaluate].values
    else:
        samples = indicators_test_db_df.iloc[positions_to_evaluate].values

    clfs_metrics[clf_name] = {}
    faithfulness = []
    monotonity = []
    for i in range(samples.shape[0]):
        prediction = model.predict(samples[i].reshape(1,-1))[0]
        feature_importances = shap_values[clf_name][prediction][positions_to_evaluate[i]]
        if clf_name in rb_clf_names:
            if prediction == 0:
                feature_defaults = non_fatal_rb_default
            else:
                feature_defaults = fatal_rb_default
        else:
            if prediction == 0:
                feature_defaults = non_fatal_db_default
            else:
                feature_defaults = fatal_db_default
        sample_metric = evaluate_explanation(model, samples[i], feature_importances, feature_defaults)
        faithfulness.append(sample_metric['faithfulness'])
        #monotonity.append(sample_metric['monotonity'])
    clfs_metrics[clf_name]['mean faithfulness'] = np.nanmean(faithfulness)
    clfs_metrics[clf_name]['std faithfulness'] = np.nanstd(faithfulness)

pd.DataFrame(clfs_metrics)


