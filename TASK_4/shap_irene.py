# %%
# TODO:
# provare diversi metodi SHAP

# %%
import pandas as pd
import matplotlib
import numpy as np
import json
import shap
from IPython.display import HTML
from explanation_utils import *

# %%
RANDOM_STATE = 42

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

def show_plot(plot):
    display(HTML(f"<div style='background-color:white;'>{plot.html()}</div>"))

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

clf_names = [clf.value for clf in Classifiers]
rb_clf_names = [Classifiers.DT.value, Classifiers.RF.value, Classifiers.XGB.value, Classifiers.RIPPER.value]

preds = get_classifiers_predictions('../data/classification_results/')
classifiers = get_classifiers_objects('../data/classification_results/')

# %% [markdown]
# # Global explanation

# %%
shap_values = {}
shap_interaction_values = {}

# %% [markdown]
# ## Decision Tree

# %%
shap_values[Classifiers.DT.value] = shap.TreeExplainer(classifiers[Classifiers.DT.value]).shap_values(indicators_test_rb_df)

# %%
shap.summary_plot(shap_values[Classifiers.DT.value], indicators_test_rb_df,plot_size=(10,10))
# in classificazione binaria le altezze delle barre sono uguali
# alto abs(shap) => grande importanza

# %%
shap.dependence_plot("avg_age", shap_values[Classifiers.DT.value][1], indicators_test_rb_df) # Fatal (shap_values_tree[0] is symmetric, only the sign changes)
# democrat è la feature che interagisce di più con avg_age
# età>60 => fatale
# età<20 => non fatale
# sotto i 20 anni lo shap value cambia tra -0.2 e 0.05, dunque dipende da altre features (quella da cui dipende di più è democrat)
# quando è nagativo lo stato è democratico => in stati democratici la mortalità in giovani è minore

# %%
# TODO: farlo per altri attributi

# %%
shap_interaction_values[Classifiers.DT.value] = shap.TreeExplainer(classifiers[Classifiers.DT.value]).shap_interaction_values(indicators_test_rb_df)

# %%
shap.summary_plot(shap_interaction_values[Classifiers.DT.value][1], indicators_test_rb_df, max_display=31) # Fatal
# il colore è il valore assunto dalla feature sulle righe (road, school, etc sono per lp più blu sulle righe)
# la posizione lungo y non ha una scala, mostra solo quanti punti ci sono
# interaction value negativo => 

# %%
shap.summary_plot(shap_interaction_values[Classifiers.DT.value][1], indicators_test_rb_df, max_display=7)
# età-eta
# tanti punti con interazione negativa; il valore dell'età è basso (blu)
# aggression-age

# %%
shap.dependence_plot(
    ("avg_age", "aggression"),
    shap_interaction_values[Classifiers.DT.value][1][:500], indicators_test_rb_df[:500]
)
# quando non c'è aggressione, avg age assume un po' tutti i valori e non c'è interazione
# sotto i 30 anni l'interazione è sia positiva che negativa
# sopra i 40 anni l'interazione è nulla o negativa
# punti in cui l'età è 20 e aggression 1 hanno interazioni positiva e nagativa?
# TODO: cosa significa il valore di interazione? non è una correlazione...

# %%
shap.initjs()
force_plot = shap.force_plot(shap.TreeExplainer(classifiers[Classifiers.DT.value]).expected_value[0], shap_values[Classifiers.DT.value][0][:500], indicators_test_rb_df[:500])
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
#shap_values[Classifier.RF.value] = shap.TreeExplainer(classifiers[Classifier.DT.value]).shap_values(indicators_test_rb_df)

# %% [markdown]
# # Attempted suicide

# %%
selected_records_to_explain_df = pd.read_csv('../data/explanation_results/selected_records_to_explain.csv', index_col=0)
attempted_suicide_pos = selected_records_to_explain_df[selected_records_to_explain_df['instance names']=='Attempted Suicide']['positions'][0]

# %% [markdown]
# ## Decision Tree

# %%
pos = attempted_suicide_pos
shap.initjs()
attempted_suicide_expected_value = shap.TreeExplainer(classifiers[Classifiers.DT.value]).expected_value[0] # prior (base value)
attempted_suicide_shap_values = shap_values[Classifiers.DT.value][0][pos]
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
if classifiers[Classifiers.KNN.value].get_params()['n_neighbors'] < 3:
    n_neighbors = 3
else:
    n_neighbors = classifiers[Classifiers.KNN.value].get_params()['n_neighbors']
distances, indeces = classifiers[Classifiers.KNN.value].kneighbors(
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
hyperplane_dists = classifiers[Classifiers.SVM.value].decision_function(
    X=pd.concat([
        indicators_test_db_df.iloc[attempted_suicide_pos].to_frame().T.reset_index(),
        indicators_train_db_df.iloc[indeces[0]].reset_index() # neighbors (could be both mortal or not)
    ]).drop(columns=['index'])
)
probas = classifiers[Classifiers.SVM.value].predict_proba(
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

# %%
# TODO: fare lo stesso per mass shooting

# %% [markdown]
# ## Evaluation

# %%
# load the already computed default values for features
non_fatal_db_default = pd.read_csv('../data/classification_results/non_fatal_db_default_features.csv').to_numpy()[0]
fatal_db_default = pd.read_csv('../data/classification_results/fatal_db_default_features.csv').to_numpy()[0]
features_db_defaults = [non_fatal_db_default, fatal_db_default]
non_fatal_rb_default = pd.read_csv('../data/classification_results/non_fatal_rb_default_features.csv').to_numpy()[0]
fatal_rb_default = pd.read_csv('../data/classification_results/fatal_rb_default_features.csv').to_numpy()[0]
features_rb_default = [non_fatal_rb_default, fatal_rb_default]

# %%
# TODO: assicurarsi di accedere a quelli giusti!
clf_names = [Classifiers.DT.value] # TODO: da togliere quando li abbiamo tutti

# %%
positions_to_explain = selected_records_to_explain_df['positions'].to_list()
instance_names_to_explain = selected_records_to_explain_df['instance names'].to_list()
true_labels_to_explain = selected_records_to_explain_df['true labels'].to_list()

metrics_random_records = []
for clf_name in clf_names:
    classifier = classifiers[clf_name]
    if clf_name in rb_clf_names:
        feature_defaults = features_rb_default
        instances = indicators_test_rb_df.iloc[positions_to_explain].values
    else:
        feature_defaults = features_db_default
        instances = indicators_test_db_df.iloc[positions_to_explain].values

    clf_metrics = {}
    for i in range(instances.shape[0]):
        prediction = classifier.predict(instances[i].reshape(1,-1))[0]
        feature_importances = shap_values[clf_name][prediction][positions_to_explain[i]] # TODO: diversi shap values
        sample_metric = evaluate_explanation(classifier, instances[i], feature_importances, feature_defaults[true_labels_to_explain[i]])
        clf_metrics[instance_names_to_explain[i]] = sample_metric

    clf_metrics_df = pd.DataFrame(clf_metrics).T
    clf_metrics_df.columns = pd.MultiIndex.from_product([clf_metrics_df.columns, [clf_name]])
    metrics_random_records.append(clf_metrics_df)

metrics_selected_records_df = metrics_random_records[0].join(metrics_random_records[1:]).sort_index(level=0, axis=1)
metrics_selected_records_df['True Label'] = true_labels_to_explain

# save faithfulness
faithfulness_df = metrics_selected_records_df[['faithfulness']]
faithfulness_df.columns = faithfulness_df.columns.droplevel()
faithfulness_df.to_csv('../data/explanation_results/shap_faithfulness.csv')

# save monotonity
monotonity_df = metrics_selected_records_df[['monotonity']]
monotonity_df.columns = monotonity_df.columns.droplevel()
monotonity_df.to_csv('../data/explanation_results/shap_monotonity.csv')

metrics_selected_records_df

# %%
random_records_to_explain_df = pd.read_csv('../data/explanation_results/random_records_to_explain.csv', index_col=0)
positions_to_explain = random_records_to_explain_df['positions'].to_list()
true_labels_to_explain = random_records_to_explain_df['true labels'].to_list()

metrics_random_records = {}
for clf_name in clf_names:
    classifier = classifiers[clf_name]
    if clf_name in rb_clf_names:
        feature_defaults = features_rb_default
        instances = indicators_test_rb_df.iloc[positions_to_explain].values
    else:
        feature_defaults = features_db_default
        instances = indicators_test_db_df.iloc[positions_to_explain].values

    faithfulness = []
    for i in range(instances.shape[0]):
        prediction = classifier.predict(instances[i].reshape(1,-1))[0]
        feature_importances = shap_values[clf_name][prediction][positions_to_explain[i]]
        sample_metric = evaluate_explanation(classifier, instances[i], feature_importances, feature_defaults[true_labels_to_explain[i]])
        faithfulness.append(sample_metric['faithfulness'])
    
    metrics_random_records[clf_name] = {}
    metrics_random_records[clf_name]['mean faithfulness'] = np.nanmean(faithfulness_df)
    metrics_random_records[clf_name]['std faithfulness'] = np.nanstd(faithfulness_df)

metrics_random_records_df = pd.DataFrame(metrics_random_records)
metrics_random_records_df.to_csv('../data/explanation_results/shap_metrics_random_records.csv')
metrics_random_records_df


