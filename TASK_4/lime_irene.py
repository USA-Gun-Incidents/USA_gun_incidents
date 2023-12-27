# %%
# TODO: spiegare che lo LIME facciamo per tutti essendo agnostic
# (anche per quelli già explainable, magari per esse confrontare queste spiegazioni con le originali)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import lime
import lime.lime_tabular
from IPython.display import HTML, Image
import pydotplus
from sklearn.tree import export_graphviz
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
def show_explanation(explanation):
    display(HTML(f"<div style='background-color:white;'>{explanation.as_html()}</div>"))

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()

incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)

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
NN = 'NeuralNetworkClassifier'
NC = 'NearestCentroidClassifier'
KNN = 'KNearestNeighborsClassifier'
SVM = 'SupportVectorMachineClassifier'
NN = 'NeuralNetworkClassifier'
TN = 'TabNetClassifier'
clf_names = [DT, RF, XGB, NC, KNN, SVM] # TODO: far funzionare con gli altri
data_dir = '../data/classification_results/'
preds = {}
models = {}

for clf_name in clf_names:
    preds[clf_name] = {}
    clf_preds_path = data_dir+clf_name+'_preds.csv'
    clf_preds = pd.read_csv(clf_preds_path)
    preds[clf_name]['labels'] = clf_preds['labels']
    if clf_name != NC and clf_name != KNN and clf_name != SVM:
        preds[clf_name]['probs'] = clf_preds['probs']
    with open(data_dir+clf_name+'.pkl', 'rb') as file:
        models[clf_name] =  pickle.load(file)

# %%
# TODO: try with different kinds of neighbors generation
explainer_db = lime.lime_tabular.LimeTabularExplainer(indicators_train_db_df.values, feature_names = features_db)
explainer_rb = lime.lime_tabular.LimeTabularExplainer(indicators_train_rb_df.values, feature_names = features_rb)

# %% [markdown]
# # Attempted suicide

# %% [markdown]
# Attempted suicides involving a single person:

# %%
attempted_suicide_index = incidents_test_df[(incidents_test_df['suicide']==1) & (true_labels_test_df['death']==0) & (incidents_test_df['n_participants']==1)].index[0]
attempted_suicide_pos = incidents_test_df.index.get_loc(attempted_suicide_index)
incidents_test_df[(incidents_test_df['suicide']==1) & (true_labels_test_df['death']==0) & (incidents_test_df['n_participants']==1)].head(1)

# %% [markdown]
# ## Decision Tree

# %%
explanation = explainer_rb.explain_instance(
    indicators_test_rb_df.iloc[attempted_suicide_pos].values,
    models[DT].predict_proba
)
show_explanation(explanation)

# %%
dot_data = export_graphviz(
    models[DT],
    out_file=None, 
    feature_names=list(indicators_train_rb_df.columns),
    filled=True,
    rounded=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# %% [markdown]
# ## Random Forest

# %%
explanation = explainer_rb.explain_instance(
    indicators_test_rb_df.iloc[attempted_suicide_pos].values,
    models[RF].predict_proba
)
show_explanation(explanation)

# %% [markdown]
# ## Extreme gradient boosting

# %%
explanation = explainer_rb.explain_instance(
    indicators_test_rb_df.iloc[attempted_suicide_pos].values,
    models[XGB].predict_proba
)
show_explanation(explanation)

# %% [markdown]
# ## K Nearest Neighbors

# %%
explanation = explainer_db.explain_instance(
    indicators_test_db_df.iloc[attempted_suicide_pos].values,
    models[KNN].predict_proba
)
show_explanation(explanation)

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
explanation = explainer_db.explain_instance(
    indicators_test_db_df.iloc[attempted_suicide_pos].values,
    models[SVM].predict_proba
)
show_explanation(explanation)

# %%
models[SVM].decision_function(
    X=indicators_test_db_df.iloc[attempted_suicide_pos].values.reshape(-1,len(features_db))
) # TODO: dovrebbe essere la distanza dall'iperpiano, valutare se farci qualcosa

# %% [markdown]
# ## Feed Forward Neural Networks

# %%
explanation = explainer_db.explain_instance(
    indicators_test_db_df.iloc[attempted_suicide_pos].values,
    models[NN].predict_proba
)
show_explanation(explanation)

# %% [markdown]
# ## TabNet

# %%
explanation = explainer_db.explain_instance(
    indicators_test_db_df.iloc[attempted_suicide_pos].values,
    models[TN].predict_proba
)
show_explanation(explanation)

# %% [markdown]
# # Mass shooting

# %% [markdown]
# Involving many killed people:

# %%
max_killed = incidents_test_df['n_killed'].max()
mass_shooting_index = incidents_test_df[incidents_test_df['n_killed'] == max_killed].index[0]
mass_shooting_pos = incidents_test_df.index.get_loc(mass_shooting_index)
incidents_test_df[incidents_test_df['n_killed'] == max_killed].head(1)

# %% [markdown]
# # Incidents predicted as Fatal with highest probability

# %%
indeces_max_prob_death = []
for clf_name in clf_names:
    indeces_max_prob_death.append(preds[clf_name]['probs'].idxmax())

max_prob_death_table = {}
for index in indeces_max_prob_death:
    max_prob_death_table[index] = {}
    max_prob_death_table[index]['True_label'] = true_labels_test_df.iloc[index]['death']
    for clf_name in clf_names:
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
    indeces_min_prob_death.append(preds[clf_name]['probs'].idxmin())

min_prob_death_table = {}
for index in indeces_min_prob_death:
    min_prob_death_table[index] = {}
    min_prob_death_table[index]['True_label'] = true_labels_test_df.iloc[index]['death']
    for clf_name in clf_names:
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
indeces_unknown_death = []
for clf_name in clf_names:
    indeces_unknown_death.append(np.abs(preds[clf_name]['probs']-0.5).idxmin())

unknown_death_table = {}
for index in indeces_unknown_death:
    unknown_death_table[index] = {}
    unknown_death_table[index]['True_label'] = true_labels_test_df.iloc[index]['death']
    for clf_name in clf_names:
        unknown_death_table[index][clf_name+'_pos_prob'] = preds[clf_name]['probs'][index]
unknown_death_table = pd.DataFrame(unknown_death_table).T
unknown_death_table.style.background_gradient(cmap='Blues', axis=1)

# %%
pd.concat([
    unknown_death_table.reset_index(),
    incidents_test_df.iloc[indeces_unknown_death].reset_index()[features_to_explore]],
    axis=1
)


