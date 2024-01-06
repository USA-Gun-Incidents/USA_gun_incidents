# %%
import pandas as pd
from explanation_utils import *

# %%
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %%
incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

clf_names = [clf.value for clf in Classifiers]

DATA_DIR = '../data/classification_results/'
preds = get_classifiers_predictions(DATA_DIR)

# %%
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)

# %%
(
    incidents_train_df[(incidents_train_df['suicide']==1) & (true_labels_train_df['death']==0)].shape[0] + 
    incidents_test_df[(incidents_test_df['suicide']==1) & (true_labels_test_df['death']==0)].shape[0]
    ) / \
(
    incidents_train_df[(incidents_train_df['suicide']==1) & (true_labels_train_df['death']==1)].shape[0] + 
    incidents_test_df[(incidents_test_df['suicide']==1) & (true_labels_test_df['death']==1)].shape[0]
)*100

# %%
selected_records_to_explain = {}
selected_records_to_explain['positions'] = []
selected_records_to_explain['instance names'] = []
selected_records_to_explain['true labels'] = []

# %% [markdown]
# ## Attempted suicides

# %%
attempted_suicides = incidents_test_df[
    (incidents_test_df['suicide']==1) &
    (true_labels_test_df['death']==0) &
    (incidents_test_df['n_participants']==1)
]
attempted_suicides

# %%
attempted_suicide_index = attempted_suicides.index[0]
attempted_suicide_pos = incidents_test_df.index.get_loc(attempted_suicide_index)
selected_records_to_explain['positions'].append(attempted_suicide_pos)
selected_records_to_explain['instance names'].append('Attempted Suicide')
selected_records_to_explain['true labels'].append(true_labels_test[attempted_suicide_pos])

# %% [markdown]
# ## Mass shootings

# %%
max_killed = incidents_test_df['n_killed'].max()
mass_shooting = incidents_test_df[incidents_test_df['n_killed'] == max_killed]
mass_shooting

# %%
mass_shooting_index = mass_shooting.index[0]
mass_shooting_pos = incidents_test_df.index.get_loc(mass_shooting_index)
selected_records_to_explain['positions'].append(mass_shooting_pos)
selected_records_to_explain['instance names'].append('Mass shooting')
selected_records_to_explain['true labels'].append(true_labels_test[mass_shooting_pos])

# %% [markdown]
# ## Incidents predicted as Fatal with highest probability

# %%
indeces_max_prob_death = []
for clf_name in clf_names:
    if clf_name != Classifiers.NC.value and clf_name != Classifiers.KNN.value:
        pos = preds[clf_name]['probs'].idxmax()
        indeces_max_prob_death.append(pos)
        selected_records_to_explain['positions'].append(pos)
        selected_records_to_explain['instance names'].append(f'Fatal with highest confidence by {clf_name}')
        selected_records_to_explain['true labels'].append(true_labels_test[pos])

max_prob_death_table = {}
for index in indeces_max_prob_death:
    max_prob_death_table[index] = {}
    max_prob_death_table[index]['True_label'] = true_labels_test[index]
    for clf_name in clf_names:
        if clf_name != Classifiers.NC.value and clf_name != Classifiers.KNN.value:
            max_prob_death_table[index][clf_name+'_pos_prob'] = preds[clf_name]['probs'][index]
max_prob_death_table = pd.DataFrame(max_prob_death_table).T
max_prob_death_table.style.background_gradient(cmap='Blues', axis=1)

# %%
pd.concat([
    max_prob_death_table.reset_index(),
    incidents_test_df.iloc[indeces_max_prob_death].reset_index()],
    axis=1
)

# %% [markdown]
# ## Incidents predict as Non-Fatal with highest probability

# %%
indeces_min_prob_death = []
for clf_name in clf_names:
    if clf_name != Classifiers.NC.value and clf_name != Classifiers.KNN.value:
        pos = preds[clf_name]['probs'].idxmin()
        indeces_min_prob_death.append(pos)
        selected_records_to_explain['positions'].append(pos)
        selected_records_to_explain['instance names'].append(f'Non-Fatal with highest confidence by {clf_name}')
        selected_records_to_explain['true labels'].append(true_labels_test[pos])

min_prob_death_table = {}
for index in indeces_min_prob_death:
    min_prob_death_table[index] = {}
    min_prob_death_table[index]['True_label'] = true_labels_test[index]
    for clf_name in clf_names:
        if clf_name != Classifiers.NC.value and clf_name != Classifiers.KNN.value:
            min_prob_death_table[index][clf_name+'_pos_prob'] = preds[clf_name]['probs'][index]
min_prob_death_table = pd.DataFrame(min_prob_death_table).T
min_prob_death_table.style.background_gradient(cmap='Blues', axis=1)

# %%
pd.concat([
    min_prob_death_table.reset_index(),
    incidents_test_df.iloc[indeces_min_prob_death].reset_index()],
    axis=1
)

# %%
## Incidents with the highest uncertainty in the predicted outcomes

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

# pd.concat([
#     unknown_death_table.reset_index(),
#     incidents_test_df.iloc[indeces_unknown_death].reset_index()],
#     axis=1
# )

# %%
selected_records_df = pd.DataFrame(selected_records_to_explain)
selected_records_df.to_csv('../data/explanation_results/selected_records_to_explain.csv')
selected_records_df

# %%
random_records_to_explain = {}
random_records_to_explain['positions'] = np.arange(0, 51) # TODO: decidere se prenderli a caso o con un criterio
random_records_to_explain['true labels'] = true_labels_test[0: 51]
random_records_df = pd.DataFrame(random_records_to_explain)
random_records_df.to_csv('../data/explanation_results/random_records_to_explain.csv')


