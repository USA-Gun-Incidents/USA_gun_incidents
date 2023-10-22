# %%
import pandas as pd
import numpy as np

# %%
DATA_FOLDER_PATH = './data/'
incidents_path = DATA_FOLDER_PATH + 'incidents.csv'
incidents_data = pd.read_csv(incidents_path, low_memory=False)

# %%
incidents_data['participant_age1'] = pd.to_numeric(incidents_data['participant_age1'], errors='coerce')
incidents_data['n_males'] = pd.to_numeric(incidents_data['n_males'], errors='coerce')
incidents_data['n_females'] = pd.to_numeric(incidents_data['n_females'], errors='coerce')
incidents_data['n_killed'] = pd.to_numeric(incidents_data['n_killed'], errors='coerce')
incidents_data['n_injured'] = pd.to_numeric(incidents_data['n_injured'], errors='coerce')
incidents_data['n_arrested'] = pd.to_numeric(incidents_data['n_arrested'], errors='coerce')
incidents_data['n_unharmed'] = pd.to_numeric(incidents_data['n_unharmed'], errors='coerce')
incidents_data['n_participants'] = pd.to_numeric(incidents_data['n_participants'], downcast='unsigned', errors='coerce')
incidents_data['min_age_participants'] = pd.to_numeric(incidents_data['min_age_participants'], errors='coerce')
incidents_data['max_age_participants'] = pd.to_numeric(incidents_data['max_age_participants'], errors='coerce')
incidents_data['n_participants_child'] = pd.to_numeric(incidents_data['n_participants_child'], errors='coerce')
incidents_data['n_participants_teen'] = pd.to_numeric(incidents_data['n_participants_teen'], errors='coerce')
incidents_data['n_participants_adult'] = pd.to_numeric(incidents_data['n_participants_adult'], errors='coerce')

# float
incidents_data['avg_age_participants'] = pd.to_numeric(incidents_data['avg_age_participants'], errors='coerce')

# CATEGORICAL ATTRIBUTES
# nominal
incidents_data['participant_gender1'] = incidents_data['participant_gender1'].astype("category")
# ordinal
incidents_data['participant_age_group1'] = incidents_data['participant_age_group1'].astype(
    pd.api.types.CategoricalDtype(categories = ["Child 0-11", "Teen 12-17", "Adult 18+"], ordered = True))

# %%
incidents_data.info()

# %%
incidents_data.describe(include='all')

# %%
incidents_data[(incidents_data['max_age_participants']>100) & (incidents_data['max_age_participants']<200)]

# %%
def set_uint8_oor_to_nan(x):
    if np.issubdtype(type(x), np.integer) and 0 <= x < np.iinfo(np.uint8).max:
        return x
    return np.nan

MAX_AGE = 150
def set_age_oor_to_nan(x):
    if np.issubdtype(type(x), np.integer) and 0 <= x < MAX_AGE:
        return x
    return np.nan

incidents_data['participant_age1'] = incidents_data['participant_age1'].apply(set_age_oor_to_nan).astype('UInt8')
incidents_data['min_age_participants'] = incidents_data['min_age_participants'].apply(set_age_oor_to_nan).astype('UInt8')
incidents_data['avg_age_participants'] = incidents_data['avg_age_participants'].apply(lambda x: x if 0 <= x < MAX_AGE else np.nan).astype('Float32')
incidents_data['max_age_participants'] = incidents_data['max_age_participants'].apply(set_age_oor_to_nan).astype('UInt8')
incidents_data['n_participants_child'] = incidents_data['n_participants_child'].apply(set_uint8_oor_to_nan).astype('UInt8')
incidents_data['n_participants_teen'] = incidents_data['n_participants_teen'].apply(set_uint8_oor_to_nan).astype('UInt8')
incidents_data['n_participants_adult'] = incidents_data['n_participants_adult'].apply(set_uint8_oor_to_nan).astype('UInt8')
incidents_data['n_males'] = incidents_data['n_males'].apply(set_uint8_oor_to_nan).astype('UInt8')
incidents_data['n_females'] = incidents_data['n_females'].apply(set_uint8_oor_to_nan).astype('UInt8')
incidents_data['n_arrested'] = incidents_data['n_arrested'].apply(set_uint8_oor_to_nan).astype('UInt8')
incidents_data['n_unharmed'] = incidents_data['n_unharmed'].apply(set_uint8_oor_to_nan).astype('UInt8')
incidents_data['n_killed'] = incidents_data['n_killed'].astype('UInt8')
incidents_data['n_injured'] = incidents_data['n_injured'].astype('UInt8')
incidents_data['n_participants'] = incidents_data['n_participants'].astype('UInt8')

# %%
incidents_data.info()

# %%
# questo codice dovrebbe fare l'equivalente di quello che viene fatto da 265 a 497
# non ho usato lazy evaluation perchè non so se i check vengono parallelizzati
# se ci sono valori a nan si assume siano consistenti

incidents_data['age_consistency'] = True
incidents_data.loc[
    (
    (incidents_data['min_age_participants'] > incidents_data['max_age_participants']) |
    (incidents_data['min_age_participants'] > incidents_data['avg_age_participants']) |
    ((incidents_data['min_age_participants'] < 12) & (incidents_data['n_participants_child'] == 0)) |
    ((incidents_data['min_age_participants']>=12) & (incidents_data['min_age_participants'] < 18) & ((incidents_data['n_participants_teen'] == 0) | (incidents_data['n_participants_child']  > 0))) |
    ((incidents_data['min_age_participants'] >= 18) & ((incidents_data['n_participants_adult'] == 0) | (incidents_data['n_participants_child'] > 0) | (incidents_data['n_participants_teen'] > 0))) |
    ((incidents_data['max_age_participants'] < 12) & ((incidents_data['n_participants_child'] == 0) | (incidents_data['n_participants_teen'] > 0) | (incidents_data['n_participants_adult'] > 0))) |
    ((incidents_data['max_age_participants'] >= 12) & (incidents_data['max_age_participants'] < 18) & ((incidents_data['n_participants_teen'] == 0) | (incidents_data['n_participants_adult'] > 0))) |
    ((incidents_data['max_age_participants'] >= 18) & (incidents_data['n_participants_adult'] == 0))
    ), # n_child + n_teen + n_adult <= 0??
    'age_consistency'] = False

incidents_data['gender_consistency'] = True
incidents_data.loc[
    incidents_data['n_males'] + incidents_data['n_females'] != incidents_data['n_participants'],
    'gender_consistency'] = False

incidents_data['n_participant_consistency'] = True
incidents_data.loc[
    (incidents_data['n_killed'] + incidents_data['n_injured'] > incidents_data['n_participants']) |
    (incidents_data['n_arrested'] > incidents_data['n_participants']) |
    (incidents_data['n_unharmed'] > incidents_data['n_participants']),
    'n_participant_consistency'] = False

incidents_data['participant1_consistency'] = True
incidents_data.loc[
    ((incidents_data['participant_age1'] < 12) & (incidents_data['participant_age_group1'] != 'Child 0-11')) |
    ((incidents_data['participant_age1'] >= 12) & (incidents_data['participant_age1'] < 18) & (incidents_data['participant_age_group1'] != 'Teen 12-17')) |
    ((incidents_data['participant_age1'] >= 18) & (incidents_data['participant_age_group1'] != 'Adult 18+')),
    'participant1_consistency'] = False

incidents_data['participant1_age_wrt_all_data_consistency'] = True
incidents_data.loc[
    (incidents_data['participant_age1'] < incidents_data['min_age_participants']) |
    (incidents_data['participant_age1'] > incidents_data['max_age_participants']),
    'participant1_age_wrt_all_data_consistency'] = False

incidents_data['participant1_age_range_wrt_all_data_consistency'] = True
incidents_data.loc[
    ((incidents_data['participant_age_group1'] == 'Child 0-11') & (incidents_data['n_participants_child'] == 0)) |
    ((incidents_data['participant_age_group1'] == 'Teen 12-17') & (incidents_data['n_participants_teen'] == 0)) |
    ((incidents_data['participant_age_group1'] == 'Adult 18+') & (incidents_data['n_participants_adult'] == 0)),
    'participant1_age_range_wrt_all_data_consistency'] = False

incidents_data['participant1_gender_wrt_all_data_consistency'] = True
incidents_data.loc[
    ((incidents_data['participant_gender1'] == 'Male') & (incidents_data['n_males'] == 0)) |
    ((incidents_data['participant_gender1'] == 'Female') & (incidents_data['n_females'] == 0)),
    'participant1_gender_wrt_all_data_consistency'] = False

incidents_data['participants1_wrt_n_participants_consistency'] = ((incidents_data['participant1_age_wrt_all_data_consistency']) &
    (incidents_data['participant1_age_range_wrt_all_data_consistency']) & (incidents_data['participant1_gender_wrt_all_data_consistency']))

incidents_data['nan_values'] = False
incidents_data.loc[incidents_data.isnull().any(axis=1), 'nan_values'] = True



