# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np

# %%
DATA_FOLDER_PATH = '../../data/'
incidents_path = DATA_FOLDER_PATH + 'incidents.csv'
incidents_data = pd.read_csv(incidents_path, low_memory=False)

# %%
# TODO: questo viene fatto anche nel notebook task 1

numerical_attributes = [
    'participant_age1',
    'n_males',
    'n_females',
    'n_killed',
    'n_injured',
    'n_arrested',
    'n_unharmed',
    'n_participants',
    'min_age_participants',
    'max_age_participants',
    'n_participants_child',
    'n_participants_teen',
    'n_participants_adult',
    'avg_age_participants'
]
incidents_data[numerical_attributes] = incidents_data[numerical_attributes].apply(pd.to_numeric, errors='coerce')

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
MAX_AGE = 130 # TODO: forse va cambiato
MAX_PARTICIPANTS = 1000 # TODO: forse va cambiato
integer_cols = [
    'participant_age1',
    'min_age_participants',
    'max_age_participants',
    'n_participants_child',
    'n_participants_teen',
    'n_participants_adult', 
    'n_males',
    'n_females',
    'n_arrested',
    'n_unharmed'
]
age_cols = [
    'participant_age1',
    'min_age_participants',
    'avg_age_participants',
    'max_age_participants'
]
participant_cols = [
    'n_participants_child',
    'n_participants_teen',
    'n_participants_adult',
    'n_males',
    'n_females',
    'n_arrested',
    'n_unharmed'
]


# %%
for col in integer_cols:
    incidents_data[col] = incidents_data[col].apply(lambda x: x if float.is_integer(x) else np.nan)

for col in age_cols:
    incidents_data[col] = incidents_data[col].apply(lambda x: x if x>=0 and x<=MAX_AGE else np.nan)

for col in participant_cols:
    incidents_data[col] = incidents_data[col].apply(lambda x: x if x>=0 and x<=MAX_PARTICIPANTS else np.nan)

# %%
incidents_data.describe()

# %%
incidents_data['age_consistency'] = True
incidents_data.loc[
    ((incidents_data['min_age_participants'] > incidents_data['max_age_participants']) |
    (incidents_data['min_age_participants'] > incidents_data['avg_age_participants']) |
    ((incidents_data['min_age_participants'] < 12) & (incidents_data['n_participants_child'] == 0)) |
    ((incidents_data['min_age_participants'] >= 12) & (incidents_data['min_age_participants'] < 18) & ((incidents_data['n_participants_teen'] == 0) | (incidents_data['n_participants_child']  > 0))) |
    ((incidents_data['min_age_participants'] >= 18) & ((incidents_data['n_participants_adult'] == 0) | (incidents_data['n_participants_child'] > 0) | (incidents_data['n_participants_teen'] > 0))) |
    ((incidents_data['max_age_participants'] < 12) & ((incidents_data['n_participants_child'] == 0) | (incidents_data['n_participants_teen'] > 0) | (incidents_data['n_participants_adult'] > 0))) |
    ((incidents_data['max_age_participants'] >= 12) & (incidents_data['max_age_participants'] < 18) & ((incidents_data['n_participants_teen'] == 0) | (incidents_data['n_participants_adult'] > 0))) |
    ((incidents_data['max_age_participants'] >= 18) & (incidents_data['n_participants_adult'] == 0)) |
    ((incidents_data['n_participants_child'] == 0) & (incidents_data['n_participants_teen'] == 0) & (incidents_data['n_participants_adult'] == 0))),
    'age_consistency'] = False

incidents_data.loc[
    ((incidents_data['min_age_participants'].isna()) |
    (incidents_data['max_age_participants'].isna()) |
    (incidents_data['avg_age_participants'].isna()) |
    (incidents_data['n_participants_child'].isna()) |
    (incidents_data['n_participants_teen'].isna()) |
    (incidents_data['n_participants_adult'].isna())),
    'age_consistency'] = False

incidents_data['gender_consistency'] = True
incidents_data.loc[
    incidents_data['n_males'] + incidents_data['n_females'] != incidents_data['n_participants'],
    'gender_consistency'] = False

incidents_data.loc[
    (((incidents_data['n_males'].isna()) | 
    (incidents_data['n_females'].isna())) |
    (incidents_data['n_participants_adult'].isna())),
    'gender_consistency'] = False

incidents_data['n_participant_consistency'] = True
incidents_data.loc[
    (incidents_data['n_killed'] + incidents_data['n_injured'] > incidents_data['n_participants']) |
    (incidents_data['n_arrested'] > incidents_data['n_participants']) |
    (incidents_data['n_unharmed'] > incidents_data['n_participants']),
    'n_participant_consistency'] = False

incidents_data.loc[
    ((incidents_data['n_killed'].isna()) |
    (incidents_data['n_injured'].isna()) |
    (incidents_data['n_arrested'].isna()) |
    (incidents_data['n_unharmed'].isna()) |
    (incidents_data['n_participants'].isna())),
    'n_participant_consistency'] = False

incidents_data['participant1_consistency'] = True
incidents_data.loc[
    ((incidents_data['participant_age1'] < 12) & (incidents_data['participant_age_group1'] != 'Child 0-11')) |
    ((incidents_data['participant_age1'] >= 12) & (incidents_data['participant_age1'] < 18) & (incidents_data['participant_age_group1'] != 'Teen 12-17')) |
    ((incidents_data['participant_age1'] >= 18) & (incidents_data['participant_age_group1'] != 'Adult 18+')),
    'participant1_consistency'] = False

incidents_data.loc[
    ((incidents_data['participant_age1'].isna()) |
    (incidents_data['participant_age_group1'].isna())),
    'participant1_consistency'] = False

incidents_data['participant1_age_wrt_all_data_consistency'] = True
incidents_data.loc[
    (incidents_data['participant_age1'] < incidents_data['min_age_participants']) |
    (incidents_data['participant_age1'] > incidents_data['max_age_participants']),
    'participant1_age_wrt_all_data_consistency'] = False

incidents_data.loc[
    ((incidents_data['participant_age1'].isna()) |
    (incidents_data['min_age_participants'].isna()) |
    (incidents_data['max_age_participants'].isna()))
    , 'participant1_age_wrt_all_data_consistency'] = False

incidents_data['participant1_age_range_wrt_all_data_consistency'] = True
incidents_data.loc[
    ((incidents_data['participant_age_group1'] == 'Child 0-11') & (incidents_data['n_participants_child'] == 0)) |
    ((incidents_data['participant_age_group1'] == 'Teen 12-17') & (incidents_data['n_participants_teen'] == 0)) |
    ((incidents_data['participant_age_group1'] == 'Adult 18+') & (incidents_data['n_participants_adult'] == 0)),
    'participant1_age_range_wrt_all_data_consistency'] = False

incidents_data.loc[
    (incidents_data['participant_age_group1'].isna()),
    'participant1_age_range_wrt_all_data_consistency'] = False # TODO: forse bisogna controllare quando non nan se il numero di partecipanti della classe a cui appartiene è nan?

incidents_data['participant1_gender_wrt_all_data_consistency'] = True
incidents_data.loc[
    ((incidents_data['participant_gender1'] == 'Male') & (incidents_data['n_males'] == 0)) |
    ((incidents_data['participant_gender1'] == 'Female') & (incidents_data['n_females'] == 0)),
    'participant1_gender_wrt_all_data_consistency'] = False

incidents_data.loc[
    (incidents_data['participant_gender1'].isna()),
    'participant1_gender_wrt_all_data_consistency'] = False # TODO: come sopra?

incidents_data['participants1_wrt_n_participants_consistency'] = ((incidents_data['participant1_age_wrt_all_data_consistency']) &
    (incidents_data['participant1_age_range_wrt_all_data_consistency']) & (incidents_data['participant1_gender_wrt_all_data_consistency']))

incidents_data.loc[
    ((incidents_data['participant1_age_wrt_all_data_consistency'].isna()) |
     (incidents_data['participant1_age_range_wrt_all_data_consistency'].isna()) |
     (incidents_data['participant1_gender_wrt_all_data_consistency'].isna())),
    'participants1_wrt_n_participants_consistency'] = False

incidents_data['nan_values'] = False
incidents_data.loc[incidents_data.isnull().any(axis=1), 'nan_values'] = True # TODO: serve? e se sì va limitato a age, gender, etc?

# %%
incidents_data.describe()


# %%
def check(x): # TODO: possiamo sostituirlo con in [np.nan]
    if np.isnan(x['n_females']):
        print("isnull")
    return x

incidents_data = incidents_data[:10].apply(check, axis=1)
