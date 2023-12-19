# %% [markdown]
# # Definition and study of the features for the classification task

# %% [markdown]
# In this notebook, we extract new features describing the incidents to enable the classification task.

# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
sys.path.append(os.path.abspath('..'))
from plot_utils import *
# %matplotlib inline
from classification_utils import *

# %% [markdown]
# We load the dataset and reaname some columns:

# %%
incidents_df = pd.read_csv('../data/incidents_cleaned.csv', index_col=0)
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')
incidents_df.rename(
    columns={
        'congressional_district': 'congd',
        'min_age_participants': 'min_age',
        'avg_age_participants': 'avg_age',
        'max_age_participants': 'max_age',
        'n_participants_child': 'n_child',
        'n_participants_teen': 'n_teen',
        'n_participants_adult': 'n_adult',
        'location_importance': 'location_imp'
    },
    inplace=True
)
dataset_original_columns = incidents_df.columns

# %%
import pyproj

projector = pyproj.Proj(proj='utm', zone=14, ellps='WGS84', preserve_units=True)
incidents_df['x'], incidents_df['y'] = projector(incidents_df['longitude'], incidents_df['latitude'])

# %%
plt.plot(incidents_df['x'], incidents_df['y'], 'o', markersize=1)
plt.axis('equal')

# %%
incidents_per_state = incidents_df[incidents_df['death']].groupby(['state', 'population_state_2010']).size()
incidents_per_state = ((incidents_per_state / incidents_per_state.index.get_level_values('population_state_2010'))*100000).to_frame(name='incidents_per_100k_inhabitants').sort_values(by='incidents_per_100k_inhabitants', ascending=True)
incidents_per_state.reset_index(inplace=True)
incidents_per_state.plot(
    kind='barh',
    x='state',
    y='incidents_per_100k_inhabitants',
    figsize=(15, 10),
    ylabel='State',
    xlabel='Mortal incidents per 100k inhabitants',
    title='Incidents per 100k inhabitants per state'
)


# %%
# one hot encoding for states
# for state in incidents_df['state'].unique():
#     incidents_df[state] = (incidents_df['state'] == state).astype(int)

# %%
def compute_record_level_ratio_indicator(df, num, den):
    df[num+'_'+den+'_ratio'] = df[num] / df[den]
    return df

# %% [markdown]
# We use the function defined above to compute the ratio between the cardinality of a subset of participants and the total number of participants involved in the incident and we visualize the distributions of the computed indicators:

# %%
incident_ratio_num_columns = ['n_males', 'n_females', #'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', # TODO: togliere
                              'n_adult', 'n_teen', 'n_child']
for feature in incident_ratio_num_columns:
    incidents_df = compute_record_level_ratio_indicator(df=incidents_df, num=feature, den='n_participants')
# store the names of the new features
record_level_ratios = []
for feature in incidents_df.columns:
    if 'ratio' in feature:
        record_level_ratios.append(feature)
# visualize the distributions of the features
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=incidents_df[record_level_ratios], ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# Additionally, we compute the age range of the participants involved in the incident and we visualize its distribution:

# %%
incidents_df['age_range'] = incidents_df['max_age'] - incidents_df['min_age']
sns.violinplot(data=incidents_df[['age_range']])

# %% [markdown]
# We define a list of the most relevant indicators with their abbreviations:

# %%
indicators_abbr = {
    # spatial data
    'location_imp': 'location_imp',
    'x': 'x',
    'y': 'y',
    # age data
    'age_range': 'age_range',
    'avg_age': 'avg_age',
    'n_child_n_participants_ratio': 'n_child_prop',
    'n_teen_n_participants_ratio': 'n_teen_prop',
    # gender data
    'n_males_n_participants_ratio': 'n_males_prop',
    # characteristics data
    'n_participants': 'n_participants',
    # temporal data
    'month': 'month',
    'day_of_week': 'day_of_week',
    # socio-economic data
    'poverty_perc': 'poverty_perc',
    'party': 'democrat'
}
# add incidents tags
incidents_tags = [
    #'firearm', 'air_gun', 'shots', # TODO: togliere
    'aggression',
    #'suicide', 'injuries', 'death',
    'road',
    'illegal_holding', 'house', 'school', 'children',
    'drugs', 'officers', 'organized', 'social_reasons',
    'defensive', 'workplace', 'abduction', 'unintentional'
    ]
for tag in incidents_tags:
    if tag=='children': # not added because we already have n_child_prop
        continue
    indicators_abbr[tag] = tag
incidents_df.rename(columns=indicators_abbr, inplace=True)

# %% [markdown]
# We convert to 0 or 1 binary attributes:

# %%
incidents_df['democrat'].replace(['REPUBLICAN', 'DEMOCRAT'], [0, 1], inplace=True)
for tag in incidents_tags:
    incidents_df[tag].replace([True, False], [1, 0], inplace=True)

# %% [markdown]
# We compute the correlation between the features:

# %%
# compute the pearson's correlation coefficient between the features
fig, ax = plt.subplots(figsize=(30, 15))
pearson_corr_matrix = incidents_df[list(indicators_abbr.values()) + ['death']].corr('pearson')
sns.heatmap(pearson_corr_matrix, annot=True, ax=ax, mask=np.triu(pearson_corr_matrix), cmap='coolwarm')

# %%
# compute the spearman's correlation coefficient between the features
fig, ax = plt.subplots(figsize=(30, 15))
spearman_corr_matrix = incidents_df[list(indicators_abbr.values()) + ['death']].corr('spearman')
sns.heatmap(spearman_corr_matrix, annot=True, ax=ax, mask=np.triu(spearman_corr_matrix), cmap='coolwarm')

# %%
pearson_corr_matrix['death'].to_frame().style.background_gradient(cmap='coolwarm', axis=None).format(precision=3)

# %%
spearman_corr_matrix['death'].to_frame().style.background_gradient(cmap='coolwarm', axis=None).format(precision=3)

# %%
scatter_by_label(
    incidents_df,
    ['location_imp',
    'age_range',
    'avg_age',
    'n_child_prop',
    'n_teen_prop',
    'n_males_prop',
    'n_participants',
    'month',
    'day_of_week',
    'poverty_perc'],
    'death',
    ncols=3,
    figsize=(35, 50)
)

# %%
# TODO:
# pca = PCA()
# X_pca = pca.fit_transform(X)
# scatter_pca_features_by_label(
#     X_pca,
#     n_components,
#     labels,
#     palette,
#     title=None
# )

# %% [markdown]
# We check for duplicated rows:

# %%
n_duplicates = incidents_df[list(indicators_abbr.values())].duplicated().sum()
print(f"Number of duplicated rows: {n_duplicates}")
print(f"Percentage of duplicated rows: {(n_duplicates/incidents_df[list(indicators_abbr.values())].shape[0])*100:.2f}%")

# %% [markdown]
# We save the names of the indicators in a json file and we save the dataset with the indicators:

# %%
import json
with open('../data/clf_indicators_names.json', 'w') as f:
    json.dump(list(indicators_abbr.values()), f)

# TODO: evitare duplicati con nomi diversi
original_features_minus_indicators = [feature for feature in dataset_original_columns if feature not in list(indicators_abbr.values())]
original_features_minus_indicators.remove('party')
incidents_df[original_features_minus_indicators + list(indicators_abbr.values())].to_csv('../data/clf_incidents_indicators.csv')
incidents_df[list(indicators_abbr.values())].to_csv('../data/clf_indicators.csv')

# %% [markdown]
# We visualize the number of nan values for each indicator:

# %%
incidents_df[list(indicators_abbr.values())].info()

# %%
print(f'The dataset has {incidents_df.shape[0]} rows')
print(f'Dropping rows with nan values in the indicators columns, {incidents_df[list(indicators_abbr.values())].dropna().shape[0]} rows remain')

# %% [markdown]
# We display a summary of the descriptive statistics of the indicators:

# %%
incidents_df[list(indicators_abbr.values())].describe()

# %%
incidents_clf = incidents_df[list(indicators_abbr.values())+['death']].dropna()

# %%
fig, ax = plt.subplots(figsize=(5, 5))
incidents_clf['death'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)

# %% [markdown]
# ## Final Indicators semantics
#
# | Name | Description | Present in the original dataset |
# | :--: | :---------: | :-----------------------------: |
# | location_imp | Location importance according to Geopy | No |
# | age_range | Difference between the maximum and the minimum age of the participants involved in the incident | No |
# | avg_age | Average age of the participants involved in the incident | Yes |
# | n_child_prop | Ratio between the number of child involved in the incident and number of people involved in the incident | No |
# | n_teen_prop | Ratio between the number of teen involved in the incident and number of people involved in the incident | No |
# | n_males_prop | Ratio between the number of males and the number of people involed in the incident | No |
# | n_participants | Number of participants involved in the incident | Yes |
# | poverty_perc | Poverty percentage in the state and year of the incident | Yes |
# | democrat | Winning party in the congressional_district and year of the incident | Yes |
# | aggression | Whether the incident involved an aggression (both with a gun or not) | No (extracted from the incident characteristics) |
# | road | Whether the incident happene in a road | No (extracted from the incident characteristics) |
# | illegal_holding | Whether the incident involved a stealing act or a gun was illegally possessed | No (extracted from the incident characteristics) |
# | house | Whether the incident happened in a house | No (extracted from the incident characteristics) |
# | school | Whether the incident happened in a school | No (extracted from the incident characteristics) |
# | drugs | Whether the incident involved drugs | No (extracted from the incident characteristics) |
# | officers | Whether one or more officiers were involved in the incident | No (extracted from the incident characteristics) |
# | organized | Whether the action was planned by an organization or a group | No (extracted from the incident characteristics) |
# | social_reasons | Whether the incident involved social discriminations or terrorism | No (extracted from the incident characteristics) |
# | defensive | Whether the incident involved the use of a gun for defensive purposes | No (extracted from the incident characteristics) |
# | workplace | Whether the incident happened in a workplace | No (extracted from the incident characteristics) |
# | abduction | Whether the incident involved any form of abduction | No (extracted from the incident characteristics) |
# | unintentional | Whether the incident was unintentional | No (extracted from the incident characteristics) |


