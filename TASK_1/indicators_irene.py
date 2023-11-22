# %% [markdown]
# # Definition and study of indicators

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
%matplotlib inline

# %% [markdown]
# We load the dataset and reaname some columns:

# %%
incidents_df = pd.read_csv('../data/incidents_cleaned.csv')
dataset_original_columns = incidents_df.columns
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')
incidents_df.rename(
    columns={
        'congressional_district': 'congd',
        'min_age_participants': 'min_age',
        'avg_age_participants': 'avg_age',
        'max_age_participants': 'max_age',
        'n_participants_child': 'n_child',
        'n_participants_teen': 'n_teen',
        'n_participants_adult': 'n_adult'
    },
    inplace=True
)

# %% [markdown]
# We associate to each record the semester (1 or 2) in which the incident occurred:

# %%
incidents_df['sem'] = (incidents_df['date'].dt.month // 7) + 1

# %% [markdown]
# For each record we copmute the ratio between the value of a specific feature in that record with the value of an aggregation function (e.g. sum or median) applied to the feature in a certain time and space window.
# The functions define below perform this computation:

# %%
def compute_window_ratio_indicator(df, gby, feature, agg_fun, suffix):
	grouped_df = df.groupby(gby)[feature].agg(agg_fun)
	df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
	df[feature+'_'+suffix+'_ratio'] = df[feature] / df[feature+suffix]
	df.loc[np.isclose(df[feature], 0), feature+'_'+suffix+'_ratio'] = 1 # TODO: when 0/0 => 1
	df.drop(columns=[feature+suffix], inplace=True)
	return df

# %% [markdown]
# We apply that function to most of the numerical features of the dataset, using:
# - as aggregation functions the sum and the mean
# - as time window the semester
# - as space window both the congressional district

# %%
# numerical columns to use to compute local ratios
window_features = [
    'n_males',
    'n_females',
    'n_killed',
    'n_injured',
    'n_arrested',
    'n_unharmed',
    'n_participants',
    'avg_age',
    'max_age',
    'min_age',
    'n_adult',
    'n_teen',
    'n_child'
]
for column in window_features:
    # column / (sum(column) in the same semester and congressional district)
    incidents_df = compute_window_ratio_indicator(
        df=incidents_df,
        gby=['year', 'sem', 'state', 'congd'],
        feature=column,
        agg_fun='sum',
        suffix='_sum_sem_congd'
    )
    # column / (sum(column) in the same year and state
    incidents_df = compute_window_ratio_indicator(
        df=incidents_df,
        gby=['year', 'state'],
        feature=column,
        agg_fun='sum',
        suffix='_sum_year_state'
    )
    # column / (mean(column) in the same semester and congressional district)
    incidents_df = compute_window_ratio_indicator(
        df=incidents_df,
        gby=['year', 'sem', 'state', 'congd'],
        feature=column,
        agg_fun='mean',
        suffix='_mean_sem_congd'
    )
    # column / (mean(column) in the same year and state)
    incidents_df = compute_window_ratio_indicator(
        df=incidents_df,
        gby=['year', 'state'],
        feature=column,
        agg_fun='mean',
        suffix='_mean_year_state'
    )
# store the names of the new columns
window_ratios_wrt_mean = []
window_ratios_wrt_total = []
for column in incidents_df.columns:
    if 'mean' in column:
        window_ratios_wrt_mean.append(column)
    elif 'sum' in column:
        window_ratios_wrt_total.append(column)

# %% [markdown]
# We visualize the distributions of the ratios w.r.t the mean:

# %%
fig, ax = plt.subplots(figsize=(40, 5))
sns.violinplot(data=incidents_df[window_ratios_wrt_mean], ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# We define a function to apply a logarithmic transformation to the ratio indicators (we sum 1% of the minimum value excluding 0 to avoid infinite values):

# %%
def log_transform(df, columns):
    for col in columns:
        eps = (df[df[col]!=0][col].min())/100 # 1% of the minimum value
        df['log_'+col] = np.log(df[col] + eps)
    return df

# %% [markdown]
# We apply a logarithmic transformation to the ratio indicators displayed above:

# %%
incidents_df = log_transform(df=incidents_df, columns=window_ratios_wrt_mean)
# store the names of the trasnformed columns
log_window_ratios_mean = []
for column in window_ratios_wrt_mean:
    log_window_ratios_mean.append('log_'+column)

# %% [markdown]
# We visualize the distributions of the features after the transformation:

# %%
fig, ax = plt.subplots(figsize=(40, 5))
sns.violinplot(data=incidents_df[log_window_ratios_mean], ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# We visualize the distributions of the indicators w.r.t the total:

# %%
fig, ax = plt.subplots(figsize=(40, 5))
sns.violinplot(data=incidents_df[window_ratios_wrt_total], ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# We apply the logarithmic transformation to the ratio indicators w.r.t the total and visualize the distributions after the transformation:

# %%
incidents_df = log_transform(df=incidents_df, columns=window_ratios_wrt_total)
# store the names of the trasnformed columns
log_window_ratios_total = []
for column in window_ratios_wrt_total:
    log_window_ratios_total.append('log_'+column)
# visualize the distributions of the new variables
fig, ax = plt.subplots(figsize=(40, 5))
sns.violinplot(data=incidents_df[log_window_ratios_total], ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# We define a function to compute the ratio of the value a feature w.r.t the value of another feature in the same record:

# %%
def compute_record_level_ratio_indicator(df, num, den):
    df[num+'_'+den+'_ratio'] = df[num] / df[den] # 0/0 never happens
    return df

# %% [markdown]
# We use the function defined above to compute the ratio between the cardinality of a subset of participants and the total number of participants involved in the incident:

# %%
incident_ratio_num_columns = ['n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 'n_adult', 'n_teen', 'n_child']
for column in incident_ratio_num_columns:
    incidents_df = compute_record_level_ratio_indicator(df=incidents_df, num=column, den='n_participants')
# store the names of the new columns
record_level_ratios = []
for column in incidents_df.columns:
    if 'ratio' in column:
        if 'mean' not in column and 'sum' not in column:
            record_level_ratios.append(column)

fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=incidents_df[record_level_ratios], ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
log_record_level_ratios = []
for column in record_level_ratios:
    log_record_level_ratios.append('log_'+column)
incidents_df = log_transform(df=incidents_df, columns=record_level_ratios)

fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=incidents_df[log_record_level_ratios], ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
def compute_entropy_indicator(df, fixed_cols, var_cols):
    occ = df.groupby(fixed_cols)[var_cols].value_counts().reset_index(name='occ')
    tot = df.groupby(fixed_cols).size().reset_index(name='total')
    probs = occ.merge(tot, how='left', on=fixed_cols)

    label = 'entropy'
    for attr in var_cols:
        label += '_' + attr
    label += '_fixing'
    for attr in fixed_cols:
        label += '_' + attr

    probs[label] = -np.log2(probs['occ']/probs['total']) # 0/0 never happens
    probs.drop(columns=['occ', 'total'], inplace=True)
    
    df = df.merge(probs, how='left', on=fixed_cols+var_cols)

    return df

# %%
entropy_single_features = ['month', 'day', 'address_type', 'n_child', 'n_teen', 'n_adult', 'min_age', 'avg_age', 'max_age']
for feature in entropy_single_features:
    incidents_df = compute_entropy_indicator(df=incidents_df, fixed_cols=['year', 'sem', 'state', 'congd'], var_cols=[feature])
    incidents_df = compute_entropy_indicator(df=incidents_df, fixed_cols=['year', 'state'], var_cols=[feature])

incidents_df = compute_entropy_indicator(df=incidents_df, fixed_cols=['year', 'sem', 'state', 'congd'], var_cols=['month', 'day'])
incidents_df = compute_entropy_indicator(df=incidents_df, fixed_cols=['year', 'sem', 'state', 'congd'], var_cols=['n_child', 'n_teen', 'n_adult'])

incidents_tags = [
    'firearm', 'air_gun', 'shots', 'aggression',
    'suicide', 'injuries', 'death', 'road',
    'illegal_holding', 'house', 'school', 'children',
    'drugs', 'officers', 'organized', 'social_reasons',
    'defensive', 'workplace', 'abduction', 'unintentional'
    ]
incidents_df = compute_entropy_indicator(df=incidents_df, fixed_cols=['year', 'sem', 'state', 'congd'], var_cols=incidents_tags)
incidents_df.rename(columns={incidents_df.columns[-1]: 'entropy_tag_congd'}, inplace=True)

entropies = []
for column in incidents_df.columns:
    if 'entropy' in column:
        entropies.append(column)

# %%
fig, ax = plt.subplots(figsize=(30, 5))
sns.violinplot(data=incidents_df[entropies], ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
incidents_df['severity'] = (
    0.7*incidents_df['n_killed'] + \
    0.3*incidents_df['n_injured']
    ) / (incidents_df['n_participants']-incidents_df['n_unharmed'])
incidents_df['severity'].replace([np.inf], 0, inplace=True)
sns.violinplot(data=incidents_df[['severity']])

# %%
incidents_df['age_range'] = incidents_df['max_age'] - incidents_df['min_age']
sns.violinplot(data=incidents_df[['age_range']])

# %%
latlong_projs = utm.from_latlon(incidents_df['latitude'].to_numpy(), incidents_df['longitude'].to_numpy())
incidents_df['latitude_proj'] = latlong[:,0]
incidents_df['longitude_proj'] = latlong[:,1]

# %%
chosen_indicators = []
incidents_df[chosen_indicators].to_csv('../data/indicators.csv')
incidents_df[dataset_original_columns+chosen_indicators].to_csv('../data/incidents_with_indicators.csv')

# %% [markdown]
# severity # da aggiungere
# 
# **Precedentemente**
# 
# age_range
# 
# n_participants_child_prop
# 
# n_participants_teen_prop
# 
# n_males_pr
# 
# n_killed_pr
# 
# n_injured_pr
# 
# n_arrested_pr
# 
# n_unharmed_pr
# 
# log_n_males_n_males_mean_semest_congd_ratio
# 
# log_avg_age_mean_SD # da togliere
# 
# avg_age_entropy
# 
# city_entropy
# 
# address_entropy
# 
# n_adults_entropy
# 
# tags_entropy
# 
# 
# ---------
# 
# Teniamo qualcosa in più degli indicatori con window
# 
# Togliere ridondanza su maschi
# 
# ---------
# 
# FROM ORIGINAL DATASET
# 
# latitude,longitude,location_importance,avg_age_participants,n_participants
# 
# Sceglierli anche in base ai nan...

# %%
# TODO:
# analisi qualità, missing values, etc...
# distribuzioni dei valori non nan su tutto
# local outlier factor
# pca
# studio correlazione
# definire ordinamento

# applicare ad esempio semplice e verificare correttezza

# %%
df = pd.DataFrame({
    'n_males': [0, 10, np.nan, 5, 5, np.nan, np.nan, np.nan, 0, np.nan],
    'n_participants': [0, 0,10,10,10, np.nan, np.nan, np.nan, 0, np.nan],
    'year':[2013, 2013, 2013, 2014, 2014, 2014, 2014, 2014, 2015, 2015],
    'state':['Illinois', 'Illinois', 'Illinois', 'California', 'California', 'California', 'Colorado', 'Colorado', 'New York', 'New York']
    })
df

confinment_columns_congd = ['year', 'state']
df = compute_local_ratio_indicator(df, confinment_columns_congd, 'n_males', 'n_males', 'mean', '_mean_year_state', )
df

# 5/0 impossibile
# 5/nan impossibile
# 0/nan impossibile
# nan/10 => nan e.g. terzo Illinois
# nan/nan => nan e.g. Colorado
# nan/0 => nan e.g. secondo New York
# 0/0 => 1 PRIMO NEW YORK se media, se somma?


