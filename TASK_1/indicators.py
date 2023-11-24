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
# %matplotlib inline

# %% [markdown]
# We load the dataset and reaname some columns:

# %%
incidents_df = pd.read_csv('../data/incidents_cleaned.csv')
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

# %% [markdown]
# We associate to each record the semester (1 or 2) in which the incident occurred:

# %%
incidents_df['sem'] = (incidents_df['date'].dt.month // 7) + 1

# %% [markdown]
# For each record we compute the ratio between the value of a specific feature in that record and the value of an aggregation function (e.g. sum or median) applied to the feature in a certain time and space window.
# The functions define below performs this computation:

# %%
def compute_window_ratio_indicator(df, gby, feature, agg_fun, suffix):
	grouped_df = df.groupby(gby)[feature].agg(agg_fun)
	df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
	df[feature+suffix+'_ratio'] = df[feature] / df[feature+suffix]
	df.loc[np.isclose(df[feature], 0), feature+suffix+'_ratio'] = 1 # TODO: when 0/0 => 1
	df.drop(columns=[feature+suffix], inplace=True)
	return df

# %% [markdown]
# We apply that function to most of the numerical features of the dataset, using:
# - as aggregation functions the sum and the mean
# - as time window the semester and the year
# - as space window both the congressional district (that we have previously inferred for all the records) and the state

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

# %% [markdown]
# The **surprisal** (also called [Information content](https://en.wikipedia.org/wiki/Information_content)) of an event $E$ 
# with probability $p(E)$ is defined as $log(1/p(E))$ (or equivalently $-log(p(E))$).
# Surprisal is inversly related to probability (hence the term $1/p(E)$): when $p(E)$ is close to $1$, the surprisal of the event is low, when $p(E)$ is close to $0$, the surprisal of the event is high.
# The $log$ gives $0$ surprise when the probability of the event is $1$.
#
# The surprisal is closely related to **entropy**, which is the expected value of the information content of a random variable, quantifying how surprising the random variable is "on average".

# %%
def compute_surprisal_indicator(df, fixed_cols, var_cols):
    occ = df.groupby(fixed_cols)[var_cols].value_counts().reset_index(name='occ')
    tot = df.groupby(fixed_cols).size().reset_index(name='total')
    probs = occ.merge(tot, how='left', on=fixed_cols)

    label = 'surprisal'
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
surpisal_single_features = ['month', 'day', 'address_type', 'n_child', 'n_teen', 'n_adult', 'min_age', 'avg_age', 'max_age', 'n_killed', 'n_injured', 'n_males', 'n_participants']
for feature in surpisal_single_features:
    incidents_df = compute_surprisal_indicator(df=incidents_df, fixed_cols=['year', 'sem', 'state', 'congd'], var_cols=[feature])
    incidents_df = compute_surprisal_indicator(df=incidents_df, fixed_cols=['year', 'state'], var_cols=[feature])

incidents_df = compute_surprisal_indicator(df=incidents_df, fixed_cols=['year', 'sem', 'state', 'congd'], var_cols=['month', 'day'])
incidents_df = compute_surprisal_indicator(df=incidents_df, fixed_cols=['year', 'sem', 'state', 'congd'], var_cols=['n_child', 'n_teen', 'n_adult'])

incidents_tags = [
    'firearm', 'air_gun', 'shots', 'aggression',
    'suicide', 'injuries', 'death', 'road',
    'illegal_holding', 'house', 'school', 'children',
    'drugs', 'officers', 'organized', 'social_reasons',
    'defensive', 'workplace', 'abduction', 'unintentional'
    ]
incidents_df = compute_surprisal_indicator(df=incidents_df, fixed_cols=['year', 'sem', 'state', 'congd'], var_cols=incidents_tags)
incidents_df.rename(columns={incidents_df.columns[-1]: 'surprisal_characteristics'}, inplace=True)

surprisals = []
for column in incidents_df.columns:
    if 'surprisal' in column:
        surprisals.append(column)

# %%
fig, ax = plt.subplots(figsize=(30, 5))
sns.violinplot(data=incidents_df[surprisals], ax=ax)
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
import utm

def project_lat_long(latidude, longitude):
    # check if the coordinates are valid
    if latidude >= -90 and latidude <= 90 and longitude >= -180 and longitude <= 180:
        utm_coordinates = utm.from_latlon(latidude, longitude)
        return utm_coordinates[0], utm_coordinates[1]
    else:
        return np.nan, np.nan

incidents_df['lat_proj'], incidents_df['lon_proj'] = zip(*incidents_df.apply(
    lambda row: project_lat_long(row['latitude'], row['longitude']), axis=1))

# %%
indicators = {
    # spatial data
    'lat_proj': 'lat_proj',
    'lon_proj': 'lon_proj',
    'location_imp': 'location_imp',
    'surprisal_address_type_fixing_year_sem_state_congd': 'surprisal_address_type',
    # age data
    'age_range': 'age_range',
    'avg_age': 'avg_age',
    'surprisal_min_age_fixing_year_sem_state_congd': 'surprisal_min_age',
    'log_min_age_mean_sem_congd_ratio': 'log_min_age_mean_ratio',
    'n_child_n_participants_ratio': 'n_child_prop',
    'n_teen_n_participants_ratio': 'n_teen_prop',
    'surprisal_n_child_n_teen_n_adult_fixing_year_sem_state_congd': 'surprisal_age_groups',
    # severity data
    'severity': 'severity',
    'n_killed_n_participants_ratio': 'n_killed_prop',
    'surprisal_n_killed_fixing_year_sem_state_congd': 'surprisal_n_killed',
    'log_n_killed_mean_sem_congd_ratio': 'log_n_killed_mean_ratio',
    'n_injured_n_participants_ratio': 'n_injured_prop',
    'log_n_injured_mean_sem_congd_ratio': 'log_n_injured_mean_ratio',
    'surprisal_n_injured_fixing_year_sem_state_congd': 'surprisal_n_injured',
    'n_unharmed_n_participants_ratio': 'n_unharmed_prop',
    # gender data
    'n_males_n_participants_ratio': 'n_males_prop',
    'log_n_males_mean_sem_congd_ratio': 'log_n_males_mean_ratio',
    'surprisal_n_males_fixing_year_sem_state_congd': 'surprisal_n_males',
    # characteristics data
    'surprisal_characteristics': 'surprisal_characteristics',
    'n_arrested_n_participants_ratio': 'n_arrested_prop',
    'log_n_participants_mean_sem_congd_ratio': 'log_n_participants_mean_ratio',
    'surprisal_n_participants_fixing_year_sem_state_congd': 'surprisal_n_participants',
    'n_participants': 'n_participants',
    # temporal data
    'surprisal_month_day_fixing_year_sem_state_congd': 'surprisal_day'
}

# %%
incidents_df.rename(columns=indicators, inplace=True)

# %%
numeric_features = list(indicators.values())

# %%
fig, ax = plt.subplots(figsize=(25, 10))
corr_matrix = incidents_df[numeric_features].corr('pearson')
sns.heatmap(corr_matrix, annot=True, ax=ax, mask=np.triu(corr_matrix), cmap='coolwarm')

# %%
fig, ax = plt.subplots(figsize=(25, 10))
corr_matrix = incidents_df[numeric_features].corr('spearman')
sns.heatmap(corr_matrix, annot=True, ax=ax, mask=np.triu(corr_matrix), cmap='coolwarm')

# %%
features_to_drop = [
    'log_min_age_mean_ratio', 
    'severity',
    'log_n_killed_mean_ratio',
    'log_n_injured_mean_ratio',
    'surprisal_n_killed',
    'suprisal_n_injured', # discarded to reduce dimensionality
    'log_n_males_mean_ratio',
    'log_n_participants_mean_ratio',
    'surprisal_n_participants',
    # surprisal_age_groups
]
numeric_features_subset = [feature for feature in numeric_features if feature not in features_to_drop]

# %%
fig, ax = plt.subplots(figsize=(20, 10))
corr_matrix = incidents_df[numeric_features_subset].corr('pearson')
sns.heatmap(corr_matrix, annot=True, ax=ax, mask=np.triu(corr_matrix), cmap='coolwarm')

# %%
fig, ax = plt.subplots(figsize=(20, 10))
corr_matrix = incidents_df[numeric_features_subset].corr('spearman')
sns.heatmap(corr_matrix, annot=True, ax=ax, mask=np.triu(corr_matrix), cmap='coolwarm')

# %%
print(incidents_df.shape[0])
print(incidents_df[list(indicators.values())].dropna().shape[0])

# %%
import json
with open('../data/indicators_names.json', 'w') as f:
    json.dump(list(indicators.values()), f)

# %%
origianal_features_minus_indicators = [feature for feature in dataset_original_columns if feature not in indicators.values()]
incidents_df[origianal_features_minus_indicators + list(indicators.values())].to_csv('../data/incidents_indicators.csv')
incidents_df[list(indicators.values())].to_csv('../data/indicators.csv')

# %% [markdown]
# ## Indicators semanthics
#
# TODO:mettere link geopy
#
# | Name | Description | Present in the original dataset |
# | :--: | :---------: | :-----------------------------: |
# | lat_proj | Latitude projection using Universal Transverse Mercator system | No |
# | long_proj | Longitude projection using Universal Transverse Mercator system | No |
# | location_imp | Location importance according to Geopy | No |
# | address_type_surprisal | Surprisal of the address type w.r.t the address types of incidents happened in the same semester of the same year and in the congressional district of the same state | No |
# | age_range | Difference between the maximum and the minimum age of the participants involved in the incident | No |
# | avg_age | Average age of the participants involved in the incident | Yes |
# | min_age_surprisal | Surprisal of the minimum age of the participants involved in the incident w.r.t the minimum age of the participants of incidents happened in the same semester of the same year and in the congressional district of the same state | No |
# | n_child_prop | Ratio between the number of child involved in the incident and number of people involved in the incident | No |
# | n_teen_prop | Ratio between the number of teen involved in the incident and number of people involved in the incident | No |
# | age_groups_surprisal | Surprisal of the number of child, teen and adult involved in the incident w.r.t the number of child, teen and adult involved in incidents happened in the same semester of the same year and in the congressional district of the same state | No |
# | n_killed_prop | Ratio between the number of people killed and number of people involved in the incident | No |
# | n_injured_prop | Ratio between the number of people injured and number of people involved in the incident | No |
# | n_unharmed_prop | Ratio between the number of people unharmed and number of people involved in the incident | No |
# | n_males_prop | Ratio between the number of males and the number of people involed in the incident | No |
# | n_males_surprisal | Surprisal of the number of males involved in the incident w.r.t the number of males involved in incidents happened in the same semester of the same year and in the congressional district of the same state | No |
# | characteristics_surprisal | Surprisal of the values of the incident characteristic tags extracted from the incident characteristics w.r.t the values of the tags for incidents happened in the same semester of the same year and in the congressional district of the same state | No |
# | n_arrested_prop | Ratio between the number of people arrested and the number of participants involved in the incident | No |
# | n_participants | Number of participants in the incident | Yes |
# | day_surprisal | Surprisal of the day of the month and of the month of the incident w.r.t the days of the month and the months of incidents happened in the same semester of the same year and in the congressional district of the same state | No |

# %%
# TODO: 
# distribuzioni variabili scelte, con e senza nan
# scatter semplici e in pca
# missing values, duplications
# controllare errori, tipo rapporti devono essere in 0,1, etc.
