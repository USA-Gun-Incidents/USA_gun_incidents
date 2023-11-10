# %%
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from plot_utils import *
# %matplotlib inline

# %%
incidents_df = pd.read_csv('../data/incidents_cleaned.csv')
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')

# %%
incidents_df['semester'] = (incidents_df['date'].dt.month // 7) + 1

# %%
incidents_df['city'] = incidents_df['city'].fillna('UKN') # to treat all points without city as belonging to the same fake city

# %%
def compute_ratio_indicator(df, ext_df, gby, num, den, suffix, agg_fun):
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    df[num+'_'+den+suffix+'_ratio'] = np.divide(df[num], df[den+suffix], out=np.zeros_like(df[num]), where=(df[den+suffix] != 0))
    df.drop(columns=[den+suffix], inplace=True)
    return df

incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_males', 'n_males', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_females', 'n_females', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_males', 'n_participants', '_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_males', '_tot_year_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_females', 'n_females', '_tot_year_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_participants', '_year_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'county'], 'n_males', 'n_males', '_tot_year_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'county'], 'n_females', 'n_females', '_tot_year_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'county'], 'n_males', 'n_participants', '_year_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_tot_year_city', 'sum') # 1
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_tot_year_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_participants', '_year_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_mean_year_city', 'mean')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_median_year_city', 'median')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_mean_year_city', 'mean')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_median_year_city', 'median')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state'], 'n_males', 'n_males', '_tot_semester_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state'], 'n_females', 'n_females', '_tot_semester_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state'], 'n_males', 'n_participants', '_semester_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_males', 'n_males', '_tot_semester_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_females', 'n_females', '_tot_semester_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_males', 'n_participants', '_semester_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'county'], 'n_males', 'n_males', '_tot_semester_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'county'], 'n_females', 'n_females', '_tot_semester_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'county'], 'n_males', 'n_participants', '_semester_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'city'], 'n_males', 'n_males', '_tot_semester_city', 'sum') # 1
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'city'], 'n_females', 'n_females', '_tot_semester_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'city'], 'n_males', 'n_participants', '_semester_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_killed', 'n_killed', '_tot_year_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_injured', 'n_injured', '_tot_year_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_tot_semester_congdist', 'sum') # 2
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_injured', 'n_injured', '_tot_semester_congdist', 'sum') # 2
incidents_df['n_killed_n_participants_ratio'] = incidents_df['n_killed'] / incidents_df['n_participants'] # 3
incidents_df['n_injured_n_participants_ratio'] = incidents_df['n_injured'] / incidents_df['n_participants']
incidents_df['n_unharmed_n_participants_ratio'] = incidents_df['n_unharmed'] / incidents_df['n_participants']
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year'], 'n_unharmed', 'n_unharmed', '_mean_year', 'mean') # 4
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester'], 'n_unharmed', 'n_unharmed', '_mean_semester', 'mean') # 4
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_child', 'n_participants_child', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_teen', 'n_participants_teen', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants_adult', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants', '_tot_year_state', 'sum')


# %%
ratios_wrt_tot = []
ratios_wrt_center = []
for att in incidents_df.columns:
    if 'ratio' in att:
        if 'mean' in att or 'median' in att:
            ratios_wrt_center.append(att)
        else:
            ratios_wrt_tot.append(att)

# %%
incidents_df[ratios_wrt_tot].describe() # females quantiles are 0 (that's why they suggested to do it for males only)

# %%
incidents_df[ratios_wrt_center].describe()

# %%
incidents_df.boxplot(
    column=ratios_wrt_tot,
    rot=90,
    figsize=(20, 10)
)

# %%
log_ratio_wrt_tot = ['log_'+col for col in ratios_wrt_tot]
for col in ratios_wrt_tot:
    c = (incidents_df[incidents_df[col]!=0][col].min())/100
    incidents_df['log_'+col] = np.log(incidents_df[col] + c) # 1% of the minimum value
incidents_df.boxplot(
    column=log_ratio_wrt_tot,
    rot=90,
    figsize=(20, 10)
)

# %%
incidents_df.boxplot(
    column=ratios_wrt_center,
    rot=90
)

# %%
hist_box_plot(
    incidents_df,
    'n_males_n_males_tot_year_city_ratio',
    title='n_males_n_males_tot_year_city_ratio',
    bins=int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
hist_box_plot(
    incidents_df,
    'n_killed_n_participants_ratio',
    title='n_killed_n_partecipants_ratio',
    bins=int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%

hist_box_plot(
    incidents_df,
    'n_injured_n_participants_ratio',
    title='n_killed_n_partecipants_ratio',
    bins=int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
hist_box_plot(
    incidents_df,
    'log_n_males_n_males_tot_year_city_ratio',
    title='log_n_males_n_males_tot_year_city_ratio',
    bins=20,#int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
population_df = pd.read_csv('../data/external_data/population.csv')
population_df['n_males'] = population_df['male_child'] + population_df['male_teen'] + population_df['male_adult']
population_df['n_females'] = population_df['female_child'] + population_df['female_teen'] + population_df['female_adult']
population_df

# %%
incidents_df = compute_ratio_indicator(incidents_df, population_df, ['year', 'state'], 'n_males', 'n_males', '_pop_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, population_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_males', '_pop_year_cong', 'sum')

# %%
pop_ratios = []
for att in incidents_df.columns:
    if 'pop' in att and 'ratio' in att:
        pop_ratios.append(att)

incidents_df.boxplot(
    column=pop_ratios,
    rot=90
)

# %%
incidents_df = incidents_df.sort_values(by=['date'])
incidents_df['days_from_last_incident_in_congdist'] = incidents_df.groupby(['congressional_district'])['date'].diff().dt.days
incidents_df['days_from_last_incident_in_congdist'].describe()

# %%
incidents_df.boxplot(by='state', column='days_from_last_incident_in_congdist', figsize=(20, 10), rot=90)

# %%
x = pd.DataFrame(data={'a': [0, np.nan,], 'b':[0, np.nan]})
x['a'] / x['b']

# %%
np.divide(x['a'], x['b'], out=np.zeros_like(x['a']), where=(x['b'] != 0))

# %%
# TODO:
# - entropia?
# commentare per bene, organizzare, fare matrice correlazione


