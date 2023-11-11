# -*- coding: utf-8 -*-
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

# %%
incidents_df = pd.read_csv('../data/incidents_cleaned.csv')
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')

# %%
incidents_df['semester'] = (incidents_df['date'].dt.month // 7) + 1

# %%
incidents_df['city'] = incidents_df['city'].fillna('UKN') # to treat all points without city as belonging to the same fake city

# %%
def compute_ratio_indicator(df, new_df, ext_df, gby, num, den, suffix, agg_fun):
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    new_df[num+'_'+den+suffix+'_ratio'] = np.divide(df[num], df[den+suffix], out=np.zeros_like(df[num]), where=(df[den+suffix] != 0))
    #df.drop(columns=[den+suffix], inplace=True)
    #return df

ratios = pd.DataFrame(index=incidents_df.index)

# %%
compute_ratio_indicator(incidents_df, ratios, incidents_df, ['year', 'state'], 'n_males', 'n_males', '_tot_year_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state'], 'n_females', 'n_females', '_tot_year_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state'], 'n_males', 'n_participants', '_year_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_males', '_tot_year_congdist', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'congressional_district'], 'n_females', 'n_females', '_tot_year_congdist', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_participants', '_year_congdist', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'county'], 'n_males', 'n_males', '_tot_year_county', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'county'], 'n_females', 'n_females', '_tot_year_county', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'county'], 'n_males', 'n_participants', '_year_county', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_tot_year_city', 'sum') # 1
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_tot_year_city', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_males', 'n_participants', '_year_city', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_mean_year_city', 'mean')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_median_year_city', 'median')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_mean_year_city', 'mean')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_median_year_city', 'median')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state'], 'n_males', 'n_males', '_tot_semester_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state'], 'n_females', 'n_females', '_tot_semester_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state'], 'n_males', 'n_participants', '_semester_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_males', 'n_males', '_tot_semester_congdist', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_females', 'n_females', '_tot_semester_congdist', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_males', 'n_participants', '_semester_congdist', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'county'], 'n_males', 'n_males', '_tot_semester_county', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'county'], 'n_females', 'n_females', '_tot_semester_county', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'county'], 'n_males', 'n_participants', '_semester_county', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'city'], 'n_males', 'n_males', '_tot_semester_city', 'sum') # 1
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'city'], 'n_females', 'n_females', '_tot_semester_city', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'city'], 'n_males', 'n_participants', '_semester_city', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_killed', 'n_killed', '_tot_year_city', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state', 'city'], 'n_injured', 'n_injured', '_tot_year_city', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_tot_semester_congdist', 'sum') # 2
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_injured', 'n_injured', '_tot_semester_congdist', 'sum') # 2
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year'], 'n_unharmed', 'n_unharmed', '_mean_year', 'mean') # 4
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'semester'], 'n_unharmed', 'n_unharmed', '_mean_semester', 'mean') # 4
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state'], 'n_participants_child', 'n_participants_child', '_tot_year_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state'], 'n_participants_teen', 'n_participants_teen', '_tot_year_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants_adult', '_tot_year_state', 'sum')
compute_ratio_indicator(incidents_df, ratios,  incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants', '_tot_year_state', 'sum')

# %%
compute_ratio_indicator(incidents_df, ratios, incidents_df, ['year', 'state'], 'n_killed', 'n_killed', '_mean_year_state', 'mean')
compute_ratio_indicator(incidents_df, ratios, incidents_df, ['year', 'state'], 'n_killed', 'n_killed', '_median_year_state', 'median')
compute_ratio_indicator(incidents_df, ratios, incidents_df, ['year', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_mean_year_congdist', 'mean')
compute_ratio_indicator(incidents_df, ratios, incidents_df, ['year', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_median_year_congdist', 'median')

# %%
ratios['n_killed_n_participants_ratio'] = incidents_df['n_killed'] / incidents_df['n_participants'] # 3
ratios['n_injured_n_participants_ratio'] = incidents_df['n_injured'] / incidents_df['n_participants']
ratios['n_unharmed_n_participants_ratio'] = incidents_df['n_unharmed'] / incidents_df['n_participants']

# %%
ratios.sample(10, random_state=1)

# %%
'''
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
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants', '_tot_year_state', 'sum')'''

# %%
ratios_wrt_tot = []
ratios_wrt_center = []
for att in ratios.columns:
    if 'mean' in att or 'median' in att:
        ratios_wrt_center.append(att)
    else:
        ratios_wrt_tot.append(att)

# %%
ratios[ratios_wrt_tot].describe() # females quantiles are 0 (that's why they suggested to do it for males only)

# %%
ratios[ratios_wrt_center].describe()

# %%
ratios.boxplot(
    column=ratios_wrt_tot,
    rot=90,
    figsize=(20, 10)
);

# %%
log_ratio_wrt_tot = ['log_'+col for col in ratios_wrt_tot]
log_ratio_wrt_center = ['log_'+col for col in ratios_wrt_center]
log_ratios = pd.DataFrame(index=ratios.index)
for col in ratios.columns:
    c = (ratios[ratios[col]!=0][col].min())/100
    log_ratios['log_'+col] = np.log(ratios[col] + c) # 1% of the minimum value
log_ratios.boxplot(
    column=log_ratio_wrt_tot,
    rot=90,
    figsize=(20, 10)
);

# %%
fig, ax = plt.subplots(figsize=(25, 10))
sns.violinplot(data=ratios[ratios_wrt_tot],ax=ax)
plt.xticks(rotation=90, ha='right');


# %%
fig, ax = plt.subplots()
sns.violinplot(data=ratios[ratios_wrt_center],ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
fig, ax = plt.subplots(figsize=(25, 10))
sns.violinplot(data=log_ratios[log_ratio_wrt_tot],ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# La trasformazione logaritmica serve a rendere i dati meno sparsi, e in questo caso è utilizzata con il proposito opposto... 
#
# Non possiamo trasformare dei dati poco significanti in dati significanti in questo modo, attenzione e io consiglierei di non utilizzare il logaritmo per i valori tra [0,1]

# %%
fig, ax = plt.subplots(figsize=(15, 10))
sns.violinplot(data=log_ratios[log_ratio_wrt_center],ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
ax = log_ratios['log_n_killed_n_killed_mean_year_state_ratio'].plot.kde()
log_ratios['log_n_killed_n_killed_median_year_state_ratio'].plot.kde(ax=ax)
plt.legend();

# %%
ax = log_ratios['log_n_unharmed_n_unharmed_mean_year_ratio'].plot.kde()
log_ratios['log_n_unharmed_n_unharmed_mean_semester_ratio'].plot.kde(ax=ax)
plt.legend();

# %% [markdown]
# Da una prima analisi, i rapporti con la media sembrano più rappresentativi di quelli con la mediana, e in generale molto più significativi di quelli con il totale, oltre che più logicamente sensati!!

# %%
hist_box_plot(
    ratios,
    'n_males_n_males_tot_year_city_ratio',
    title='n_males_n_males_tot_year_city_ratio',
    bins=int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
hist_box_plot(
    log_ratios,
    'log_n_males_n_males_mean_year_city_ratio',
    title='log_n_males_n_males_mean_year_city_ratio',
    bins=int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)


# %%
def compute_square_distance_indicator(df, new_df, ext_df, gby, minuend, subtrahend, suffix, agg_fun):
    grouped_df = ext_df.groupby(gby)[subtrahend].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    new_df[minuend+'_'+subtrahend+suffix+'_SD'] = np.square((df[minuend]- df[subtrahend+suffix]))#, out=np.zeros_like(df[num]), where=(df[den+suffix] != 0)
    #df.drop(columns=[den+suffix], inplace=True)
    #return df

def log_normalization(df, new_df, columns):
    for col in columns:
        c = (df[df[col]!=0][col].min())/100
        new_df['log_'+col] = np.log(df[col] + c) # 1% of the minimum value

square_distances = pd.DataFrame(index=incidents_df.index)

# %%

c_list = ['n_participants_child','n_participants_teen','n_participants_adult','n_males','n_females','n_killed','n_injured','n_arrested','n_unharmed', 'n_participants']
for l in c_list:
    compute_square_distance_indicator(incidents_df, square_distances, incidents_df, ['year', 'state'], l, l, '_mean_year_state', 'mean')
    compute_square_distance_indicator(incidents_df, square_distances, incidents_df, ['year', 'state', 'congressional_district'], l, l, '_mean_year_congdist', 'mean')


# %%
square_distances.sample(5)

# %%
log_square_distances = pd.DataFrame(index=square_distances.index)
log_normalization(square_distances, log_square_distances, square_distances.columns)

# %%
hist_box_plot(
    square_distances,
    'n_killed_n_killed_mean_year_state_SD',
    title='n_killed_n_killed_mean_year_state_SD',
    bins=int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)
square_distances.sample(10, random_state=1)

# %%
fig, ax = plt.subplots(figsize=(25, 10))
sns.violinplot(data=square_distances,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
fig, ax = plt.subplots(figsize=(25, 10))
sns.violinplot(data=log_square_distances,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
def compute_entropy_indicator(df, new_df, col_aggr, col_group, lab=''):
    a = incidents_df.groupby(col_aggr)[col_group].value_counts().reset_index()
    b = incidents_df.groupby(col_aggr[:-1])[col_aggr[-1]].value_counts().reset_index()
    a_b = a.merge(b, how='left', on=col_aggr, suffixes=['_occ', '_tot'])
    a_b['prob_occ'] = a_b['count_occ']/a_b['count_tot']
    a_b['entropy'] = np.log2(a_b['prob_occ'])*([-1]*len(a_b['prob_occ']))

    if lab == '':
        lab = 'entropy'
        for s in col_group:
            lab += '_' + s
        lab += '_fixed'
        for s in col_aggr:
            lab += '_' + s

    new_df[lab] = df.merge(a_b, how='left', on=(col_aggr+col_group))['entropy']

# %%
entropies = pd.DataFrame(index=ratios.index)
dummy_col = ['date','month','day','day_of_week','county','city','address_type','congressional_district','avg_age_participants','n_killed', 'party']
for col in dummy_col:
    if not col in ['state', 'year']:
        compute_entropy_indicator(incidents_df, entropies, ['state', 'year'], [col])
compute_entropy_indicator(incidents_df, entropies, ['state', 'year'], ['firearm','suicide','death','house'], 'mix_col_1')
compute_entropy_indicator(incidents_df, entropies, ['state', 'year'], ['firearm','air_gun','shots','aggression','suicide','injuries','death','road','illegal_holding','house','school','children','drugs','officers','organized','social_reasons','defensive','workplace','abduction','unintentional'], 'mix_col_2')

# %%
compute_entropy_indicator(incidents_df, entropies, ['state', 'year'], ['school','children','drugs'], 'mix_col_3')

# %%
entropies

# %%
hist_box_plot(
    entropies,
    'mix_col_3',
    title='mix_col_3',
    bins=int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
fig, ax = plt.subplots(figsize=(30, 10))
sns.violinplot(data=entropies,ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# INCREDIBLE notare che il mix 3 ha poca entropia!!! (roba grossa?)

# %%
hist_box_plot(
    entropies,
    'mix_col_2',
    title='mix_col_2',
    bins=int(np.log(incidents_df.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
population_df = pd.read_csv('../data/external_data/population.csv')
population_df['n_males'] = population_df['male_child'] + population_df['male_teen'] + population_df['male_adult']
population_df['n_females'] = population_df['female_child'] + population_df['female_teen'] + population_df['female_adult']
population_df

# %%
compute_ratio_indicator(incidents_df, ratios, population_df, ['year', 'state'], 'n_males', 'n_males', '_pop_year_state', 'sum')
compute_ratio_indicator(incidents_df, ratios, population_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_males', '_pop_year_cong', 'sum')

# %%
pop_ratios = []
for att in ratios.columns:
    if 'pop' in att and 'ratio' in att:
        pop_ratios.append(att)

ratios.boxplot(
    column=pop_ratios,
    rot=90
)

# %%
incidents_df = incidents_df.sort_values(by=['date'])
incidents_df['days_from_last_incident_in_congdist'] = incidents_df.groupby(['congressional_district'])['date'].diff().dt.days
incidents_df['days_from_last_incident_in_congdist'].describe()

# %%
ratios.boxplot(by='state', column='days_from_last_incident_in_congdist', figsize=(20, 10), rot=90)

# %%
x = pd.DataFrame(data={'a': [0, np.nan,], 'b':[0, np.nan]})
x['a'] / x['b']

# %%
np.divide(x['a'], x['b'], out=np.zeros_like(x['a']), where=(x['b'] != 0))

# %%
incidents_df.sample(5, random_state=1)

# %% [markdown]
# # Quali sono, circa, i migliori indici individuati:
# - uccisi, feriti ecc.. rispetto alla media, con norm. logaritmica
# - rapporto degli uccisi/totali o feriti/totali dell'incidente (magari sostituiti)
# - entropie pazzerelle (su tutti i tag o combinazioni di tag)
#

# %%



