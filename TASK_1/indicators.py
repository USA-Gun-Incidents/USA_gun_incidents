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
incidents_df

# %%
def compute_ratio_indicator(df, ext_df, gby, num, den, suffix, agg_fun):
    df = df.dropna(axis=0, subset=gby + [num, den])
    ext_df = ext_df.loc[df.index]
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix]).set_index(df.index)
    df[num+'_'+den+suffix+'_ratio'] = np.divide(df[num], df[den+suffix], out=np.zeros_like(df[num]), where=(df[den+suffix] != 0))
    return df[[num+'_'+den+suffix+'_ratio']]

ratios = pd.DataFrame(index=incidents_df.index)

# %%
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_males', 'n_males', '_tot_year_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_females', 'n_females', '_tot_year_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_males', 'n_participants', '_year_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df,incidents_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_males', '_tot_year_congdist', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_females', 'n_females', '_tot_year_congdist', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_participants', '_year_congdist', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'county'], 'n_males', 'n_males', '_tot_year_county', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'county'], 'n_females', 'n_females', '_tot_year_county', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'county'], 'n_males', 'n_participants', '_year_county', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_tot_year_city', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_tot_year_city', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_participants', '_year_city', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_mean_year_city', 'mean'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_median_year_city', 'median'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_mean_year_city', 'mean'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_median_year_city', 'median'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state'], 'n_males', 'n_males', '_tot_semester_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state'], 'n_females', 'n_females', '_tot_semester_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state'], 'n_males', 'n_participants', '_semester_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_males', 'n_males', '_tot_semester_congdist', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_females', 'n_females', '_tot_semester_congdist', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_males', 'n_participants', '_semester_congdist', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'county'], 'n_males', 'n_males', '_tot_semester_county', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'county'], 'n_females', 'n_females', '_tot_semester_county', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'county'], 'n_males', 'n_participants', '_semester_county', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'city'], 'n_males', 'n_males', '_tot_semester_city', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'city'], 'n_females', 'n_females', '_tot_semester_city', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'city'], 'n_males', 'n_participants', '_semester_city', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_killed', 'n_killed', '_tot_year_city', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_injured', 'n_injured', '_tot_year_city', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_tot_semester_congdist', 'sum')) # 2
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_injured', 'n_injured', '_tot_semester_congdist', 'sum')) # 2
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year'], 'n_unharmed', 'n_unharmed', '_mean_year', 'mean')) # 4
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester'], 'n_unharmed', 'n_unharmed', '_mean_semester', 'mean')) # 4
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_child', 'n_participants_child', '_tot_year_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_teen', 'n_participants_teen', '_tot_year_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants_adult', '_tot_year_state', 'sum'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants', '_tot_year_state', 'sum'))

# %%
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_killed', 'n_killed', '_mean_year_state', 'mean'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state'], 'n_killed', 'n_killed', '_median_year_state', 'median'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_mean_year_congdist', 'mean'))
ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_median_year_congdist', 'median'))

ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_participants', 'n_participants', '_mean_year_congdist', 'mean'))

# %%
ratios.tail(6)

# %%
def compute_simple_division(df, col_1, col_2):
    dummy = df.loc[(df[col_1].notna()) & (df[col_2].notna())]
    return dummy[col_1]/dummy[col_2]

ratios['n_killed_n_participants_ratio'] = compute_simple_division(incidents_df, 'n_killed', 'n_participants')
ratios['n_injured_n_participants_ratio'] = compute_simple_division(incidents_df, 'n_injured', 'n_participants')
ratios['n_unharmed_n_participants_ratio'] = compute_simple_division(incidents_df, 'n_unharmed', 'n_participants')

# %%
ratios

# %%
ratios_wrt_tot = []
ratios_wrt_center = []
for att in ratios.columns:
    if 'mean' in att or 'median' in att:
        ratios_wrt_center.append(att)
    else:
        ratios_wrt_tot.append(att)

# %%
def log_normalization(df, new_df, columns):
    for col in columns:
        c = (df[df[col]!=0][col].min())/100
        new_df['log_'+col] = np.log(df[col] + c) # 1% of the minimum value

# %%
log_ratios = pd.DataFrame(index=ratios.index)
log_normalization(ratios, log_ratios, ratios.columns)
log_ratios_wrt_tot = ['log_'+x for x in ratios_wrt_tot]
log_ratios_wrt_center = ['log_'+x for x in ratios_wrt_center]

# %%
incidents_df.tail(10)

# %%
log_ratios.tail(10)

# %%
ratios[ratios_wrt_tot].describe() # females quantiles are 0 (that's why they suggested to do it for males only)

# %%
ratios[ratios_wrt_center].describe()

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
sns.violinplot(data=log_ratios[log_ratios_wrt_tot],ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# La trasformazione logaritmica serve a rendere i dati meno sparsi, e in questo caso è utilizzata con il proposito opposto... 
#
# Non possiamo trasformare dei dati poco significanti in dati significanti in questo modo, attenzione e io consiglierei di non utilizzare il logaritmo per i valori tra [0,1]

# %%
fig, ax = plt.subplots(figsize=(15, 8))
sns.violinplot(data=log_ratios[log_ratios_wrt_center],ax=ax)
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
    bins=int(np.log(ratios.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
hist_box_plot(
    log_ratios,
    'log_n_males_n_males_mean_year_city_ratio',
    title='log_n_males_n_males_mean_year_city_ratio',
    bins=int(np.log(log_ratios.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
def compute_square_distance_indicator(df, ext_df, gby, minuend, subtrahend, suffix, agg_fun):

    df = df.dropna(axis=0, subset=gby + [minuend, subtrahend])
    ext_df = ext_df.loc[df.index]
    
    grouped_df = ext_df.groupby(gby)[subtrahend].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix]).set_index(df.index)
    df[minuend+'_'+subtrahend+suffix+'_SD'] = np.square((df[minuend]- df[subtrahend+suffix]))
    return df[[minuend+'_'+subtrahend+suffix+'_SD']]

square_distances = pd.DataFrame(index=incidents_df.index)

# %%
c_list = ['n_participants_child','n_participants_teen','n_participants_adult','n_males','n_females','n_killed','n_injured','n_arrested','n_unharmed', 'n_participants']
for l in c_list:
    square_distances = square_distances.join(compute_square_distance_indicator(incidents_df, incidents_df, ['year', 'state'], l, l, '_mean_year_state', 'mean'))
    square_distances = square_distances.join(compute_square_distance_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], l, l, '_mean_year_congdist', 'mean'))

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
    bins=int(np.log(square_distances.shape[0])), # Sturger's rule
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
dummy = incidents_df.dropna(axis=0, subset=['state', 'year', 'day_of_week'])

dummy.groupby(['state', 'year'])[['day_of_week']].value_counts().reset_index()

# %%
def compute_entropy_indicator(df, col_aggr, col_group, lab=''):
    df = df.dropna(axis=0, subset=col_aggr + col_group)

    a = incidents_df.groupby(col_aggr)[col_group].value_counts().reset_index()
    b = incidents_df.groupby(col_aggr[:-1])[col_aggr[-1]].value_counts().reset_index()
    a_b = a.merge(b, how='left', on=col_aggr, suffixes=['_occ', '_tot'])
    a_b['prob_occ'] = a_b['count_occ']/a_b['count_tot']
    a_b['entropy'] = np.log2(a_b['prob_occ'])*(-1)
    if lab == '':
        lab = 'entropy'
        for s in col_group:
            lab += '_' + s
        lab += '_fixed'
        for s in col_aggr:
            lab += '_' + s

    ret = df.merge(a_b, how='left', on=(col_aggr+col_group)).set_index(df.index).rename(columns={'entropy':lab})
    return ret[[lab]]
entropies = pd.DataFrame(index=ratios.index)

# %%
dummy_col = ['date','month','day','day_of_week','county','city','address_type','congressional_district','avg_age_participants','n_killed', 'party']
for col in dummy_col:
    if not col in ['state', 'year']:
        entropies = entropies.join(compute_entropy_indicator(incidents_df, ['state', 'year'], [col]))

entropies = entropies.join(compute_entropy_indicator(incidents_df, ['state', 'year'], ['firearm','suicide','death','house'], 'mix_col_1'))
entropies = entropies.join(compute_entropy_indicator(incidents_df, ['state', 'year'], ['firearm','air_gun','shots','aggression','suicide','injuries','death','road','illegal_holding','house','school','children','drugs','officers','organized','social_reasons','defensive','workplace','abduction','unintentional'], 'mix_col_2'))
entropies = entropies.join(compute_entropy_indicator(incidents_df, ['state', 'year'], ['school','children','drugs'], 'mix_col_3'))




# %%
entropies

# %%
hist_box_plot(
    entropies,
    'mix_col_3',
    title='mix_col_3',
    bins=int(np.log(entropies.shape[0])), # Sturger's rule
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
    'entropy_day_of_week_fixed_state_year',
    title='entropy_day_of_week_fixed_state_year',
    bins=int(np.log(entropies.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
hist_box_plot(
    entropies,
    'mix_col_2',
    title='mix_col_2',
    bins=int(np.log(entropies.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
from sklearn.neighbors import LocalOutlierFactor
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#col_ num = ['date','latitude','longitude', 'location_importance', 'participant_age1','participant1_child','participant1_teen','participant1_adult',participant1_male,participant1_female,min_age_participants,avg_age_participants,max_age_participants,n_participants_child,n_participants_teen,n_participants_adult,n_males,n_females,n_killed,n_injured,n_arrested,n_unharmed,n_participants,notes,incident_characteristics1,incident_characteristics2,firearm,air_gun,shots,aggression,suicide,injuries,death,road,illegal_holding,house,school,children,drugs,officers,organized,social_reasons,defensive,workplace,abduction,unintentional]
incidents_numeric = incidents_df.select_dtypes(include=numerics).dropna(axis=0).drop(columns=['poverty_perc','candidate_votes','total_votes','candidate_perc','population_state_2010'])
print(incidents_numeric.shape)

ground_truth = np.ones(incidents_numeric.shape[0], dtype=int)
#ground_truth[-n_outliers:] = -1

N_NEIGHBORS = 20
CONTAMINATION = 0.1
clf = LocalOutlierFactor(n_neighbors=N_NEIGHBORS, contamination=CONTAMINATION)

# %%
y_pred = clf.fit_predict(incidents_numeric)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_
X_scores

# %%
local_outlier_factors = pd.DataFrame(index=incidents_numeric.index, data=X_scores).rename(columns={0:'local_outlier_factor'})
local_outlier_factors['log_inv_local_outlier_factor'] = np.log2(local_outlier_factors['local_outlier_factor']*([-1]*len(local_outlier_factors['local_outlier_factor'])))
local_outlier_factors

# %%
hist_box_plot(
    local_outlier_factors,
    'local_outlier_factor',
    title='local_outlier_factor',
    bins=int(np.log(local_outlier_factors.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
hist_box_plot(
    local_outlier_factors,
    'log_inv_local_outlier_factor',
    title='log_inv_local_outlier_factor',
    bins=int(np.log(local_outlier_factors.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
#import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection


def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

plt.scatter(incidents_numeric['longitude'], incidents_numeric['latitude'], color="k", s=3.0, label="Data points")
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    incidents_numeric['longitude'],
    incidents_numeric['latitude'],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("tight")
#plt.xlim((-5, 5))
#plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("Local Outlier Factor (LOF)")
plt.show()

# %%
import matplotlib.pyplot as mplt
import plotly.express as px
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %%
incidents_numeric

# %%
pca = PCA()
X_pca = pca.fit_transform(incidents_numeric)

# %%
nrows=5
ncols=6
row=0
fig, axs = mplt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), sharex=True, sharey=True)
for i, col in enumerate(incidents_numeric.columns):
    if i != 0 and i % ncols == 0:
        row += 1
    axs[row][i % ncols].scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40, c=incidents_numeric[col])
    axs[row][i % ncols].set_title(col)
    axs[row][i % ncols].set_xlabel("1st eigenvector")
    axs[row][i % ncols].set_ylabel("2nd eigenvector")

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = X_pca[:, 0]
y = X_pca[:, 2]
z = X_pca[:, 1]

ax.set_xlabel("1st eigenvector")
ax.set_ylabel("3rd eigenvector")
ax.set_zlabel("2nd eigenvector")

ax.scatter(x, y, z)

fig = px.scatter_3d(x=x, y=y, z=z, labels={'x': '1st eigenvector', 'y': '3rd eigenvector', 'z': '2nd eigenvector'})
fig.show()


# %%
X_reconstructed = pca.inverse_transform(X_pca)
PCA_errors = pd.DataFrame(index=incidents_numeric.index)
PCA_errors['reconstruction_error'] = np.sum(np.square(X_pca-X_reconstructed), axis=1)

#incidents_df['pca_reconstruction_error'] = square_error


# %%
hist_box_plot(
    PCA_errors,
    'reconstruction_error',
    title='reconstruction_error',
    bins=int(np.log(ratios.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%

# %%

# %%


# %%
'''population_df = pd.read_csv('../data/external_data/population.csv')
population_df['n_males'] = population_df['male_child'] + population_df['male_teen'] + population_df['male_adult']
population_df['n_females'] = population_df['female_child'] + population_df['female_teen'] + population_df['female_adult']
population_df'''

# %%
#compute_ratio_indicator(incidents_df, ratios, population_df, ['year', 'state'], 'n_males', 'n_males', '_pop_year_state', 'sum')
#compute_ratio_indicator(incidents_df, ratios, population_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_males', '_pop_year_cong', 'sum')

# %%
'''pop_ratios = []
for att in ratios.columns:
    if 'pop' in att and 'ratio' in att:
        pop_ratios.append(att)

ratios.boxplot(
    column=pop_ratios,
    rot=90
)'''

# %%
'''incidents_df = incidents_df.sort_values(by=['date'])
incidents_df['days_from_last_incident_in_congdist'] = incidents_df.groupby(['congressional_district'])['date'].diff().dt.days
incidents_df['days_from_last_incident_in_congdist'].describe()'''

# %%
#ratios.boxplot(by='state', column='days_from_last_incident_in_congdist', figsize=(20, 10), rot=90)

# %%
'''x = pd.DataFrame(data={'a': [0, np.nan,], 'b':[0, np.nan]})
x['a'] / x['b']'''

# %%
#np.divide(x['a'], x['b'], out=np.zeros_like(x['a']), where=(x['b'] != 0))

# %%
#incidents_df.sample(5, random_state=1)

# %% [markdown]
# # Quali sono, circa, i migliori indici individuati:
# - uccisi, feriti ecc.. rispetto alla media, con norm. logaritmica
# - rapporto degli uccisi/totali o feriti/totali dell'incidente (magari sostituiti)
# - entropie pazzerelle (su tutti i tag o combinazioni di tag)
#

# %%
final_indicators = pd.DataFrame(index=ratios.index)
final_indicators['n_killed_n_participants_ratio'] = ratios['n_killed_n_participants_ratio']
final_indicators['n_unharmed_n_participants_ratio'] = ratios['n_unharmed_n_participants_ratio']

final_indicators['log_n_killed_n_killed_mean_year_state_ratio'] = log_ratios['log_n_killed_n_killed_mean_year_state_ratio']
final_indicators['log_n_participants_n_participants_mean_year_congdist_ratio'] = log_ratios['log_n_participants_n_participants_mean_year_congdist_ratio']

final_indicators['log_n_males_n_males_mean_year_congdist_SD'] = log_square_distances['log_n_males_n_males_mean_year_congdist_SD']
final_indicators['log_n_females_n_females_mean_year_congdist_SD'] = log_square_distances['log_n_females_n_females_mean_year_congdist_SD']

final_indicators['entropy_city_fixed_state_year'] = entropies['entropy_city_fixed_state_year']
final_indicators['mix_col_1'] = entropies['mix_col_1']
final_indicators['mix_col_2'] = entropies['mix_col_2']

DATA_FOLDER_PATH = '../data/'
final_indicators.to_csv(DATA_FOLDER_PATH +'incidents_cleaned_indicators.csv', index=False)

