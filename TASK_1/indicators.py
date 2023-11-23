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
incidents_df.describe()

# %%
def compute_ratio_indicator(df, ext_df, gby, num, den, suffix, agg_fun):
    df = df.dropna(axis=0, subset=gby + [num, den])
    ext_df = ext_df.loc[df.index]
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix]).set_index(df.index)
    df[num+'_'+den+suffix+'_ratio'] = np.divide(df[num], df[den+suffix], out=np.zeros_like(df[num]), where=(df[den+suffix] != 0))
    return df[[num+'_'+den+suffix+'_ratio']]


# %%
def compute_simple_division(df, col_1, col_2):
    dummy = df.loc[(df[col_1].notna()) & (df[col_2].notna())]
    return dummy[col_1]/dummy[col_2]


# %%
def log_normalization(df, new_df, columns):
    for col in columns:
        c = (df[df[col]!=0][col].min())/100
        new_df['log_'+col] = np.log(df[col] + c) # 1% of the minimum value

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
from sklearn.neighbors import LocalOutlierFactor
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#col_ num = ['date','latitude','longitude', 'location_importance', 'participant_age1','participant1_child','participant1_teen','participant1_adult',participant1_male,participant1_female,min_age_participants,avg_age_participants,max_age_participants,n_participants_child,n_participants_teen,n_participants_adult,n_males,n_females,n_killed,n_injured,n_arrested,n_unharmed,n_participants,notes,incident_characteristics1,incident_characteristics2,firearm,air_gun,shots,aggression,suicide,injuries,death,road,illegal_holding,house,school,children,drugs,officers,organized,social_reasons,defensive,workplace,abduction,unintentional]
incidents_numeric = incidents_df.select_dtypes(include=numerics).dropna(axis=0).drop(columns=['poverty_perc','candidate_votes','total_votes','candidate_perc','population_state_2010', 'congressional_district','state_house_district',	'state_senate_district'])
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
local_outlier_factors

# %%
dummy = incidents_df.join(local_outlier_factors).select_dtypes(include=numerics).dropna(axis=0).drop(columns=['poverty_perc','candidate_votes','total_votes','candidate_perc','population_state_2010', 'congressional_district','state_house_district',	'state_senate_district', 'semester'])
dummy[dummy['local_outlier_factor']<-1.8].sample(10)

# %%
hist_box_plot(
    local_outlier_factors,
    'local_outlier_factor',
    title='local_outlier_factor',
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
    s=500 * radius,
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
final_col = ['latitude', 'longitude', 'location_importance',
             'participant_age1',
            'avg_age_participants',
            'n_participants_child', 'n_participants_teen', 'n_participants_adult',
            'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested',
            'n_unharmed', 'n_participants']

# %%
def compute_simple_subtraction(df, col_1, col_2):
    dummy = df.loc[(df[col_1].notna()) & (df[col_2].notna())]
    return dummy[col_1] - dummy[col_2]

# %%
indicators = pd.DataFrame(index=incidents_df.index, data=incidents_df[['latitude', 'longitude', 'location_importance',
            'avg_age_participants',
            'n_participants']].copy(deep=True))

# %%
indicators['age_range'] = compute_simple_subtraction(incidents_df, 'max_age_participants', 'min_age_participants')

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
for col in ['n_participants_child', 'n_participants_teen', 'n_participants_adult',
            'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested',
            'n_unharmed']:
    indicators[col +'_prop'] = compute_simple_division(incidents_df, col, 'n_participants')

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
ratios = pd.DataFrame(index=incidents_df.index)
log_ratios = pd.DataFrame(index=incidents_df.index)

coppie = [['n_males', 'n_males'], ['n_killed', 'n_killed'], ['n_injured', 'n_injured'], ['avg_age_participants', 'avg_age_participants'], ['n_participants_adult', 'n_participants']]
for cop in coppie:
    ratios = ratios.join(compute_ratio_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], cop[0], cop[1], '_mean_semest_congd', 'mean'))
log_normalization(ratios, log_ratios, ratios.columns)

# %%
log_ratios

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=log_ratios,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
indicators = indicators.join(log_ratios[['log_n_males_n_males_mean_semest_congd_ratio','log_n_killed_n_killed_mean_semest_congd_ratio','log_n_injured_n_injured_mean_semest_congd_ratio']])

# %%
indicators

# %%
dist = pd.DataFrame(index=incidents_df.index)
log_dist = pd.DataFrame(index=incidents_df.index)

coppie = [['n_males', 'n_males'], ['n_killed', 'n_killed'], ['n_injured', 'n_injured'], ['avg_age_participants', 'avg_age_participants']]
for cop in coppie:
    dist = dist.join(compute_square_distance_indicator(incidents_df, incidents_df, ['year', 'semester', 'state', 'congressional_district'], cop[0], cop[1], '_tot_year_state', 'mean'))
log_normalization(dist, log_dist, dist.columns)

# %%
log_dist

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=log_dist,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
indicators = indicators.join(log_dist['log_avg_age_participants_avg_age_participants_tot_year_state_SD'])

# %%
entropies = pd.DataFrame(index=incidents_df.index)
dummy_col = ['month', 'day', 'state', 'county', 'city', 'address_type', 'congressional_district', 'n_participants_child', 'n_participants_teen', 'n_participants_adult'
             , 'min_age_participants', 'avg_age_participants', 'max_age_participants']

for col in dummy_col:
    if not col in ['year', 'semester', 'state', 'congressional_district']:
        entropies = entropies.join(compute_entropy_indicator(incidents_df, ['year', 'semester', 'state', 'congressional_district'], [col]))

tag = [ 'firearm', 'air_gun', 'shots',
       'aggression', 'suicide', 'injuries', 'death', 'road', 'illegal_holding',
       'house', 'school', 'children', 'drugs', 'officers', 'organized',
       'social_reasons', 'defensive', 'workplace', 'abduction',
       'unintentional']


# %%
entropies

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=entropies,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
indicators = indicators.join(entropies['entropy_avg_age_participants_fixed_year_semester_state_congressional_district'])
indicators = indicators.join(entropies['entropy_city_fixed_year_semester_state_congressional_district'])
indicators = indicators.join(entropies['entropy_address_type_fixed_year_semester_state_congressional_district'])
indicators = indicators.join(entropies['entropy_n_participants_adult_fixed_year_semester_state_congressional_district'])

# %%
hist_box_plot(
    entropies,
    'entropy_city_fixed_year_semester_state_congressional_district',
    title='entropy_city_fixed_year_semester_state_congressional_district',
    bins=int(np.log(ratios.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
tag_e = compute_entropy_indicator(incidents_df, ['year', 'semester', 'state', 'congressional_district'], tag, 'mix_tag')
indicators = indicators.join(tag_e)

# %%
hist_box_plot(
    tag_e,
    'mix_tag',
    title='mix_tag',
    bins=int(np.log(ratios.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
hist_box_plot(
    incidents_df,
    'location_importance',
    title='location_importance',
    bins=int(np.log(ratios.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
incidents_df.loc[(incidents_df['avg_age_participants']==0)]

# %%
alpha = 0.8
severity = ((alpha * incidents_df['n_killed'] + (1 - alpha) * incidents_df['n_injured']) /
            (incidents_df['n_participants'] - incidents_df['n_unharmed']))

indicators['severity'] = severity

# %%
indicators['severity'].replace([np.inf, -np.inf], 0, inplace=True) # if all partecipants are unharmed the severity is 0

indicators['severity'].min()

# %%
indicators.head(10)

# %%
indicators.dropna().describe()

# %%
sns.heatmap(indicators.corr(), vmin=-1, vmax=1)

# %%
col_to_drop = ['n_participants_adult_prop',  'n_females_prop', 'log_n_killed_n_killed_mean_semest_congd_ratio', 'log_n_injured_n_injured_mean_semest_congd_ratio', 'n_males_prop']
final_indicators = indicators.drop(columns=col_to_drop)

# %%
sns.heatmap(final_indicators.corr(), vmin=-1, vmax=1)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=final_indicators,ax=ax)
plt.xticks(rotation=90, ha='right')

# %%
a = {}
for c in final_indicators.columns:
    a[c]='popo'

# %%
a

# %%
final_indicators.rename(columns={'n_males_prop': 'n_males_pr',
 'n_killed_prop': 'n_killed_pr',
 'n_injured_prop': 'n_injured_pr',
 'n_arrested_prop': 'n_arrested_pr',
 'n_unharmed_prop': 'n_unharmed_pr',
 'log_n_males_n_males_tot_semest_congd_ratio': 'log_males_mean_ratio',
 'log_avg_age_participants_avg_age_participants_tot_year_state_SD': 'log_avg_age_mean_SD',
 'entropy_avg_age_participants_fixed_year_semester_state_congressional_district': 'avg_age_entropy',
 'entropy_city_fixed_year_semester_state_congressional_district': 'city_entropy',
 'entropy_address_type_fixed_year_semester_state_congressional_district': 'address_entropy',
 'entropy_n_participants_adult_fixed_year_semester_state_congressional_district': 'n_adults_entropy',
 'mix_tag': 'tags_entropy'}, inplace=True)

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler_obj = MinMaxScaler()
normalized_indicators = pd.DataFrame(data=scaler_obj.fit_transform(final_indicators.values), columns=final_indicators.columns)

# %%
incidents_df[['avg_age_participants', 'max_age_participants', 'min_age_participants', 'n_participants_child', 'n_participants_teen', 'semester',
              'congressional_district', 'year']].describe()

# %%
normalized_indicators.describe()

# %%
normalized_indicators.sample(2)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=normalized_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
DATA_FOLDER_PATH = '../data/'
normalized_indicators.to_csv(DATA_FOLDER_PATH +'incidents_cleaned_indicators.csv')
final_indicators.to_csv(DATA_FOLDER_PATH +'incidents_cleaned_indicators_not_norm.csv')

# %%
normalized_indicators.describe()

# %%
import matplotlib.pyplot as mplt
import plotly.express as px
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %%
normalized_indicators_notna = normalized_indicators.dropna()
pca = PCA()
X_pca = pca.fit_transform(normalized_indicators_notna)
pca_df = pd.DataFrame(index=incidents_df.index)

# %%
nrows=4
ncols=6
row=0
fig, axs = mplt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), sharex=True, sharey=True)
for i, col in enumerate(normalized_indicators_notna.columns):
    if i != 0 and i % ncols == 0:
        row += 1
    axs[row][i % ncols].scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40, c=normalized_indicators_notna[col])
    axs[row][i % ncols].set_title(col)
    axs[row][i % ncols].set_xlabel("1st eigenvector")
    axs[row][i % ncols].set_ylabel("2nd eigenvector")

# %%
nrows=4
ncols=6
row=0
fig, axs = mplt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), sharex=True, sharey=True)
for i, col in enumerate(normalized_indicators_notna.columns):
    if i != 0 and i % ncols == 0:
        row += 1
    axs[row][i % ncols].scatter(X_pca[:, 18], X_pca[:, 19], edgecolor='k', s=40, c=normalized_indicators_notna[col])
    axs[row][i % ncols].set_title(col)
    axs[row][i % ncols].set_xlabel("19th eigenvector")
    axs[row][i % ncols].set_ylabel("20th eigenvector")

# %%
x = X_pca[:, 0]
y = X_pca[:, 2]
z = X_pca[:, 1]
fig = px.scatter_3d(x=x, y=y, z=z, labels={'x': '1st eigenvector', 'y': '3rd eigenvector', 'z': '2nd eigenvector'})
fig.show()

# %%
exp_var_pca = pca.explained_variance_ratio_
diff_var = []

for i, var in enumerate(exp_var_pca[:-1]):
    diff_var.append( var-exp_var_pca[i+1])


xtick = []
gap = 0
for i, var in enumerate(diff_var):
    xtick.append(i+gap)
    if i != 0 and diff_var[i-1] <= var:
        gap += 0.5
        if gap == 0.5:
            plt.axvline(x = i+gap+0.25, color = 'green', linestyle = '-.', alpha=0.5, label='possible cut')
        else:
             plt.axvline(x = i+gap+0.25, color = 'green', linestyle = '-.', alpha=0.5)
    

#xtick = [0,1,2,3,4,5.5,6.5,7.5,8.5,9.5,10.5,12,13,14,15,16,17,18,19,20]
#diff_var = list(zip(xtick, diff_var))
xtick.append(23)

plt.bar(xtick, exp_var_pca, align='center')
plt.plot(xtick[1:], diff_var, label='difference from prevoius variance', color='orange')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component')
plt.title('Explained variance by principal component')
plt.xticks(xtick, range(20))
plt.legend();


# %%
def get_reconstruction_error(x_pca, x_orig, pca, n_comp):
    dummy = np.matmul(x_pca[:,:n_comp], pca.components_[:n_comp,:]) + pca.mean_
    return pd.DataFrame(index=x_orig.index, data=np.sum((dummy - x_orig.values)**2, axis=1))


# %%
pca_col = ['1st_comp',
 '2nd_comp',
 '3rd_comp',
 '4th_comp',
 '5th_comp',
 '6th_comp',
 '7th_comp',
 '8th_comp',
 '9th_comp',
 '10th_comp',
 '11th_comp',
 '12th_comp',
 '13th_comp',
 '14th_comp',
 '15th_comp',
 '16th_comp',
 '17th_comp',
 '18th_comp',
 '19th_comp',
 '20th_comp']


# %%
pca_indicators = pd.DataFrame(index=normalized_indicators_notna.index, data=X_pca, columns=pca_col)

# %%
pca_indicators['PCA_rec_error_5C'] = get_reconstruction_error(X_pca, normalized_indicators_notna, pca, 5)
pca_indicators['PCA_rec_error_11C'] = get_reconstruction_error(X_pca, normalized_indicators_notna, pca, 11)
pca_indicators['PCA_rec_error_20C'] = get_reconstruction_error(X_pca, normalized_indicators_notna, pca, 20)

# %%
pca_indicators.sample(3)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=pca_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
pca_normalized_indicators = pd.DataFrame(data=scaler_obj.fit_transform(pca_indicators.values), columns=pca_indicators.columns)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=pca_normalized_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
hist_box_plot(
    pca_normalized_indicators,
    'PCA_rec_error_5C',
    title='PCA_rec_error_5C',
    bins=int(np.log(ratios.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
hist_box_plot(
    pca_normalized_indicators,
    'PCA_rec_error_11C',
    title='PCA_rec_error_11C',
    bins=int(np.log(ratios.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
pca_normalized_indicators.to_csv(DATA_FOLDER_PATH +'incidents_cleaned_indicators_PCA.csv')
pca_indicators.to_csv(DATA_FOLDER_PATH +'incidents_cleaned_indicators_PCA_not_norm.csv')

# %% [markdown]
# ['date', 'date_original', 'year', 'month', 'day', 'day_of_week', 'state',
#        'address', 'latitude', 'longitude', 'county', 'city',
#        'location_importance', 'address_type', 'congressional_district',
#        'state_house_district', 'state_senate_district', 'px_code',
#        'participant_age1', 'participant1_child', 'participant1_teen',
#        'participant1_adult', 'participant1_male', 'participant1_female',
#        'min_age_participants', 'avg_age_participants', 'max_age_participants',
#        'n_participants_child', 'n_participants_teen', 'n_participants_adult',
#        'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested',
#        'n_unharmed', 'n_participants', 'notes', 'incident_characteristics1',
#        'incident_characteristics2', 'firearm', 'air_gun', 'shots',
#        'aggression', 'suicide', 'injuries', 'death', 'road', 'illegal_holding',
#        'house', 'school', 'children', 'drugs', 'officers', 'organized',
#        'social_reasons', 'defensive', 'workplace', 'abduction',
#        'unintentional', 'poverty_perc', 'party', 'candidate_votes',
#        'total_votes', 'candidate_perc', 'population_state_2010', 'semester']


