# %% [markdown]
# # Time Series Analysis

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import zlib
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from tslearn.piecewise import PiecewiseAggregateApproximation
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
from scipy.spatial.distance import euclidean, cityblock
from plot_utils import sankey_plot
from matrixprofile import *
from matrixprofile.discords import discords

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)
incidents_df.drop_duplicates(inplace=True)

# drop rows where year is outside of 2014-2017
incidents_df = incidents_df[incidents_df['year'].between(2014, 2017)]

# %%
incidents_df.head(2)

# %%
incidents_df['year'].unique()

# %%
pd.to_datetime('2014-1-1', format='%Y-%m-%d').day_name()

# %%
pd.to_datetime('2014-1-6', format='%Y-%m-%d').day_name()

# %%
pd.to_datetime('2017-12-31', format='%Y-%m-%d').day_name()

# %% [markdown]
# Weeks are numbered from 0. The first week has 2 days less than the others.
# 
# 6th January 2014 is Monday and belongs to week 1 (the second).
# 
# 31th December 2017 is Sunday and belongs to week 208 (last one).

# %%
# Add columns week number, start from 0 for the first week of 2014
incidents_df['week'] = (incidents_df['date'] - pd.to_datetime('2013-12-30')).dt.days // 7

# %%
# number of weeks in the dataset
incidents_df['week'].max()

# %%
print('Number of weeks in the dataset: ', incidents_df['week'].max()+1)
print('Number of weeks of 7 days expected: ', ((365*4+1) - 5 ) / 7) # ok :)

# %%
incidents_df['week'].unique().shape # all weeks are present

# %%
# gruop by week and count incidents
plt.figure(figsize=(20, 5))
plt.bar(
    incidents_df.groupby('week').size().index,
    incidents_df.groupby('week').size().values
)
plt.title('Number of incidents per week');

# %% [markdown]
# Group incidents by City and Week

# %%
# group by wee, city and state
incidents_df.groupby(['week', 'city', 'state']).count()

# %% [markdown]
# We consider only cities with a number of weeks with incidents greater than 15% of the total number of the weeks of the 4 years.

# %%
0.15 * 209 # consider only cities with incidents in more than 30 weeks

# %%
incidents_df.groupby(['city', 'state'])['week'].count() # 10200 distinct cities

# %%
# drop incidents where n_participants is nan
incidents_df = incidents_df[~incidents_df['n_participants'].isna()]

# %%
(incidents_df.groupby(['city', 'state'])['week'].count() > 30).to_list().count(True) # 588 cities with incidents in more than 30 weeks

# %%
# list of index of incidents in city with incidents in more than 30 weeks
index_list = np.where(incidents_df.groupby(['city', 'state'])['week'].transform('count') > 30)

# %%
# create a df with incidents_df where index is in index_list
incidents_df = incidents_df.iloc[index_list]
incidents_df.head(2)

# %%
incidents_df.groupby(['week', 'city', 'state']).count()

# %%
incidents_df['state'].unique().shape # 51: all states are present

# %%
# gruop by week and count incidents
plt.figure(figsize=(20, 5))
plt.bar(
    incidents_df.groupby('week').size().index,
    incidents_df.groupby('week').size().values
)
plt.title('Number of incidents per week');

# %%
# number of incidents in which n_killed is nan
incidents_df[incidents_df['n_killed'].isna()].shape[0] # 0

# %% [markdown]
# ## Create Time series

# %%
# Model each city as a sequence of incidents
incidents_df.groupby(['city', 'state'])['week'].count().sort_values(ascending=False) # 664 time series

# %% [markdown]
# Time series: mean number of participants per incident per week in each city
# 
# 0 if we have no incidents in the week or NaN values (i.e. incidents where we don not know the nember of participants)

# %%
# create a dataset with series of mean number of participants per incident per week in each city
incidents_by_city_df = incidents_df.groupby(['city', 'state', 'week'])['n_participants'].mean().reset_index()
incidents_by_city_df = incidents_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_participants')
incidents_by_city_df = incidents_by_city_df.fillna(0) # substitute NaN with 0
incidents_by_city_df

# %%
# create a dataset with series of mean number of killed per incident per week in each city
incidents_killed_by_city_df = incidents_df.groupby(['city', 'state', 'week'])['n_killed'].mean().reset_index()
incidents_killed_by_city_df = incidents_killed_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_killed')
incidents_killed_by_city_df = incidents_killed_by_city_df.fillna(0) # substitute NaN with 0

# %%
incidents_by_city_df.groupby('state')[0].count().sort_values(ascending=False).plot(kind='bar', figsize=(20, 5));

# %%
n_weeks_per_year = 52

plt.figure(figsize=(20, 5))
plt.plot(np.nanmean(incidents_by_city_df.values, axis=0), '.--', label='n_participants')
#plt.plot(np.nanmean(incidents_killed_by_city_df.values, axis=0), '.--', label='n_killed')
plt.plot(np.nanmean((incidents_killed_by_city_df.values+incidents_by_city_df.values.mean()-incidents_killed_by_city_df.values.mean()
    ), axis=0), '.--', label='n_killed traslated') # traslate n_killed to have the same mean of n_participants
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
plt.legend()
plt.title('Average number of participants per week (mean over all cities)');

# %%
plt.figure(figsize=(20, 5))
new_york_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'City of New York')].values[0]
los_angeles_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'Los Angeles')].values[0]
chicago_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'Chicago')].values[0]
plt.plot(new_york_ts, '.--', label='New York')
plt.plot(los_angeles_ts, '.--', label='Los Angeles')
plt.plot(chicago_ts, '.--', label='Chicago')
plt.title('Average number of participants per week in New York, Los Angeles and Chicago')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
plt.legend();

# %%
# plot time series for city in ALASKA state
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.T, '.--')
plt.title('Average number of participants per week in ALASKA cities')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
plt.legend(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'
    ].index.get_level_values('city'), loc='upper left', bbox_to_anchor=(1, 1));

# %%
# visualize how many zeros in time series for each city
plt.figure(figsize=(20, 20))
plt.subplot(1, 5, 1)
plt.barh(
    incidents_by_city_df[:118].index.get_level_values('city'),
    incidents_by_city_df[:118].apply(lambda x: np.sum(x == 0), axis=1).values,
)
plt.subplot(1, 5, 2)
plt.barh(
    incidents_by_city_df[118:236].index.get_level_values('city'),
    incidents_by_city_df[118:236].apply(lambda x: np.sum(x == 0), axis=1).values,
)
plt.subplot(1, 5, 3)
plt.barh(
    incidents_by_city_df[236:354].index.get_level_values('city'),
    incidents_by_city_df[236:354].apply(lambda x: np.sum(x == 0), axis=1).values,
)
plt.subplot(1, 5, 4)
plt.barh(
    incidents_by_city_df[354:471].index.get_level_values('city'),
    incidents_by_city_df[354:471].apply(lambda x: np.sum(x == 0), axis=1).values,
)
plt.subplot(1, 5, 5)
plt.barh(
    incidents_by_city_df[471:].index.get_level_values('city'),
    incidents_by_city_df[471:].apply(lambda x: np.sum(x == 0), axis=1).values,
)
#plt.title('Number of zeros in time series for each city')
plt.tight_layout();

# %% [markdown]
# ## Preprocessing

# %% [markdown]
# ### Translate and Scaling Time Series

# %%
new_york_killed_ts = incidents_killed_by_city_df[(incidents_killed_by_city_df.index.get_level_values('city'
    ) == 'City of New York')].values[0]
los_angeles_killed_ts = incidents_killed_by_city_df[(incidents_killed_by_city_df.index.get_level_values('city'
    ) == 'Los Angeles')].values[0]
chicago_killed_ts = incidents_killed_by_city_df[(incidents_killed_by_city_df.index.get_level_values('city'
    ) == 'Chicago')].values[0]

# %% [markdown]
# Offset translation:

# %%
plt.figure(figsize=(20, 5))
plt.plot(new_york_ts-np.nanmean(new_york_ts), '.--', label='New York')
plt.plot(los_angeles_ts-np.nanmean(los_angeles_ts), '.--', label='Los Angeles')
plt.plot(chicago_ts-np.nanmean(chicago_ts), '.--', label='Chicago')
plt.title('Average number of participants per week in New York, Los Angeles and Chicago')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
plt.legend();

# %%
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.T - 
    incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.mean(axis=1), '.--')
plt.title('Average number of participants per week in ALASKA cities, offset translation')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
plt.legend(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].index.get_level_values('city'), 
    loc='upper left', bbox_to_anchor=(1, 1));

# %%
fig, ax = plt.subplots(3, 1, figsize=(20, 7))
ax[0].plot(new_york_ts, '.--', label='New York (original)')
ax[0].plot(new_york_ts-np.nanmean(new_york_ts), '.--', label='New York')
ax[0].plot(new_york_killed_ts-np.nanmean(new_york_killed_ts), '.--', label='New York killed')
ax[0].legend()
ax[1].plot(los_angeles_ts, '.--', label='Los Angeles (original)')
ax[1].plot(los_angeles_ts-np.nanmean(los_angeles_ts), '.--', label='Los Angeles')
ax[1].plot(los_angeles_killed_ts-np.nanmean(los_angeles_killed_ts), '.--', label='Los Angeles killed')
ax[1].legend()
ax[2].plot(chicago_ts, '.--', label='Chicago (original)')
ax[2].plot(chicago_ts-np.nanmean(chicago_ts), '.--', label='Chicago')
ax[2].plot(chicago_killed_ts-np.nanmean(chicago_killed_ts), '.--', label='Chicago killed')
ax[2].legend()
fig.suptitle('Average number of participants per week in New York, Los Angeles and Chicago, offset translation')
plt.tight_layout();

# %% [markdown]
# Amplitude scale:

# %%
fig, ax = plt.subplots(3, 1, figsize=(20, 7))
ax[0].plot(new_york_ts, '.--', label='New York (original)')
ax[0].plot(new_york_ts-np.nanmean(new_york_ts)/np.std(new_york_ts), '.--', label='New York')
ax[0].plot(new_york_killed_ts-np.nanmean(new_york_killed_ts)/np.std(new_york_killed_ts), '.--', label='New York killed')
ax[0].legend()
ax[1].plot(los_angeles_ts, '.--', label='Los Angeles (original)')
ax[1].plot(los_angeles_ts-np.nanmean(los_angeles_ts)/np.std(los_angeles_ts), '.--', label='Los Angeles')
ax[1].plot(los_angeles_killed_ts-np.nanmean(los_angeles_killed_ts)/np.std(los_angeles_killed_ts), '.--', label='Los Angeles killed')
ax[1].legend()
ax[2].plot(chicago_ts, '.--', label='Chicago (original)')
ax[2].plot(chicago_ts-np.nanmean(chicago_ts)/np.std(chicago_ts), '.--', label='Chicago')
ax[2].plot(chicago_killed_ts-np.nanmean(chicago_killed_ts)/np.std(chicago_killed_ts), '.--', label='Chicago killed')
ax[2].legend()
fig.suptitle('Average number of participants per week in New York, Los Angeles and Chicago, amplitude normalized')
fig.tight_layout();


# %% [markdown]
# ### Noise detection: tslearn library

# %%
scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(incidents_by_city_df.values)

# %%
fig, ax = plt.subplots(3, 1, figsize=(20, 7))
ax[0].plot(new_york_ts, '.--', label='New York (original)')
ax[0].plot(scaler.fit_transform(new_york_ts.reshape(1, -1))[0], '.--', label='New York')
ax[0].plot(scaler.fit_transform(new_york_killed_ts.reshape(1, -1))[0], '.--', label='New York killed')
ax[0].legend()
ax[1].plot(los_angeles_ts, '.--', label='Los Angeles')
ax[1].plot(scaler.fit_transform(los_angeles_ts.reshape(1, -1))[0], '.--', label='Los Angeles')
ax[1].plot(scaler.fit_transform(los_angeles_killed_ts.reshape(1, -1))[0], '.--', label='Los Angeles killed')
ax[1].legend()
ax[2].plot(chicago_ts, '.--', label='Chicago')
ax[2].plot(scaler.fit_transform(chicago_ts.reshape(1, -1))[0], '.--', label='Chicago scaled')
ax[2].plot(scaler.fit_transform(chicago_killed_ts.reshape(1, -1))[0], '.--', label='Chicago killed')
ax[2].legend()
fig.suptitle('Average number of participants per week in New York, Los Angeles and Chicago')
fig.tight_layout();

# %% [markdown]
# ### Compute Distance Between Time Series

# %%
def ts_distance(ts1, ts2, city):
    print(f'Time Series distance for {city}')
    ts_distance_df = pd.DataFrame()
    ts_distance_df.loc['euclidean', 'Original Time Series'] = euclidean(ts1, ts2)
    ts_distance_df.loc['cityblock', 'Original Time Series'] = cityblock(ts1, ts2)
    ts_distance_df.loc['euclidean', 'Offset Translation'] = euclidean(ts1-np.nanmean(ts1), ts2-np.nanmean(ts2))
    ts_distance_df.loc['cityblock', 'Offset Translation'] = cityblock(ts1-np.nanmean(ts1), ts2-np.nanmean(ts2))
    ts_distance_df.loc['euclidean', 'Amplitude Scaling'] = euclidean(ts1-np.nanmean(ts1)/np.std(ts1), ts2-np.nanmean(ts2)/np.std(ts2))
    ts_distance_df.loc['cityblock', 'Amplitude Scaling'] = cityblock(ts1-np.nanmean(ts1)/np.std(ts1), ts2-np.nanmean(ts2)/np.std(ts2))
    scaler = TimeSeriesScalerMeanVariance()
    ts_distance_df.loc['euclidean', 'Mean Variance Scaling'] = euclidean(scaler.fit_transform(ts1.reshape(1, -1))[0].reshape(-1), 
        scaler.fit_transform(ts2.reshape(1, -1))[0].reshape(-1))
    ts_distance_df.loc['cityblock', 'Mean Variance Scaling'] = cityblock(scaler.fit_transform(ts1.reshape(1, -1))[0].reshape(-1), 
        scaler.fit_transform(ts2.reshape(1, -1))[0].reshape(-1))
    display(ts_distance_df)

# %%
ts_distance(new_york_ts, new_york_killed_ts, 'City of New York')
ts_distance(los_angeles_ts, los_angeles_killed_ts, 'Los Angeles')
ts_distance(chicago_ts, chicago_killed_ts, 'Chicago')
ts_distance(incidents_by_city_df.mean(axis=0).values, incidents_killed_by_city_df.mean(axis=0).values, 'Average All Cities')

# %% [markdown]
# ## Create dataset for Cities

# %%
# create a dataframe with as index city and state
cities_df = incidents_df.groupby(['city', 'state'])['population_state_2010'].mean() # population_state_2010
cities_df = pd.DataFrame(cities_df)

# quantile of population_state_2010
cities_df['population_quantile'] = pd.qcut(cities_df['population_state_2010'], 4, labels=False)

# n_incidents
cities_df['n_incidents_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['week'].count()
cities_df['n_incidents_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['week'].count()
cities_df['n_incidents_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['week'].count()
cities_df['n_incidents_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['week'].count()
cities_df['n_incidents'] = incidents_df.groupby(['city', 'state'])['week'].count()

# n_weeks_with_incidents
cities_df['n_weeks_with_incidents_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['week'].nunique()
cities_df['n_weeks_with_incidents_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['week'].nunique()
cities_df['n_weeks_with_incidents_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['week'].nunique()
cities_df['n_weeks_with_incidents_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['week'].nunique()
cities_df['n_weeks_with_incidents'] = incidents_df.groupby(['city', 'state'])['week'].nunique()

# n_participants
cities_df['n_participants_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['n_participants'].sum()
cities_df['n_participants_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['n_participants'].sum()
cities_df['n_participants_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['n_participants'].sum()
cities_df['n_participants_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['n_participants'].sum()
cities_df['n_participants'] = incidents_df.groupby(['city', 'state'])['n_participants'].sum()

# n_participants_avg
cities_df['n_participants_avg_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['n_participants'].mean()
cities_df['n_participants_avg_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['n_participants'].mean()
cities_df['n_participants_avg_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['n_participants'].mean()
cities_df['n_participants_avg_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['n_participants'].mean()
cities_df['n_participants_avg'] = incidents_df.groupby(['city', 'state'])['n_participants'].mean()

# n_killed
cities_df['n_killed_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['n_killed'].sum()
cities_df['n_killed_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['n_killed'].sum()
cities_df['n_killed_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['n_killed'].sum()
cities_df['n_killed_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['n_killed'].sum()
cities_df['n_killed'] = incidents_df.groupby(['city', 'state'])['n_killed'].sum()

# n_killed_avg
cities_df['n_killed_avg_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['n_killed'].mean()
cities_df['n_killed_avg_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['n_killed'].mean()
cities_df['n_killed_avg_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['n_killed'].mean()
cities_df['n_killed_avg_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['n_killed'].mean()
cities_df['n_killed_avg'] = incidents_df.groupby(['city', 'state'])['n_killed'].mean()

# %%
cities_df.head(2)

# %%
cities_df[['population_state_2010', 'population_quantile', 'n_incidents_2014', 'n_incidents_2015', 'n_incidents_2016',
    'n_incidents_2017', 'n_incidents']].describe()

# %%
cities_df[['n_weeks_with_incidents_2014', 'n_weeks_with_incidents_2015', 
    'n_weeks_with_incidents_2016', 'n_weeks_with_incidents_2017', 'n_weeks_with_incidents']].describe()

# %%
cities_df[['n_participants_2014', 'n_participants_2015', 'n_participants_2016', 'n_participants_2017',
    'n_participants', 'n_participants_avg_2014', 'n_participants_avg_2015', 'n_participants_avg_2016',
    'n_participants_avg_2017', 'n_participants_avg']].describe()

# %%
cities_df[['n_killed_2014', 'n_killed_2015', 'n_killed_2016', 'n_killed_2017', 'n_killed',
    'n_killed_avg_2014', 'n_killed_avg_2015', 'n_killed_avg_2016', 'n_killed_avg_2017', 'n_killed_avg']].describe()

# %%
# make a plot to compare n_incidents, n_participants, n_killed an population
fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax[0, 0].scatter(cities_df['n_incidents'].values, cities_df['population_state_2010'].values, 
    c=cities_df['population_quantile'].values)
ax[0, 0].set_xlabel('n_incidents')
ax[0, 0].set_ylabel('Population')
ax[0, 1].scatter(cities_df['n_participants'].values, cities_df['population_state_2010'].values, 
    c=cities_df['population_quantile'].values)
ax[0, 1].set_xlabel('n_participants')
ax[0, 1].set_ylabel('Population')
ax[0, 2].scatter(cities_df['n_killed'].values, cities_df['population_state_2010'].values, 
    c=cities_df['population_quantile'].values)
ax[0, 2].set_xlabel('n_killed')
ax[0, 2].set_ylabel('Population')
ax[1, 0].scatter(cities_df['n_incidents'].values, cities_df['n_participants'].values,
    c=cities_df['population_quantile'].values)
ax[1, 0].set_xlabel('n_incidents')
ax[1, 0].set_ylabel('n_participants')
ax[1, 1].scatter(cities_df['n_incidents'].values, cities_df['n_killed'].values,
    c=cities_df['population_quantile'].values)
ax[1, 1].set_xlabel('n_incidents')
ax[1, 1].set_ylabel('n_killed')
ax[1, 2].scatter(cities_df['n_participants'].values, cities_df['n_killed'].values,
    c=cities_df['population_quantile'].values)
ax[1, 2].set_xlabel('n_participants')
ax[1, 2].set_ylabel('n_killed')
fig.suptitle('Correlation between population and number of incidents, participants and killed')
fig.tight_layout();

# %% [markdown]
# ## Clustering

# %% [markdown]
# ### Shape-based clustering: k-means

# %% [markdown]
# Choose best k:

# %%
X = TimeSeriesScalerMeanVariance().fit_transform(incidents_by_city_df.values) # scale time series
inertia_list = [] # sum of distances of samples to their closest cluster center

for k in range(2, 20):
    km = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=100, random_state=42)
    km.fit(X)
    pred = km.predict(X)
    print("n clusters = ", k, "\t Clusters =", np.unique(pred,return_counts=True)[1], "\t Inertia =", km.inertia_)
    inertia_list.append(km.inertia_)

# %%
plt.figure(figsize=(20, 5))
plt.plot(inertia_list, '.--')
plt.xticks(range(len(inertia_list)), range(2, 20))
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia for different number of clusters');

# %% [markdown]
# Fit chosen model

# %%
best_k = 10
km = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=100, random_state=42)
km.fit(X)

# %%
plt.figure(figsize=(20, 5))
plt.plot(km.cluster_centers_.reshape(incidents_by_city_df.values.shape[1], best_k), '.-')
plt.title('Centroids of clusters')
plt.legend(range(11), loc='upper left', bbox_to_anchor=(1, 1));

# %%
km.inertia_ # Sum of distances of samples to their closest cluster center

# %%
cluster = km.fit_predict(incidents_by_city_df.values)

# %% [markdown]
# Visualize clusters

# %%
cluster_df = incidents_df.groupby(['city', 'state'])[['latitude', 'longitude']].mean().reset_index()
cluster_df['cluster_kmeans'] = cluster
cluster_df.head(2)

# %%
fig = px.scatter_mapbox(
    lat=cluster_df['latitude'],
    lon=cluster_df['longitude'],
    zoom=2, 
    color=cluster_df['cluster_kmeans'],
    height=400,
    width=1000,
    text=cluster_df['city'] + ', ' + cluster_df['state']
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %%
plt.figure(figsize=(20, 5))
plt.bar(
    cluster_df.groupby('cluster_kmeans').size().index,
    cluster_df.groupby('cluster_kmeans').size().values
)
plt.title('Number of cities per cluster');

# %%
# visualize time series for each cluster (mean)
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df.groupby(cluster).mean().values.T, '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(incidents_by_city_df.groupby(cluster).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %% [markdown]
# Visualize if there is a correlation between city in the same cluster and population:

# %%
km_crosstab = pd.crosstab(km.labels_, cities_df['population_quantile'], rownames=['cluster'], colnames=['population_quantile'])
km_crosstab.plot(kind='bar', stacked=False, title='Population quantile for each cluster');

# %% [markdown]
# no :(

# %% [markdown]
# Performe the same cluster using time series with n_killed in order to compare the results

# %%
km = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=100, random_state=42)
cluster_df['cluster_kmeans_killed'] = km.fit_predict(
    TimeSeriesScalerMeanVariance().fit_transform(incidents_killed_by_city_df.values))
cluster_df.head(2)

# %%
sankey_plot(
    [cluster_df['cluster_kmeans'], cluster_df['cluster_kmeans_killed']],
    labels_titles=['n_participants', 'n_killed'],
    title='Clusterings comparison TimeSeriesKMeans'
)

# %% [markdown]
# ### Compression-based clustering

# %%
def cdm_dist(x, y):
    # compounding dissimilarity measure
    x_str = (' '.join([str(v) for v in x.ravel()])).encode('utf-8')
    y_str = (' '.join([str(v) for v in y.ravel()])).encode('utf-8')
    return len(zlib.compress(x_str + y_str)) / (len(zlib.compress(x_str)) + len(zlib.compress(y_str)))

X = incidents_by_city_df.values
M = pairwise_distances(X.reshape(X.shape[0], X.shape[1]), metric=cdm_dist)

# %% [markdown]
# #### Hierarchical Clustering measuring the distance between each pair of points in a dataset via Pairwise Distances

# %%
algorithms = ["single", "complete", "average", "ward"]
linkages = []
distance_thresholds = []
dm = pdist(M, metric='euclidean')
for algorithm in algorithms:
    linkage_res = linkage(dm, method=algorithm, metric='euclidean', optimal_ordering=False)
    linkages.append(linkage_res)
    distance_thresholds.append(0.7 * max(linkage_res[:,2]))

# %%
def plot_dendrograms(linkages, algorithms, thresholds):
    _, axs = plt.subplots(ncols=len(linkages), figsize=(32,7))

    for i in range(len(linkages)):
        axs[i].set_title(algorithms[i])
        axs[i].set_xlabel('IncidentID or (Cluster Size)')
        axs[i].set_ylabel('Distance')
        axs[i].axhline(thresholds[i], ls='--', color='k', label='threshold')
        color_threshold = thresholds[i]
        dendrogram(linkages[i], truncate_mode="lastp", p=30, leaf_rotation=60, leaf_font_size=8,
                show_contracted=True, ax=axs[i], color_threshold=color_threshold)

    plt.suptitle(('Hierarchical Clustering Dendograms'), fontsize=18, fontweight='bold')
    plt.show()

# %%
plot_dendrograms(linkages, algorithms, distance_thresholds)

# %% [markdown]
# We evaluate the clustering results using the cophenetic correlation coefficient:

# %%
cophenetic_coefs = []
for linkage in linkages:
    cophenetic_matrix = cophenet(linkage)
    cophenetic_coef = np.corrcoef(cophenetic_matrix, dm)[0][1]
    cophenetic_coefs.append(cophenetic_coef)
pd.DataFrame({'algorithm': algorithms, 'cophenetic_coef': cophenetic_coefs})

# %% [markdown]
# The best results are obtained using average linkage.

# %% [markdown]
# Try using 4 clusters:

# %%
hier = AgglomerativeClustering(n_clusters=4, linkage='average')
hier.fit(M)

# %%
plt.figure(figsize=(20, 5))
for i in range(4):
    plt.plot(np.mean(X[np.where(hier.labels_ == i)[0]], axis=0), '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(incidents_by_city_df.groupby(hier.labels_).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %%
cluster_df['cluster_hierarchical4'] = hier.labels_
cluster_df.head(2)

# %%
fig = px.scatter_mapbox(
    lat=cluster_df['latitude'],
    lon=cluster_df['longitude'],
    zoom=2, 
    color=cluster_df['cluster_hierarchical4'],
    height=400,
    width=1000,
    text=cluster_df['city'] + ', ' + cluster_df['state']
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %%
plt.figure(figsize=(20, 5))
plt.bar(
    cluster_df.groupby('cluster_hierarchical4').size().index,
    cluster_df.groupby('cluster_hierarchical4').size().values
)
plt.title('Number of cities per cluster');

# %% [markdown]
# Molto sbilanciate e compunque sono quasi tutti in 1 cluster, inutile
# 
# Use n_clusters = 6:

# %%
hier = AgglomerativeClustering(n_clusters=6, linkage='average')
hier.fit(M)

# %%
plt.figure(figsize=(20, 5))
for i in range(6):
    plt.plot(np.mean(X[np.where(hier.labels_ == i)[0]], axis=0), '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(incidents_by_city_df.groupby(hier.labels_).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %%
cluster_df['cluster_hierarchical6'] = hier.labels_
cluster_df.head(2)

# %%
fig = px.scatter_mapbox(
    lat=cluster_df['latitude'],
    lon=cluster_df['longitude'],
    zoom=2, 
    color=cluster_df['cluster_hierarchical6'],
    height=400,
    width=1000,
    text=cluster_df['city'] + ', ' + cluster_df['state']
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %%
plt.figure(figsize=(20, 5))
plt.bar(
    cluster_df.groupby('cluster_hierarchical6').size().index,
    cluster_df.groupby('cluster_hierarchical6').size().values
)
plt.title('Number of cities per cluster');

# %%
hier = AgglomerativeClustering(n_clusters=6, linkage='average')
X = incidents_killed_by_city_df.values
hier.fit(pairwise_distances(X.reshape(X.shape[0], X.shape[1]), metric=cdm_dist))
cluster_df['cluster_hierarchical6_killed'] = hier.labels_

# %%
sankey_plot(
    [cluster_df['cluster_hierarchical6'], cluster_df['cluster_hierarchical6_killed']],
    labels_titles=['n_participants', 'n_killed'],
    title='Clusterings comparison HierarchicalClustering'
)

# %% [markdown]
# #### K-means using Piecewise Aggregate Approximation of time series

# %% [markdown]
# Piecewise Aggregate Approximation (PAA) is a technique used in time series analysis to reduce the dimensionality of a time series while preserving its essential characteristics.
# 
# PAA approximates a time-series $X$ of length $n$ into vector $\hat{X}=(\hat{x}_1,…,\hat{x}_M)$
#  of any arbitrary length  $M\leq n$
#  
# $x_i = \frac{M}{n} \sum_{j=\frac{M}{n}(i-1)+1}^{\frac{M}{n}i} X_j$

# %%
n_paa_segments = 100
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
X_paa = paa.fit_transform(incidents_by_city_df.values) # PAA transformation

# %%
plt.figure(figsize=(20, 5))
plt.plot(X_paa.reshape(X_paa.shape[1], X_paa.shape[0]))
plt.title('PAA representation of time series');

# %%
km = TimeSeriesKMeans(n_clusters=11, metric="dtw", max_iter=5, random_state=0)
km.fit(X_paa)

# %%
plt.figure(figsize=(20, 5))
plt.plot(km.cluster_centers_.reshape(X_paa.shape[1], 11))
plt.title('Centroids of clusters')
plt.legend(range(11), loc='upper left', bbox_to_anchor=(1, 1));

# %%
plt.figure(figsize=(20, 5))
for i in range(11):
    plt.plot(np.mean(X[np.where(km.labels_ == i)[0]], axis=0), '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(incidents_by_city_df.groupby(km.labels_).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %%
cluster_df['cluster_kmeans_paa'] = km.labels_
cluster_df.head(2)

# %%
fig = px.scatter_mapbox(
    lat=cluster_df['latitude'],
    lon=cluster_df['longitude'],
    zoom=2, 
    color=cluster_df['cluster_kmeans_paa'],
    height=400,
    width=1000,
    text=cluster_df['city'] + ', ' + cluster_df['state']
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %%
plt.figure(figsize=(20, 5))
plt.bar(
    cluster_df.groupby('cluster_kmeans_paa').size().index,
    cluster_df.groupby('cluster_kmeans_paa').size().values
)
plt.title('Number of cities per cluster');

# %%
paa = PiecewiseAggregateApproximation(n_segments=100)
X_paa = paa.fit_transform(incidents_killed_by_city_df.values) # PAA transformation
km = TimeSeriesKMeans(n_clusters=11, metric="dtw", max_iter=5, random_state=0)
km.fit(X_paa)
cluster_df['cluster_kmeans_paa_killed'] = km.labels_

# %%
sankey_plot(
    [cluster_df['cluster_kmeans_paa'], cluster_df['cluster_kmeans_paa_killed']],
    labels_titles=['n_participants', 'n_killed'],
    title='Clusterings comparison KMeansTimeSeries PAA'
)

# %%
km_crosstab = pd.crosstab(km.labels_, quantile_list, rownames=['cluster'], colnames=['population_quantile'])
km_crosstab.plot(kind='bar', stacked=False, title='Population quantile for each cluster');

# %% [markdown]
# ### Clustering comparison

# %%
sankey_plot(
    [cluster_df['cluster_kmeans'], cluster_df['cluster_kmeans_paa'], cluster_df['cluster_hierarchical6'], 
    cluster_df['cluster_hierarchical4']],
    labels_titles=['TimeSeriesKMeans', 'TimeSeriesKMeans PAA', 'Hierarchical (6 clusters)', 'Hierarchical (4 clusters)'],
    title='Clusterings comparison'
)

# %% [markdown]
# ## Matrix profile

# %% [markdown]
# The matrix profile is constructed by combining distance profiles for all possible subsequence pairs in the time series. It highlights repeated patterns or motifs by identifying low values in the matrix profile.

# %% [markdown]
# Compute and visualize matrix profile (a data structure that annotates time series by using a sliding window to compare the pairwise distance among the subsequences):

# %%
# estrapolate time series for new york, los angeles and chicago
new_york_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'City of New York')].values[0]
los_angeles_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'Los Angeles')].values[0]
chicago_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'Chicago')].values[0]

w_list = [8, 13, 26, 52] # length of subsequence to compare: two month, quarter, half year, year

fig, ax = plt.subplots(5, 1, figsize=(18, 12))
ax[0].plot(new_york_ts, '.--', label='New York')
ax[0].plot(los_angeles_ts, '.--', label='Los Angeles')
ax[0].plot(chicago_ts, '.--', label='Chicago')
ax[0].set_ylabel('Time series', size=10)
ax[0].axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
ax[0].axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
ax[0].axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
ax[0].legend()

for i, w in enumerate(w_list):
    mp, mpi = matrixProfile.stomp(new_york_ts, w)
    ax[i+1].plot(mp, '.--')
    mp, mpi = matrixProfile.stomp(los_angeles_ts, w)
    ax[i+1].plot(mp, '.--')
    mp, mpi = matrixProfile.stomp(chicago_ts, w)
    ax[i+1].plot(mp, '.--')
    ax[i+1].set_ylabel('Matrix Profile, w = ' + str(w), size=10)

# %% [markdown]
# ##  Motifs extraction

# %% [markdown]
# Parameters
# - max_motifs: stop finding new motifs once we have max_motifs
# - radius: For each motif found, find neighbors that are within radius*motif_mp of the first.
# - n_neighbors: number of neighbors from the first to find. If it is None, find all.
# - ex_zone: minimum distance between indices for after each subsequence is identified. Defaults to m/2 where m is the subsequence length. If ex_zone = 0, only the found index is exclude, if ex_zone = 1 then if idx is found as a motif idx-1, idx, idx+1 are excluded.
# 
# 
# The function returns a tuple (top_motifs, distances) which are lists of the same length.
# 
# - top_motifs: This is a list of the indices found for each motif. The first index is the nth motif followed by all nearest neighbors found sorted by distances.
# - distances: Minimum Matrix profile value for each motif set.

# %%
mo, mod  = motifs.motifs(new_york_ts, (mp, mpi), max_motifs=5)

# %%
plt.figure(figsize=(20, 5))
plt.plot(new_york_ts)
colors = ['r', 'g', 'k', 'b', 'y'][:len(mo)]
for m, d, c in zip(mo, mod, colors):
    for i in m:
        m_shape = new_york_ts[i:i+w]
        plt.plot(range(i,i+w), m_shape, color=c, lw=3)
plt.title('Motifs in New York time series');

# %%
fig, ax = plt.subplots(len(mo[0]), 1, figsize=(18, 10))
for m, d, c in zip(mo, mod, colors):
    for j, i in enumerate(m):
        m_shape = new_york_ts[i:i+w]
        ax[j].plot(range(i,i+w), m_shape, color=c, lw=3)
        ax[j].set_ylabel('Motif ' + str(j+1), size=10)
plt.suptitle('Motifs in New York time series', size=20);

# %% [markdown]
# ## Anomalies extraction

# %% [markdown]
# Parameters  
# - mp: matrix profile numpy array
# - k: the number of discords to discover
# - ex_zone: the number of samples to exclude and set to Inf on either side of a found discord   
# 
# 
# Returns a list of indexes represent the discord starting locations. MaxInt indicates there were no more discords that could be found due to too many exclusions or profile being too small. Discord start indices are sorted by highest matrix profile value.

# %%
anoms = discords(mp, ex_zone=3, k=5)

# %%
plt.figure(figsize=(20, 5))
plt.plot(new_york_ts)
colors = ['r', 'g', 'k', 'b', 'y'][:len(mo)]
for a, c in zip(anoms, colors):
    a_shape = new_york_ts[a:a+w]
    plt.plot(range(a, a+w), a_shape, color=c, lw=3)
plt.title('Anomalies in New York time series');

# %% [markdown]
# ## Shaplet discovery

# %%
from keras.optimizers import Adagrad
from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from sklearn.metrics import accuracy_score

# %% [markdown]
# Create binary var is_killed:

# %%
# create binary var isKilled
incidents_df['n_killed'] = incidents_df['n_killed'].apply(lambda x: 1 if x > 0 else 0)
print('Number of killed incidents: ', incidents_df['n_killed'].sum())
print('Number of incidents: ', incidents_df['n_killed'].shape[0])

# %%
# asssociate each incident to a city
is_killed_by_city_df = incidents_df.groupby(['city', 'state'])['n_killed'].sum()
is_killed_by_city_df.describe()

# %%
is_killed = is_killed_by_city_df.apply(lambda x: 1 if x > np.quantile(is_killed_by_city_df, 0.75) else 0)

# %%
print('Number of cities with number of killed people in incidents in third quantile: ', is_killed.sum())
print('Number of cities: ', is_killed.shape[0])

# %%
#paa = PiecewiseAggregateApproximation(n_segments=100)
#X = paa.fit_transform(incidents_by_city_df.values)
X = incidents_by_city_df.values
y = is_killed

# %%
n_ts, ts_sz = X.shape[0], X.shape[1]
n_classes = len(set(y))

shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz,
    n_classes=n_classes, l=0.1, r=1)
print('n_ts', n_ts)
print('ts_sz', ts_sz)
print('n_classes', n_classes)
print('shapelet_sizes', shapelet_sizes)

# %%
shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer="sgd",
                        weight_regularizer=.01,
                        max_iter=200,
                        verbose=1)

# %%
shp_clf.fit(X, y)

# %%
predicted_labels = shp_clf.predict(X)
print("Correct classification rate:", accuracy_score(y, predicted_labels))
predicted_locations = shp_clf.locate(X)

# %%
ts_id = 0
plt.figure()
n_shapelets = sum(shapelet_sizes.values())
plt.title(f"""Example locations of shapelet matches ({n_shapelets} shapelets extracted)""")

plt.plot(X[ts_id].ravel())
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[ts_id, idx_shp]
    plt.plot(np.arange(t0, t0 + len(shp)), shp, linewidth=2)


