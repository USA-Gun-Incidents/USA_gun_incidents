# %% [markdown]
# # Time Series Analysis

# %% [markdown]
# Import library and dataset:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import zlib
import datetime
import tensorflow as tf
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import pairwise_distances
from tslearn.metrics import dtw_path as dtw
from sklearn.cluster import AgglomerativeClustering
from tslearn.piecewise import PiecewiseAggregateApproximation
from sklearn.cluster import KMeans
import scipy.stats as stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
from scipy.spatial.distance import euclidean
from time_series_utils import *
from matrixprofile import *
from matrixprofile.discords import discords
from claspy.window_size import highest_autocorrelation
from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sktime.classification.kernel_based import RocketClassifier
from tslearn.shapelets import LearningShapelets
from sklearn.tree import DecisionTreeClassifier

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

# %% [markdown]
# We use only data corresponding to incidents in 2014, 2015, 2016 or 2017. 

# %%
incidents_df['year'].unique()

# %%
pd.to_datetime('2014-1-1', format='%Y-%m-%d').day_name()

# %%
pd.to_datetime('2014-1-6', format='%Y-%m-%d').day_name()

# %%
pd.to_datetime('2017-12-31', format='%Y-%m-%d').day_name()

# %% [markdown]
# We extract a time series for each city, computing a value for each week of the 4 years a score.
# 
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
# group by week and count incidents
plt.figure(figsize=(20, 3))
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

# %% [markdown]
# We drop incidents with NaN values for n_particpants (i.e. incidents where we don not know the number of participants).

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
# group by week and count incidents
plt.figure(figsize=(20, 3))
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

# %% [markdown]
# We create time series model each city as a sequence of incidents.

# %%
# Model each city as a sequence of incidents
incidents_df.groupby(['city', 'state'])['week'].count().sort_values(ascending=False) # 664 time series

# %% [markdown]
# We create time series that rappresent the mean number of participants per incident per week in each city.
# 
# The time series assumes value 0 if we have no incidents in the week.

# %%
# create a dataset with series of mean number of participants per incident per week in each city
incidents_by_city_df = incidents_df.groupby(['city', 'state', 'week'])['n_participants'].mean().reset_index()
incidents_by_city_df = incidents_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_participants')
incidents_by_city_df = incidents_by_city_df.fillna(0) # substitute NaN with 0
incidents_by_city_df

# %% [markdown]
# We also create time series rappresent the mean number of killed people per incident per week in each city, in order to compare the results optained with the previusly definte time series.

# %%
# create a dataset with series of mean number of killed people per incident per week in each city
incidents_killed_by_city_df = incidents_df.groupby(['city', 'state', 'week'])['n_killed'].mean().reset_index()
incidents_killed_by_city_df = incidents_killed_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_killed')
incidents_killed_by_city_df = incidents_killed_by_city_df.fillna(0) # substitute NaN with 0

# %% [markdown]
# We visualize how many time series (i.e. cities) we have for each state:

# %%
incidents_by_city_df.groupby('state')[0].count().sort_values(ascending=False).plot(kind='bar', figsize=(20, 3));

# %% [markdown]
# Visualize average of all time series:

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

# %% [markdown]
# Visualize time series for New York, Los Angeles, Chicago:

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

# %% [markdown]
# Visualize time series for city in ALASKA state:

# %%
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.T, '.--')
plt.title('Average number of participants per week in ALASKA cities')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
plt.axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
plt.legend(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'
    ].index.get_level_values('city'), loc='upper left', bbox_to_anchor=(1, 1));

# %% [markdown]
# We visualize the number of zero values in the time series of each city:

# %%
# visualize how many zeros in time series for each city
plt.figure(figsize=(14, 10))
plt.subplot(1, 5, 1)
plt.barh(incidents_by_city_df[:118].index.get_level_values('city'),
    incidents_by_city_df[:118].apply(lambda x: np.sum(x == 0), axis=1).values)
plt.yticks(size=6)
plt.subplot(1, 5, 2)
plt.barh(incidents_by_city_df[118:236].index.get_level_values('city'),
    incidents_by_city_df[118:236].apply(lambda x: np.sum(x == 0), axis=1).values)
plt.yticks(size=6)
plt.subplot(1, 5, 3)
plt.barh(incidents_by_city_df[236:354].index.get_level_values('city'),
    incidents_by_city_df[236:354].apply(lambda x: np.sum(x == 0), axis=1).values)
plt.yticks(size=6)
plt.subplot(1, 5, 4)
plt.barh(incidents_by_city_df[354:471].index.get_level_values('city'),
    incidents_by_city_df[354:471].apply(lambda x: np.sum(x == 0), axis=1).values)
plt.yticks(size=6)
plt.subplot(1, 5, 5)
plt.barh(incidents_by_city_df[471:].index.get_level_values('city'),
    incidents_by_city_df[471:].apply(lambda x: np.sum(x == 0), axis=1).values)
plt.yticks(size=6)
#plt.suptitle('Number of zeros in time series for each city')
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

# %% [markdown]
# tslearn is a Python package that provides machine learning tools for the analysis of time series. This package builds on (and hence depends on) scikit-learn, numpy and scipy libraries.
# 
# We used the TimeSeriesScalerMeanVariance from tslear.
# This scaler standardizes time series data so that the mean in each dimension is set to *mu* and the standard deviation is set to *std*. In your case, you've mentioned that you used the default parameters with *mu=0* and *std=1*.

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
def ts_distance(ts1, ts2, city, verbose=True):
    if verbose: print(f'Time Series distance for {city}')
    ts_distance_df = pd.DataFrame()
    ts_distance_df.loc['euclidean', 'Original Time Series'] = euclidean(ts1, ts2)
    _, ts_distance_df.loc['DTW', 'Original Time Series'] = dtw(ts1, ts2)
    ts_distance_df.loc['euclidean', 'Offset Translation'] = euclidean(ts1-np.nanmean(ts1), ts2-np.nanmean(ts2))
    _, ts_distance_df.loc['DTW', 'Offset Translation'] = dtw(ts1-np.nanmean(ts1), ts2-np.nanmean(ts2))
    ts_distance_df.loc['euclidean', 'Amplitude Scaling'] = euclidean(ts1-np.nanmean(ts1)/np.std(ts1), ts2-np.nanmean(ts2)/np.std(ts2))
    _, ts_distance_df.loc['DTW', 'Amplitude Scaling'] = dtw(ts1-np.nanmean(ts1)/np.std(ts1), ts2-np.nanmean(ts2)/np.std(ts2))
    scaler = TimeSeriesScalerMeanVariance()
    ts_distance_df.loc['euclidean', 'Mean Variance Scaling'] = euclidean(scaler.fit_transform(ts1.reshape(1, -1))[0].reshape(-1), 
        scaler.fit_transform(ts2.reshape(1, -1))[0].reshape(-1))
    _, ts_distance_df.loc['DTW', 'Mean Variance Scaling'] = dtw(scaler.fit_transform(ts1.reshape(1, -1))[0].reshape(-1), 
        scaler.fit_transform(ts2.reshape(1, -1))[0].reshape(-1))
    if verbose: display(ts_distance_df)
    return ts_distance_df

# %% [markdown]
# We compute the distance between time series generated using the n_participants values and those generated using n_killed for all the transformations implemented before.
# 
# For visualization purpuse, we calculated and display the distances for the time series related to the cities of New York, Los Angeles, and Chicago, as well as the time series representing the average of all cities.
# 
# As metrics to compute distance we use Dynamic Time Warping (DTW) and Euclidan distance.

# %%
ts_distance(new_york_ts, new_york_killed_ts, 'City of New York')
ts_distance(los_angeles_ts, los_angeles_killed_ts, 'Los Angeles')
ts_distance(chicago_ts, chicago_killed_ts, 'Chicago')
ts_distance(incidents_by_city_df.mean(axis=0).values, incidents_killed_by_city_df.mean(axis=0).values, 'Average All Cities')

# %% [markdown]
# Cumpute distance between time series generated using the n_participants values and those generated using n_killed for all the transformations implemented before for all the time series and visualize them:

# %%
original_ts_distance = []
offset_translation_ts_distance = []
amplitude_scaling_ts_distance = []
mean_variance_scaling_ts_distance = []

for ts1, ts2 in zip(incidents_by_city_df.values, incidents_killed_by_city_df.values):
    ts_distance_df = ts_distance(ts1, ts2, '', verbose=False)
    original_ts_distance.append(ts_distance_df.loc['DTW', 'Original Time Series'])
    offset_translation_ts_distance.append(ts_distance_df.loc['DTW', 'Offset Translation'])
    amplitude_scaling_ts_distance.append(ts_distance_df.loc['DTW', 'Amplitude Scaling'])
    mean_variance_scaling_ts_distance.append(ts_distance_df.loc['DTW', 'Mean Variance Scaling'])

# %%
fig, ax = plt.subplots(3, 1, figsize=(20, 7), sharex=True, sharey=True)
ax[0].plot(original_ts_distance, '.--', label='Original Time Series')
ax[0].plot(offset_translation_ts_distance, '.--', label='Offset Translation')
ax[0].legend()
ax[0].set_ylabel('DTW distance')
ax[0].set_ylim(0, 30)

ax[1].plot(original_ts_distance, '.--', label='Original Time Series')
ax[1].plot(amplitude_scaling_ts_distance, '.--', label='Amplitude Scaling')
ax[1].legend()
ax[1].set_ylabel('DTW distance')

ax[2].plot(original_ts_distance, '.--', label='Original Time Series')
ax[2].plot(mean_variance_scaling_ts_distance, '.--', label='Mean Variance Scaling')
ax[2].legend()
ax[2].set_ylabel('DTW distance')

fig.suptitle('DTW distance between time series');

# %% [markdown]
# Each point on the plot represents the distance between the time series for a specific city.
# 
# We note that the time series scaled using the tslearn library are those for which the distance is minimized between the time series generated using n_participants and those generated using n_killed.

# %% [markdown]
# ## Create dataset for Cities

# %% [markdown]
# To better understand the cluster analysis on the time series we are going to perform afterward, we have created a dataset containing data for each city by extracting information from the available dataset.

# %% [markdown]
# | Feature                           | Description                                             |
# |-----------------------------------|---------------------------------------------------------|
# | population_state_2010              | Mean population of the state in the year 2010.           |
# | population_quantile                | Quantile of population_state_2010 (categorical).        |
# | n_incidents_2014                   | Number of incidents in the year 2014.                    |
# | n_incidents_2015                   | Number of incidents in the year 2015.                    |
# | n_incidents_2016                   | Number of incidents in the year 2016.                    |
# | n_incidents_2017                   | Number of incidents in the year 2017.                    |
# | n_incidents                        | Total number of incidents across all years.              |
# | n_incidents_quantile               | Quantile of n_incidents (categorical).                   |
# | n_weeks_with_incidents_2014        | Number of weeks with incidents in the year 2014.         |
# | n_weeks_with_incidents_2015        | Number of weeks with incidents in the year 2015.         |
# | n_weeks_with_incidents_2016        | Number of weeks with incidents in the year 2016.         |
# | n_weeks_with_incidents_2017        | Number of weeks with incidents in the year 2017.         |
# | n_weeks_with_incidents             | Total number of weeks with incidents across all years.   |
# | n_weeks_quantile                   | Quantile of n_weeks_with_incidents (categorical).        |
# | n_participants_2014               | Total number of participants in incidents in 2014.       |
# | n_participants_2015               | Total number of participants in incidents in 2015.       |
# | n_participants_2016               | Total number of participants in incidents in 2016.       |
# | n_participants_2017               | Total number of participants in incidents in 2017.       |
# | n_participants                    | Total number of participants across all years.           |
# | n_participants_quantile           | Quantile of n_participants (categorical).                |
# | n_participants_avg_2014           | Average number of participants in incidents in 2014.     |
# | n_participants_avg_2015           | Average number of participants in incidents in 2015.     |
# | n_participants_avg_2016           | Average number of participants in incidents in 2016.     |
# | n_participants_avg_2017           | Average number of participants in incidents in 2017.     |
# | n_participants_avg                | Average number of participants across all years.         |
# | n_killed_2014                     | Total number of killed in incidents in 2014.             |
# | n_killed_2015                     | Total number of killed in incidents in 2015.             |
# | n_killed_2016                     | Total number of killed in incidents in 2016.             |
# | n_killed_2017                     | Total number of killed in incidents in 2017.             |
# | n_killed                          | Total number of killed across all years.                 |
# | n_killed_quantile                 | Quantile of n_killed (categorical).                      |
# | n_killed_avg_2014                 | Average number of killed in incidents in 2014.           |
# | n_killed_avg_2015                 | Average number of killed in incidents in 2015.           |
# | n_killed_avg_2016                 | Average number of killed in incidents in 2016.           |
# | n_killed_avg_2017                 | Average number of killed in incidents in 2017.           |
# | n_killed_avg                      | Average number of killed across all years.               |
# | n_females                         | Total number of females across all years.                |
# | n_females_quantile                | Quantile of n_females (categorical).                     |
# | n_males                           | Total number of males across all years.                  |
# | n_males_quantile                  | Quantile of n_males (categorical).                       |
# | n_injured                         | Total number of injured across all years.                |
# | n_injured_quantile                | Quantile of n_injured (categorical).                     |
# | n_arrested                        | Total number of arrested across all years.               |
# | n_arrested_quantile               | Quantile of n_arrested (categorical).                    |
# | fatal_incidents                   | Total number of fatal incidents across all years.       |
# | fatal_incidents_quantile          | Quantile of fatal_incidents (categorical).               |
# | fatal_incidents_ratio             | Ratio of fatal_incidents to n_incidents.                 |
# 

# %%
cities_df = create_cities_df(incidents_df)

# %%
cities_features_list = ['population_state_2010', 'n_incidents', 'n_participants', 'n_weeks_with_incidents',
    'n_killed', 'n_females', 'n_males', 'n_injured', 'n_arrested', 'fatal_incidents', 'fatal_incidents_ratio']

cities_categorical_features_list = ['population_quantile', 'n_incidents_quantile', 'n_weeks_quantile', 'n_participants_quantile',
    'n_killed_quantile', 'n_females_quantile', 'n_males_quantile', 'n_injured_quantile', 'n_arrested_quantile',
    'fatal_incidents_quantile']

# %%
cities_df.head(2)

# %% [markdown]
# We display a description of the features in the dataset:

# %%
cities_df[cities_features_list].describe()

# %%
cities_df[cities_categorical_features_list].describe()

# %% [markdown]
# We compute and visualize the correlation between the features n_incidents, n_participants, n_killed and population_state_2010:

# %%
# make a plot to compare population, n_incidents, n_participants, n_killed,  population, fatal_incidents
fig, ax = plt.subplots(2, 4, figsize=(20, 8))
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
ax[0, 3].scatter(cities_df['fatal_incidents'].values, cities_df['n_participants'].values, 
    c=cities_df['population_quantile'].values)
ax[0, 3].set_xlabel('fatal_incidents')
ax[0, 3].set_ylabel('n_participants')
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
ax[1, 3].scatter(cities_df['fatal_incidents'].values, cities_df['n_killed'].values,
    c=cities_df['population_quantile'].values)
ax[1, 3].set_xlabel('fatal_incidents')
ax[1, 3].set_ylabel('n_killed')
fig.suptitle('Correlation between population and number of incidents, participants and killed')
fig.tight_layout();

# %% [markdown]
# Note that in the scatterplot:
# - Purple represents the first quantile of the population of the state.
# - Blue represents the second quantile.
# - Green represents the third quantile.
# - Yellow represents the fourth quantile.

# %% [markdown]
# We compute correlation values using Kendall correlation coefficient:

# %%
# correlation coefficient
cities_df[['n_incidents', 'n_participants', 'n_killed', 'population_state_2010', 'fatal_incidents']].corr('kendall')

# %% [markdown]
# ## Clustering

# %% [markdown]
# We conduct clustering analysis on time series data utilizing various techniques. 
# 
# Our objective is to group cities with similar temporal patterns by leveraging the created time series and a predefined scoring mechanism. 
# 
# Following the clustering process, we analyze the results and extract motifs and anomalies from the time series data, aiming for a comprehensive understanding and exploration of the udata.

# %% [markdown]
# ### Shape-based clustering
# #### k-means

# %% [markdown]
# We utilize the k-means clustering algorithm on time series data, employing the TimeSeriesScalerMeanVariance object from the tslearn library for scaling. 
# 
# The clustering process is executed through the TimeSeriesKMeans method from the tslearn library, where we utilize the Dynamic Time Warping (DTW) distance as the metric. 
# We set the maximum number of iterations to 100 to optimize the clustering results.

# %% [markdown]
# $DTW(X, Y) = \min_{\pi} \sqrt{\sum_{(i, j) \in \pi} d(x_i, y_j)^2}$
# 
# This equation defines the DTW distance between two time series, \(X\) and \(Y\), where \(\pi\) represents the alignment path, and \(d(x_i, y_j)\) is the local distance between corresponding elements \(x_i\) and \(y_j\). The DTW distance is the minimum cumulative distance along the optimal alignment path.
# 
# Using Dynamic Time Warping (DTW) as a metric requires longer computations compared to using the Euclidean distance, as their time complexity asymptotically differs. The Euclidean distance exhibits linear complexity, while DTW does not.
# 
# However, despite the increased computational cost, preliminary analysis indicates that using DTW yields better results. This justifies our choice to opt for DTW over the Euclidean distance.

# %% [markdown]
# Choose best k:

# %%
X = TimeSeriesScalerMeanVariance().fit_transform(incidents_by_city_df.values) # scale time series

# %%
"""inertia_list = [] # sum of distances of samples to their closest cluster center

for k in range(2, 20):
    km = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=100, random_state=42)
    km.fit(X)
    pred = km.predict(X)
    print("n clusters = ", k, "\t Clusters =", np.unique(pred,return_counts=True)[1], "\t Inertia =", km.inertia_)
    inertia_list.append(km.inertia_)"""

# %%
"""plt.figure(figsize=(20, 5))
plt.plot(inertia_list, '.--')
plt.xticks(range(len(inertia_list)), range(2, 20))
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia for different number of clusters');"""

# %% [markdown]
# Fit chosen model (k=10):

# %%
best_k = 10
km = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=100, random_state=42)
km.fit(X)

# %% [markdown]
# We visualize cluster's centroids:

# %%
plt.figure(figsize=(20, 5))
plt.plot(km.cluster_centers_.reshape(incidents_by_city_df.values.shape[1], best_k), '.--')
plt.title('Centroids of clusters')
plt.legend(range(11), loc='upper left', bbox_to_anchor=(1, 1));

# %%
km.inertia_ # Sum of distances of samples to their closest cluster center

# %%
cluster = km.fit_predict(incidents_by_city_df.values)

# %% [markdown]
# Evaluate clustering:

# %%
results = kmeans_evaluation(X.reshape(X.shape[0], X.shape[1]), km, 
    cluster_centers=km.cluster_centers_.reshape(best_k, incidents_by_city_df.values.shape[1]))

cluster_results_df = pd.DataFrame(results, index=[0])
cluster_results_df

# %% [markdown]
# Visualize clusters

# %%
cluster_df = incidents_df.groupby(['city', 'state'])[['latitude', 'longitude']].mean().reset_index()
cluster_df['cluster_kmeans'] = cluster
cluster_df.head(2)

# %% [markdown]
# We visualize clusters on a map:

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

# %% [markdown]
# Visualize the number of time series in each cluster:

# %%
plt.figure(figsize=(20, 5))
plt.bar(cluster_df.groupby('cluster_kmeans').size().index, cluster_df.groupby('cluster_kmeans').size().values)
for i, v in enumerate(cluster_df.groupby('cluster_kmeans').size().values):
    plt.text(i-0.3, v+1, 'n cities = '+str(v))
plt.title('Number of cities per cluster');

# %% [markdown]
# Visualize average of time series for each clusters:

# %%
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df.groupby(cluster).mean().values.T, '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(incidents_by_city_df.groupby(cluster).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %% [markdown]
# Cluster 5: Higher values in time series indicate more incidents and involve a larger number of people.
# 
# Cluster 9: Lower values observed, likely representing cities with many zeros in the time series, i.e. fewer incidents.
# 
# Cluster 6: Time series in this cluster exhibit high variability with numerous fluctuations.
# 
# Cluster 1: Appears to contain time series with high values, although less pronounced than those in Cluster 5.

# %% [markdown]
# Visualize the time series belonging to Cluster 6 to better understand the high variability (y-axis is in logarithmic scale).

# %%
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df[cluster==6].values.T, '.--')
plt.title('Number of participants per incident per week, cluster 5')
plt.yscale('log')
plt.legend(incidents_by_city_df[cluster==6].index, loc='upper left', bbox_to_anchor=(1, 1));

# %%
cities_cluster_6 = incidents_by_city_df[cluster==6].index.get_level_values('city').to_list()
cities_df[cities_df.index.get_level_values('city').isin(cities_cluster_6)][
    ['n_incidents', 'n_incidents_quantile', 'n_participants', 'n_participants_avg', 'n_killed', 'n_killed_avg',
    'population_state_2010', 'population_quantile']].sort_values('n_incidents', ascending=False)

# %%
print('Averege number of participants per incident in the whole dataset: ', incidents_df['n_participants'].mean())
print('Averege number of killed people per incident in the whole dataset: ', incidents_df['n_killed'].mean())

# %% [markdown]
# We can observe that almost all the time series in Cluster 6 are associated with cities where the average number of participants and killed is higher than the overall dataset average.

# %% [markdown]
# Visualize how the population, the total number of incidents over the four years, and the number of weeks in which at least one accidents is happened in a city are distributed across clusters.

# %%
def plot_categorical_features_per_cluster(labels, feature, feauture_name, ax):
    crosstab = pd.crosstab(labels, feature, rownames=['cluster'], colnames=[feauture_name])
    crosstab.plot.bar(ax=ax, stacked=False, title=f'Total number of {feauture_name} in the city quantile for each cluster')

# %%
plt.subplots(5, 2, figsize=(20, 10))
for i, feature in enumerate(cities_categorical_features_list):
    plot_categorical_features_per_cluster(cluster, cities_df[feature], feature, plt.subplot(5, 2, i+1))
plt.tight_layout();

# %% [markdown]
# The population of the states to which the cities belong appears to be uniformly distributed across the clusters. 
# 
# Consistent with the earlier observation, in Cluster 5, only time series related to cities where the total number of participants in incidents and all the others features in which at least one incident occurred fall into the third quantile of the dataset with all the time series.
# 
# Within Cluster 9, the majority of cities are characterized by the number of participants in incidents and the number of weeks in which at least one incident occurred falling into the first quantile.
# 
# Cluster 1 conteins cities with features similar to cluster 5.

# %% [markdown]
# Performe the same cluster using time series with n_killed in order to compare the results.

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
# From the Sankey plot, we can observe that there is no apparent correlation between the cluster analyses performed on the two distinct time series.

# %%
plt.subplots(5, 2, figsize=(20, 10))
for i, feature in enumerate(cities_categorical_features_list):
    plot_categorical_features_per_cluster(cluster_df['cluster_kmeans_killed'].to_list(), cities_df[feature], 
    feature, plt.subplot(5, 2, i+1))
plt.tight_layout();

# %% [markdown]
# We can observe that clusters 3 and 9 in this analysis contain features belonging to the third quantile. In fact, they correspond to the cities belonging to cluster 5 in the previous analysis.

# %% [markdown]
# ### Compression-based Clustering

# %% [markdown]
# We perform clustering on compressed versions of time series data. 
# The compression is achieved using the following custom dissimilarity measure *cdm_dist*:
# - The values of each time series are concatenated into strings and encoded in UTF-8.
# - The strings are then compressed using zlib compression.
# - The dissimilarity is calculated as the ratio of the compressed size of the concatenated strings to the sum of the compressed sizes of the individual strings.
# 
# After defining the dissimilarity measure, we apply it to calculate pairwise distances between the rows of the reshaped matrix X. The resulting matrix M contains the dissimilarity values and is used for clustering.
# 
# This approach allows us to run clustering on the compressed representations of time series data.

# %%
def cdm_dist(x, y):
    # compounding dissimilarity measure
    x_str = (' '.join([str(v) for v in x.ravel()])).encode('utf-8')
    y_str = (' '.join([str(v) for v in y.ravel()])).encode('utf-8')
    return len(zlib.compress(x_str + y_str)) / (len(zlib.compress(x_str)) + len(zlib.compress(y_str)))

X = incidents_by_city_df.values
M = pairwise_distances(X.reshape(X.shape[0], X.shape[1]), metric=cdm_dist)

# %% [markdown]
# #### Hierarchical Clustering on Compressed Time Series

# %% [markdown]
# We conduct hierarchical clustering on compressed time series data using the AgglomerativeClustering algorithm from scikit-learn.
# 
# Before performing hierarchical clustering, we conduct a preliminary analysis to determine the appropriate linkage method and the suitable number of clusters.
# 
# We evaluate different linkage methods to find the one that best captures the relationships in the data. Common linkage methods include:
# - Single Linkage
# - Complete Linkage
# - Average Linkage
# - Ward Linkage
# 
# To identify the optimal number of clusters, we explore different cluster sizes. The dendrogram inspection was employed to guide the selection of the most suitable number of clusters.

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

# %% [markdown]
# Visualize average of time series for each clusters:

# %%
plt.figure(figsize=(20, 5))
for i in range(4):
    plt.plot(np.mean(X[np.where(hier.labels_ == i)[0]], axis=0), '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(incidents_by_city_df.groupby(hier.labels_).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %% [markdown]
# Evalueate clusters:

# %%
results = hierarchical_evaluation(X.reshape(X.shape[0], X.shape[1]), hier)

cluster_results_df = pd.concat([cluster_results_df, pd.DataFrame(results, index=[1])])
cluster_results_df

# %%
cluster_df['cluster_hierarchical4'] = hier.labels_
cluster_df.head(2)

# %% [markdown]
# We visualize clusters on a map:

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

# %% [markdown]
# Visualize the number of time series in each cluster:

# %%
plt.figure(figsize=(20, 5))
plt.bar(cluster_df.groupby('cluster_hierarchical4').size().index, cluster_df.groupby('cluster_hierarchical4').size().values)
for i, v in enumerate(cluster_df.groupby('cluster_hierarchical4').size().values):
    plt.text(i-0.1, v+1, 'n cities = '+str(v))
plt.title('Number of cities per cluster');

# %% [markdown]
# As observed, the majority of cities are clustered into Cluster 1. This clustering is highly imbalanced, with nearly all cities assigned to this single cluster. Such a clustering result may be considered ineffective or uninformative due to its lack of diversity across clusters

# %% [markdown]
# Use n_clusters = 6:

# %%
hier = AgglomerativeClustering(n_clusters=6, linkage='average')
hier.fit(M)

# %% [markdown]
# Visualize average of time series for each clusters:

# %%
plt.figure(figsize=(20, 5))
for i in range(6):
    plt.plot(np.mean(X[np.where(hier.labels_ == i)[0]], axis=0), '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(incidents_by_city_df.groupby(hier.labels_).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %%
results = hierarchical_evaluation(X.reshape(X.shape[0], X.shape[1]), hier)

cluster_results_df = pd.concat([cluster_results_df, pd.DataFrame(results, index=[2])])
cluster_results_df

# %%
cluster_df['cluster_hierarchical6'] = hier.labels_
cluster_df.head(2)

# %% [markdown]
# We visualize clusters on a map:

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

# %% [markdown]
# Visualize the number of time series in each cluster:

# %%
plt.figure(figsize=(20, 5))
plt.bar(cluster_df.groupby('cluster_hierarchical6').size().index, cluster_df.groupby('cluster_hierarchical6').size().values)
for i, v in enumerate(cluster_df.groupby('cluster_hierarchical6').size().values):
    plt.text(i-0.2, v+1, 'n cities = '+str(v))
plt.title('Number of cities per cluster');

# %% [markdown]
# Visualize how the features relative at the cities over the four years are distributed across clusters.

# %%
plt.subplots(5, 2, figsize=(20, 10))
for i, feature in enumerate(cities_categorical_features_list):
    plot_categorical_features_per_cluster(hier.labels_, cities_df[feature], feature, plt.subplot(5, 2, i+1))
plt.tight_layout();

# %% [markdown]
# In clusters 0, 1, and 4, there are cities where the number features falls within the third quartile concerning the total weeks in which incidents occurred in the dataset.
# 
# Cluster 2, the most populated ones, containe features distribuited in the first three quantiles.
# 
# Only 4 cities belong to Cluster 3, and all their features are in the first quantile.
# 
# In Cluster 5, there are cities with features that mostly fall into the second and third quantiles.

# %% [markdown]
# Performe the same cluster using time series with n_killed in order to compare the results.

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
# 
# We use PiecewiseAggregateApproximation from tslearn libraty.

# %%
n_paa_segments = 100
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
X_paa = paa.fit_transform(incidents_by_city_df.values) # PAA transformation

# %% [markdown]
# Visualize the PAA representation of time series:

# %%
plt.figure(figsize=(20, 5))
plt.plot(X_paa.reshape(X_paa.shape[1], X_paa.shape[0]))
plt.title('PAA representation of time series');

# %% [markdown]
# Choose best k:

# %%
"""inertia_list = [] # sum of distances of samples to their closest cluster center

for k in range(2, 20):
    km = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=100, random_state=42)
    km.fit(X_paa)
    pred = km.predict(X_paa)
    print("n clusters = ", k, "\t Clusters =", np.unique(pred,return_counts=True)[1], "\t Inertia =", km.inertia_)
    inertia_list.append(km.inertia_)"""

# %%
"""plt.figure(figsize=(20, 5))
plt.plot(inertia_list, '.--')
plt.xticks(range(len(inertia_list)), range(2, 20))
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia for different number of clusters');"""

# %% [markdown]
# Performe k-means clusternig on the PAA representation of time series, unsing the same paramters that in the previously experiment.

# %%
best_k = 9

# %%
km = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=100, random_state=42)
km.fit(X_paa)

# %%
plt.figure(figsize=(20, 5))
plt.plot(km.cluster_centers_.reshape(X_paa.shape[1], best_k))
plt.title('Centroids of clusters')
plt.legend(range(11), loc='upper left', bbox_to_anchor=(1, 1));

# %%
plt.figure(figsize=(20, 5))
for i in range(best_k):
    plt.plot(np.mean(X[np.where(km.labels_ == i)[0]], axis=0), '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
#plt.yscale('log')
plt.legend(incidents_by_city_df.groupby(km.labels_).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %%
results = kmeans_evaluation(X_paa.reshape(X_paa.shape[0], X_paa.shape[1]), km, 
    cluster_centers=km.cluster_centers_.reshape(best_k, X_paa.shape[1]))

cluster_results_df = pd.concat([cluster_results_df, pd.DataFrame(results, index=[3])])
cluster_results_df

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
plt.bar(cluster_df.groupby('cluster_kmeans_paa').size().index, cluster_df.groupby('cluster_kmeans_paa').size().values)
for i, v in enumerate(cluster_df.groupby('cluster_kmeans_paa').size().values):
    plt.text(i-0.3, v+1, 'n cities = '+str(v))
plt.title('Number of cities per cluster');

# %%
plt.subplots(5, 2, figsize=(20, 10))
for i, feature in enumerate(cities_categorical_features_list):
    plot_categorical_features_per_cluster(km.labels_, cities_df[feature], feature, plt.subplot(5, 2, i+1))
plt.tight_layout();

# %% [markdown]
# No cluster seems to exhibit a distinct pattern relative to the population.
# 
# Clusters 1, 2, 3, and 6 predominantly exhibit features belonging to the third quantile.
# 
# Clusters 4 and 0 have features distributed across the first three quantiles. Notably, Cluster 4, the most populous cluster, predominantly contains features from the first quantile.

# %%
plot_timeseries_per_cluster(X, km.labels_)

# %%
paa = PiecewiseAggregateApproximation(n_segments=100)
X_paa = paa.fit_transform(incidents_killed_by_city_df.values) # PAA transformation
km = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=5, random_state=0)
km.fit(X_paa)
cluster_df['cluster_kmeans_paa_killed'] = km.labels_

# %%
sankey_plot(
    [cluster_df['cluster_kmeans_paa'], cluster_df['cluster_kmeans_paa_killed']],
    labels_titles=['n_participants', 'n_killed'],
    title='Clusterings comparison KMeansTimeSeries PAA'
)

# %% [markdown]
# No discernible pattern emerges between the clusters formed using time series generated based on the number of participants and those formed using the number of killed incidents.

# %% [markdown]
# ### Features-based Clustering

# %% [markdown]
# #### K-mean

# %% [markdown]
# Create list of features conteined statistical values for each time series:

# %%
def calculate_features(values):
    features = {
        'avg': np.mean(values),
        'std': np.std(values),
        'var': np.var(values),
        'med': np.median(values),
        '10p': np.percentile(values, 10),
        '25p': np.percentile(values, 25),
        '50p': np.percentile(values, 50),
        '75p': np.percentile(values, 75),
        '90p': np.percentile(values, 90),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'cov': 1.0 * np.mean(values) / np.std(values),
        'skw': stats.skew(values),
        'kur': stats.kurtosis(values)
    }

    return features

# %%
X = incidents_by_city_df.values
time_series_features_list = [list(calculate_features(x).values())[:-2] for x in X]

# %%
inertia_list = [] # sum of distances of samples to their closest cluster center

for k in range(2, 30):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    pred = kmeans.predict(X)
    print("n clusters = ", k, "\t Clusters =", np.unique(pred,return_counts=True)[1], "\t Inertia =", kmeans.inertia_)
    inertia_list.append(kmeans.inertia_)

# %%
plt.figure(figsize=(20, 5))
plt.plot(inertia_list, '.--')
plt.xticks(range(len(inertia_list)), range(2, 30))
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia for different number of clusters');

# %% [markdown]
# Apply clustering algorithm to the statistical features list:

# %%
best_k=11

# %%
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
kmeans.fit(time_series_features_list)

# %%
plt.figure(figsize=(20, 5))
for i in range(best_k):
    plt.plot(np.mean(X[np.where(kmeans.labels_ == i)[0]], axis=0), '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(range(best_k), loc='upper left', bbox_to_anchor=(1, 1));
plt.yscale('log')
plt.show()

# %%
results = kmeans_evaluation(np.array(time_series_features_list), kmeans, cluster_centers=kmeans.cluster_centers_)

cluster_results_df = pd.concat([cluster_results_df, pd.DataFrame(results, index=[4])])
cluster_results_df

# %%
cluster_df['cluster_kmeans_features'] = kmeans.labels_
cluster_df.head(2)

# %%
fig = px.scatter_mapbox(
    lat=cluster_df['latitude'],
    lon=cluster_df['longitude'],
    zoom=2, 
    color=cluster_df['cluster_kmeans_features'],
    height=400,
    width=1000,
    text=cluster_df['city'] + ', ' + cluster_df['state']
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %%
plt.figure(figsize=(20, 5))
plt.bar(cluster_df.groupby('cluster_kmeans_features').size().index, cluster_df.groupby('cluster_kmeans_features').size().values)
for i, v in enumerate(cluster_df.groupby('cluster_kmeans_features').size().values):
    plt.text(i-0.3, v+1, 'n cities = '+str(v))
plt.title('Number of cities per cluster');

# %%
plt.figure(figsize=(20, 3))
plt.plot(X[kmeans.labels_==8].T, '.--')
plt.title(incidents_by_city_df[kmeans.labels_==8].index.get_level_values('city').values[0]+' time serie, cluster 8')
plt.yscale('log');

plt.figure(figsize=(20, 3))
plt.plot(X[kmeans.labels_==9].T, '.--')
plt.title(incidents_by_city_df[kmeans.labels_==9].index.get_level_values('city').values[0]+' time serie, cluster 9');

# %%
incidents_by_city_df[kmeans.labels_==8].index

# %%
incidents_by_city_df[kmeans.labels_==9].index

# %%
plt.subplots(5, 2, figsize=(20, 10))
for i, feature in enumerate(cities_categorical_features_list):
    plot_categorical_features_per_cluster(kmeans.labels_, cities_df[feature], feature, plt.subplot(5, 2, i+1))
plt.tight_layout();

# %%
plt.figure(figsize=(10, 7))
plt.plot(kmeans.cluster_centers_.T, '.-')
plt.title('Centroids of clusters')
plt.xticks(range(len(time_series_features_list[0])), list(calculate_features(X[0]).keys())[:-2], rotation=90)
plt.grid()
plt.legend(range(best_k), loc='upper left', bbox_to_anchor=(1, 1));

# %%
# time series without zeros
incidents_by_city_df[np.count_nonzero(incidents_by_city_df.values, axis=1) == 209].index

# %%
# print cluster for 
list_index = [('Chicago','ILLINOIS'), ('City of New York','NEW YORK'),
    ('Columbus','OHIO'), ('Milwaukee','WISCONSIN'), ('New Orleans','LOUISIANA')]
for index in list_index:
    print(index, kmeans.labels_[incidents_by_city_df.index == index][0])

# %%
# plot time series for list_index

for index in list_index:
    plt.figure(figsize=(20, 3))
    plt.plot(incidents_by_city_df[incidents_by_city_df.index == index].values.T, '.--')
    plt.title(index[0]+' ('+index[1]+') time serie, cluster '+str(kmeans.labels_[incidents_by_city_df.index == index][0]))

# %% [markdown]
# Extrapolate incidents correspondig to peak data for San bernardino time series:

# %%
def week_to_date(week):
    return datetime.datetime.strptime('2013-12-30', '%Y-%m-%d') + datetime.timedelta(weeks=week)

def date_to_week(date):
    return (pd.to_datetime(date) - pd.to_datetime('2013-12-30')).days // 7

# %%
week = np.where(np.max(incidents_by_city_df[kmeans.labels_==9].values.T) == incidents_by_city_df[kmeans.labels_==9].values.T)[0][0]
start_date = week_to_date(int(week))
end_date = start_date + 6*datetime.timedelta(days=1)
print('Peak week: ', week, 'corresponding to ', start_date, ' - ', end_date)

# %%
incidents_df[(incidents_df['city'] == 'San Bernardino') & (incidents_df['state'] == 'CALIFORNIA') &
    (incidents_df['date'] >= start_date) & (incidents_df['date'] <= end_date)][['date', 'n_participants', 'n_killed',
    'n_injured', 'n_arrested', 'n_unharmed', 'avg_age_participants', 'incident_characteristics1', 'incident_characteristics2']]

# %% [markdown]
# [San Bernardino attack](https://en.wikipedia.org/wiki/2015_San_Bernardino_attack)
# *"On December 2, 2015, a terrorist attack, consisting of a mass shooting and an attempted bombing, occurred at the Inland Regional Center in San Bernardino, California. The perpetrators, Syed Rizwan Farook and Tashfeen Malik, a married couple living in the city of Redlands."*

# %% [markdown]
# [North Park Elementary School shooting](https://en.wikipedia.org/wiki/2017_North_Park_Elementary_School_shooting)
# *"On April 10, 2017, a shooting occurred inside a special education classroom at North Park Elementary School in San Bernardino, California.
# The shooting followed a terrorist attack that occurred in San Bernardino in December 2015."*

# %%
incidents_df[(incidents_df['city'] == 'San Bernardino') & (incidents_df['state'] == 'CALIFORNIA') & 
    (incidents_df['date'] == '2017-04-10')][['date', 'n_participants', 'n_killed',
    'n_injured', 'n_arrested', 'n_unharmed', 'avg_age_participants', 'incident_characteristics1', 'incident_characteristics2']]

# %%
# cluster of incidents incidents_df[(incidents_df['city'] == 'San Bernardino') & (incidents_df['state'] == 'CALIFORNIA') & (incidents_df['date'] == '2017-04-10')]
week_north_parck_attack = date_to_week('2017-04-10')
week_north_parck_attack

# %% [markdown]
# Cardinality of non zero values for time series:

# %%
print('Cardinality of non zero values for time series in cluster 6: ',
    np.count_nonzero(incidents_by_city_df[kmeans.labels_==6].values, axis=1))
print('Cardinality of non zero values for time series in cluster 0: ',
    np.count_nonzero(incidents_by_city_df[kmeans.labels_==0].values, axis=1))

# %%
print('Cardinality of non zero values for time series in cluster 2: ',
    np.count_nonzero(incidents_by_city_df[kmeans.labels_==2].values, axis=1))
print('Mean cardinality of non zero values for time series in cluster 2: ',
    np.mean(np.count_nonzero(incidents_by_city_df[kmeans.labels_==2].values, axis=1)))
print('Cardinality of non zero values for time series in cluster 7: ',
    np.count_nonzero(incidents_by_city_df[kmeans.labels_==7].values, axis=1))
print('Mean cardinality of non zero values for time series in cluster 7: ',
    np.mean(np.count_nonzero(incidents_by_city_df[kmeans.labels_==7].values, axis=1)))

# %%
# plot all the time series for each cluster
color = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:green', 'tab:purple', 'tab:orange', 'k', 'tab:pink']
for i in range(best_k):
    plt.figure(figsize=(20, 2))
    plt.plot(incidents_by_city_df[kmeans.labels_==i].values.T, 'b', linewidth=0.5)
    plt.title('Time series for cluster '+str(i));


# %% [markdown]
# Clustering results:
# 
# There is no discernible pattern in cluster division based on population size.
# 
# Clusters 0, 6, 8, 9, and 10 are primarily composed of cities exhibiting features (total number of incidents, number of week in which at least one incidents is present, sum of the number of killed person, females, males injured person and arrested during all the incidents occured in a city) predominantly situated in the third quantile. The mean values of their respective time series consistently surpass others across all time points, signifying time series characterized by fewer zero values. In essence, these clusters encapsulate cities where incidents occur in most weeks.
# 
# Clusters 8 and 9 each consist of a single city: Chicago (ILLINOIS) and San Bernardino (CALIFORNIA), respectively. Despite attempts to reduce the number of clusters to 9 and eliminate the cluster with only one city, this adjustment proved unsuccessful, as the two clusters with a singular city persisted, and larger clusters absorbed them.
# 
# San Bernardino time serie displays a peak in its time series (value=37), representing the maximum number of participants in incidents within a specific week in any city throughout the entire dataset. Excluding this value, the time series exhibits a significant number of null values (94 out of 209). The mean of its values is 1.125. If we disregard the value 37, the series would share characteristics similar to those belonging to Cluster 4.
# 
# Chicago stands out with a time series entirely devoid of null values. The only other cities sharing this characteristic and their respective clusters are as follows: City of New York (NEW YORK), Columbus (OHIO), Milwaukee (WISCONSIN), and New Orleans (LOUISIANA), all this cities are clustering into Cluster 6.
# 
# Chicago's time series is more regular, with values ranging between 1.2 and 2.2, presenting a relatively constant trend over time. In contrast, time series in Cluster 6 exhibit a more irregular pattern, covering a broader range of values (1-4.5).
# 
# In contrast, time series in Cluster 6 exhibit a more irregular pattern, covering a broader range of values (1-4.5).
# Cluster 6 encompasses 17 time series, characterized by a range in the cardinality of non-zero value between 201 to 209 out of 209.
# Cluster 0 contains time series with characteristics similar to those in Cluster 6. However, the number of non-zero values is slightly lower, ranging between 201 to 209 out of 209.
# 
# Clusters 2 and 7 consist of cities associated with features primarily falling above the second quantile. With the exception of the number of fatal incidents, these cities experience a higher-than-average number of incidents throughout the years, with a higher number of participants per incident. However, the average number of incidents with at least one fatality is lower on average compared to cities clustered in Clusters 0, 6, 8, and 10.
# The main distinction between the two clusters lies in the number of non-zero values. For Cluster 2, this number ranges between 89 to 104, with a mean of 94.054. On the other hand, for Cluster 7, it ranges between 107 to 134 out of 209, with a mean of 118.731.
# 
# Clusters 1 and 4, the second and first most populated clusters, predominantly include cities with features mostly belonging to the first two quantiles. Specifically, in Cluster 4, cities mostly have features in the first quantile, indicating time series associated with many null values.
# 
# Cluster 3 contains features predominantly in the third quantile, and it is the third-largest cluster.
# 
# Cluster 5 features predominantly fall within the second and third quantiles, making it the third-largest cluster.

# %% [markdown]
# ### Clustering comparison

# %%
sankey_plot(
    [cluster_df['cluster_kmeans'], cluster_df['cluster_kmeans_paa'], cluster_df['cluster_hierarchical6'], 
    cluster_df['cluster_hierarchical4'], cluster_df['cluster_kmeans_features']],
    labels_titles=['TimeSeriesKMeans Shape-based', 'TimeSeriesKMeans PAA', 'Hierarchical Compression-based', 
    'Hierarchical Compression-based', 'KMeans Features-based'],
    title='Clusterings comparison'
)

# %%
cluster_results_df

# %% [markdown]
# **Shape-based Clustering (K-means):**
# 
# In the results obtained from shape-based clustering using k-means, we observe distinct patterns across different clusters:
# 
# - **Cluster 5:** Time series with higher values, indicating more incidents and involving a larger number of people.
# - **Cluster 9:** Lower values, likely representing cities with many zeros in the time series, i.e., fewer incidents.
# - **Cluster 6:** High variability with numerous fluctuations in time series.
# - **Cluster 1:** Contains time series with high values, though less pronounced than those in Cluster 5.
# 
# The population distribution of the states to which the cities belong seems uniformly distributed across the clusters. Notably, in Cluster 5, time series relate to cities where various features, including the total number of participants in incidents, fall into the third quantile of the dataset. Within Cluster 9, the majority of cities exhibit characteristics falling into the first quantile.
# 
# **Hierarchical Clustering on Compressed Time Series:**
# 
# The hierarchical clustering algorithm yields different results when clustering similar time series obtained using n_killed. When using 4 clusters, an imbalanced result is observed, with the majority of cities assigned to Cluster 1, indicating a lack of diversity. With 6 clusters, the distribution improves, and specific patterns emerge:
# 
# - **Clusters 0, 1, and 4:** Cities where features fall within the third quartile concerning the total weeks of incidents in the dataset.
# - **Cluster 2:** The most populated cluster, with features distributed across the first three quantiles.
# - **Cluster 5**: Cities with features that mostly fall into the second and third quantiles.
# 
# This is the only clustering algorithm in which the results reveal a pattern between the clusters formed using time series generated based on the number of participants and those formed using the number of killed incidents.
# 
# **K-means using Piecewise Aggregate Approximation (PAA) of Time Series:**
# 
# The k-means clustering based on PAA does not reveal a distinct pattern relative to the population. Key observations include:
# 
# - **Clusters 1, 2, and 6** predominantly exhibit features belonging to the third quantile.
# - **Clusters 4 and 0** have features distributed across the first three quantiles, with Cluster 4, the most populous, containing predominantly first quantile features.
# - Only 4 cities belong to **Cluster 3**, and all their features fall into the first quantile.
# 
# **Features-based Clustering (K-means):**
# This clustering algorithm performs relatively better. See before for more comment.

# %% [markdown]
# ## Matrix profile

# %% [markdown]
# The matrix profile is constructed by combining distance profiles for all possible subsequence pairs in the time series. It highlights repeated patterns or motifs by identifying low values in the matrix profile.
# 
# We use matrixprofile-ts library for evaluating time series data using the Matrix Profile algorithms.
# 
# The Matrix Profile value jumps at each phase change. High Matrix Profile values are associated with "discords": time series behavior that hasn't been observed before.
# Repeated patterns in the data (or "motifs") lead to low Matrix Profile values.

# %% [markdown]
# Compute and visualize matrix profile (a data structure that annotates time series by using a sliding window to compare the pairwise distance among the subsequences) for New York, Los Angeles and Chicago:

# %%
# estrapolate time series for new york, los angeles and chicago
new_york_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'City of New York')].values[0]
los_angeles_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'Los Angeles')].values[0]
chicago_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'Chicago')].values[0]

w_list = [8, 13, 26, 52, 104] # length of subsequence to compare: two month, quarter, half year, year, two years

fig, ax = plt.subplots(6, 1, figsize=(18, 12))
ax[0].plot(new_york_ts, '.--', label='New York')
ax[0].plot(los_angeles_ts, '.--', label='Los Angeles')
ax[0].plot(chicago_ts, '.--', label='Chicago')
ax[0].set_ylabel('Time series', size=10)
ax[0].axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
ax[0].axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
ax[0].axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
ax[0].legend()

for i, window in enumerate(w_list):
    mp, mpi = matrixProfile.stomp(new_york_ts, window)
    ax[i+1].plot(mp, '.--')
    mp, mpi = matrixProfile.stomp(los_angeles_ts, window)
    ax[i+1].plot(mp, '.--')
    mp, mpi = matrixProfile.stomp(chicago_ts, window)
    ax[i+1].plot(mp, '.--')
    ax[i+1].set_ylabel('Matrix Profile, w = ' + str(window), size=10)

# %% [markdown]
# Matrix profile for the whole dataset:

# %%
w_list = [8, 13, 26, 52] # length of subsequence to compare: two month, quarter, half year, year

fig, ax = plt.subplots(5, 1, figsize=(18, 12))
ax[0].plot(incidents_by_city_df.mean().values, '.--', label='Averege all time series')
ax[0].plot(incidents_killed_by_city_df.mean().values, '.--', label='Averege all n_killed  time series')
ax[0].set_ylabel('Time series', size=10)
ax[0].axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
ax[0].axvline(x=n_weeks_per_year*2-1, color='k', linestyle='--')
ax[0].axvline(x=n_weeks_per_year*3-1, color='k', linestyle='--')
ax[0].legend()

for i, window in enumerate(w_list):
    mp, mpi = matrixProfile.stomp(incidents_by_city_df.mean().values, window)
    ax[i+1].plot(mp, '.--')
    mp, mpi = matrixProfile.stomp(incidents_killed_by_city_df.mean().values, window)
    ax[i+1].plot(mp, '.--')
    ax[i+1].set_ylabel('Matrix Profile, w = ' + str(window), size=10)

# %% [markdown]
# Following the approach descrive by Ermshaus et al. in [Window Size Selection In Unsupervised
# Time Series Analytics: A Review and Benchmark](https://project.inria.fr/aaltd22/files/2022/08/AALTD22_paper_3876.pdf) we find optimal window size using the Highest Autocorrelation methon from claspy library

# %% [markdown]
# Display the optimale window size for New York, Los Angeles, Chicago and mean of all time series:

# %%
print('Optimal window size for New York time series: ', highest_autocorrelation(new_york_ts))
print('Optimal window size for Los Angeles time series: ', highest_autocorrelation(los_angeles_ts))
print('Optimal window size for Chicago time series: ', highest_autocorrelation(chicago_ts))

# %%
print('Optimal window size for mean of all the time series: ', highest_autocorrelation(incidents_by_city_df.mean().values))

# %% [markdown]
# Compute optimal window size for the whole dataset:

# %%
optimal_window_size_list = []
for ts in incidents_by_city_df.values:
    optimal_window_size_list.append(highest_autocorrelation(ts))

# %% [markdown]
# Visualize the distribution of optimal window sizes:

# %%
plt.figure(figsize=(20, 3))
plt.hist(optimal_window_size_list, bins=100, density=True, alpha=0.8, edgecolor='k')
plt.title('Distribution of optimal window size');

# %%
print('Mean of optimal window size: ', np.mean(optimal_window_size_list))
print('Median of optimal window size: ', np.median(optimal_window_size_list))
print('Standard deviation of optimal window size: ', np.std(optimal_window_size_list))
print('Min of optimal window size: ', np.min(optimal_window_size_list))
print('Max of optimal window size: ', np.max(optimal_window_size_list))

# %%
cluster_df['optimal_window_size'] = optimal_window_size_list

# %%
cluster_df[cluster_df['optimal_window_size']==10][['city', 'state', 'cluster_kmeans_features']]

# %%
cluster_df[cluster_df['optimal_window_size']==103][['city', 'state', 'cluster_kmeans_features']]

# %% [markdown]
# Visualize matrix profile with optimal window for some time series:

# %%
def matrix_profile_plot(ts, window, city):
    mp, mpi = matrixProfile.stomp(ts, window)
    plt.figure(figsize=(20, 3))
    plt.plot(ts, '.--', label='Time series')
    plt.plot(mp, '.--', label='Matrix Profile')
    plt.title(f'Matrix Profile, w = {str(window)}, {city}')
    plt.ylabel('Matrix Profile')
    plt.legend()
    plt.tight_layout();

# %%
matrix_profile_plot(new_york_ts, window=highest_autocorrelation(new_york_ts), city='New York')
matrix_profile_plot(los_angeles_ts, window=highest_autocorrelation(los_angeles_ts), city='Los Angeles')
matrix_profile_plot(chicago_ts, window=highest_autocorrelation(chicago_ts), city='Chicago')

# %% [markdown]
# ##  Motifs extraction

# %% [markdown]
# Motifs: *"a set of frequently appearing and similar subsequence"*
# 
# Our aim is to finding motif sets calculate anomalies as out of distribution values of the matrix profile.

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

# %% [markdown]
# #### Chicago time serie

# %%
window = highest_autocorrelation(chicago_ts)
mp, mpi = matrixProfile.stomp(chicago_ts, window)

# %%
motifs_idx, distance  = motifs.motifs(chicago_ts, (mp, mpi), max_motifs=5)

# %%
plt.figure(figsize=(20, 3))
plt.plot(chicago_ts)
colors = ['r', 'g', 'k', 'b', 'y'][:len(motifs_idx)]
for idx, dist, color in zip(motifs_idx, distance, colors):
    for i in idx:
        m_shape = chicago_ts[i:i+window]
        plt.plot(range(i,i+window), m_shape, color=color, lw=3)
plt.title('Motifs in Chicago time series');

# %%
fig, ax = plt.subplots(len(motifs_idx), 1, figsize=(18, 6))
for j, (idx, dist, color) in enumerate(zip(motifs_idx, distance, colors)):
    for i in idx:
        m_shape = chicago_ts[i:i+window]
        ax[j].plot(range(i,i+window), m_shape, color=color, lw=3)
        ax[j].set_ylabel('Motif ' + str(j+1), size=10)
plt.suptitle('Motifs in Chicago time series')
plt.tight_layout();

# %%
# print week of the year for each motif
print('Motifs in Chicago time series, window size = ', window, 'weeks')
chicago_motifs_df = pd.DataFrame(columns=['motif number', 'week start', 'week end', 'date start', 'date end'])
n = 1
for idx, dist in zip(motifs_idx, distance):
    for i in idx:
        date_start = week_to_date(int(i))
        end_date = week_to_date(int(i+window)) + 6*datetime.timedelta(days=1)
        chicago_motifs_df = pd.concat([chicago_motifs_df, pd.DataFrame([[n, i, i+window, date_start, end_date]], 
            columns=['motif number', 'week start', 'week end', 'date start', 'date end'])])
    n += 1
chicago_motifs_df

# %% [markdown]
# #### San bernardino time serie

# %%
san_bernardino_ts = incidents_by_city_df[(incidents_by_city_df.index == ('San Bernardino', 'CALIFORNIA'))].values[0]
window = highest_autocorrelation(san_bernardino_ts)
mp, mpi = matrixProfile.stomp(san_bernardino_ts, window)
motifs_idx, distance  = motifs.motifs(san_bernardino_ts, (mp, mpi), max_motifs=5)

plt.figure(figsize=(20, 3))
plt.plot(san_bernardino_ts)
colors = ['r', 'g', 'k', 'b', 'y'][:len(motifs_idx)]
for idx, dist, color in zip(motifs_idx, distance, colors):
    for i in idx:
        m_shape = san_bernardino_ts[i:i+window]
        plt.plot(range(i,i+window), m_shape, color=color, lw=3)
plt.title('Motifs in San Bernardino time series');

# %%
plt.figure(figsize=(20, 3))
for idx, dist, color in zip(motifs_idx, distance, colors):
    for i in idx:
        m_shape = san_bernardino_ts[i:i+window]
        plt.plot(range(i,i+window), m_shape, color=color, lw=3)
plt.title('Motifs in San Bernardino time series')
plt.tight_layout();

# %%
# print week of the year for each motif
print('Motifs in San Bernardino time series, window size = ', window, 'weeks')
san_bernardino_motifs_df = pd.DataFrame(columns=['motif number', 'week start', 'week end', 'date start', 'date end'])
n = 1
for idx, dist in zip(motifs_idx, distance):
    for i in idx:
        date_start = week_to_date(int(i))
        end_date = week_to_date(int(i+window)) + 6*datetime.timedelta(days=1)
        san_bernardino_motifs_df = pd.concat([san_bernardino_motifs_df, pd.DataFrame([[n, i, i+window, date_start, end_date]], 
            columns=['motif number', 'week start', 'week end', 'date start', 'date end'])])
    n += 1
san_bernardino_motifs_df

# %% [markdown]
# #### Motifs extraction for clusters

# %% [markdown]
# Motifs extraction for mean time series in the same cluster:

# %%
# plot in the same figure motifs for mean time series in the same cluster
def plot_motifs_per_cluster(time_series, cluster, window):
    plt.figure(figsize=(20, 3))
    ts = np.mean(time_series, axis=0)
    window = highest_autocorrelation(ts)
    mp, mpi = matrixProfile.stomp(ts, window)
    motifs_idx, distance  = motifs.motifs(ts, (mp, mpi), max_motifs=5)
    plt.plot(ts)
    colors = ['r', 'g', 'k', 'b', 'y'][:len(motifs_idx)]
    for idx, dist, color in zip(motifs_idx, distance, colors):
        for i in idx:
            m_shape = ts[i:i+window]
            plt.plot(range(i,i+window), m_shape, color=color)
    plt.title('Motifs in cluster '+str(cluster));

# %%
for cluster in range(best_k):
    time_series = incidents_by_city_df[cluster_df['cluster_kmeans_features'].values==cluster].values
    plot_motifs_per_cluster(time_series, cluster, window)

# %%
for ts, w, city in zip(incidents_by_city_df.values[104:108], 
        optimal_window_size_list[108:113], 
        cities_df.index.get_level_values('city').to_list()[108:113]):
    mp, mpi = matrixProfile.stomp(ts, w)
    motifs_idx, distance  = motifs.motifs(ts, (mp, mpi), max_motifs=5)
    plt.figure(figsize=(20, 3))
    plt.plot(ts)
    colors = ['r', 'g', 'k', 'b', 'y'][:len(motifs_idx)]
    for idx, dist, color in zip(motifs_idx, distance, colors):
        for i in idx:
            m_shape = ts[i:i+w]
            plt.plot(range(i,i+w), m_shape, color=color)
    plt.title(f'Motifs in {city} time series')
    plt.tight_layout();

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
window = highest_autocorrelation(chicago_ts)
mp, mpi = matrixProfile.stomp(chicago_ts, window)
anoms = discords(mp, ex_zone=10, k=10)

# %%
plt.figure(figsize=(20, 3))
plt.plot(chicago_ts)
colors = ['r', 'g', 'k', 'b', 'y'][:len(motifs_idx)]
for a, color in zip(anoms, colors):
    a_shape = chicago_ts[a:a+window]
    plt.plot(range(a, a+window), a_shape, color=color)
plt.title('Anomalies in Chicago time series');

# %%
# creade df with anomalies statistics
anomalies_df = pd.DataFrame(columns=['', 'average', 'std', 'median', 'min', 'max'])
anomalies_df = anomalies_df.append({
    '': 'Chicago time series anomalies',
    'average': np.mean(chicago_ts[anoms]),
    'std': np.std(chicago_ts[anoms]),
    'median': np.median(chicago_ts[anoms]),
    'min': np.min(chicago_ts[anoms]),
    'max': np.max(chicago_ts[anoms])}, ignore_index=True)
anomalies_df = anomalies_df.append({
    '': 'Chicago time series',
    'average': np.mean(chicago_ts),
    'std': np.std(chicago_ts),
    'median': np.median(chicago_ts),
    'min': np.min(chicago_ts),
    'max': np.max(chicago_ts)}, ignore_index=True)

# %%
for anomaly in anoms:
    print('week: ', anomaly+window, '\t value: ', chicago_ts[anomaly])
anomalies_df

# %%
window = highest_autocorrelation(san_bernardino_ts)
mp, mpi = matrixProfile.stomp(san_bernardino_ts, window)
anoms = discords(mp, ex_zone=10, k=10)

plt.figure(figsize=(20, 3))
plt.plot(san_bernardino_ts)
colors = ['r', 'g', 'k', 'b', 'y'][:len(motifs_idx)]
window = highest_autocorrelation(san_bernardino_ts)
for a, color in zip(anoms, colors):
    a_shape = san_bernardino_ts[a:a+window]
    plt.plot(range(a, a+window), a_shape, color=color)
plt.title('Anomalies in San Bernardino time series');

# %%
sb_anomalies_df = pd.DataFrame(columns=['', 'average', 'std', 'median', 'min', 'max'])
sb_anomalies_df = sb_anomalies_df.append({
    '': 'San Bernardino time series anomalies',
    'average': np.mean(san_bernardino_ts[anoms]),
    'std': np.std(san_bernardino_ts[anoms]),
    'median': np.median(san_bernardino_ts[anoms]),
    'min': np.min(san_bernardino_ts[anoms]),
    'max': np.max(san_bernardino_ts[anoms])}, ignore_index=True)
sb_anomalies_df = sb_anomalies_df.append({
    '': 'San Bernardino time series',
    'average': np.mean(san_bernardino_ts),
    'std': np.std(san_bernardino_ts),
    'median': np.median(san_bernardino_ts),
    'min': np.min(san_bernardino_ts),
    'max': np.max(san_bernardino_ts)}, ignore_index=True)

# %%
for anomaly in anoms:
    print('week: ', anomaly+window, '\t value: ', san_bernardino_ts[anomaly])
sb_anomalies_df

# %%
print('Cardinality zero values for San Bernardino time serie: ', np.count_nonzero(san_bernardino_ts == 0))

# %%
san_bernardino_motifs_df

# %% [markdown]
# ## Shaplet discovery

# %% [markdown]
# Create binary var is_killed:

# %%
cities_df['fatal_incidents_ratio'].describe()

# %% [markdown]
# We define the binary variable is_killed for each city time series.
# 
# A time series related to a city is labeled as True if the ratio of fatal incidents in the city between 2014 and 2017 is above the second quantile of the distribuition of fatal_incidents_ratio related to all time series; otherwise, it is labeled as False.

# %%
is_killed = cities_df['fatal_incidents_ratio'].apply(lambda x: 1 if x > np.quantile(cities_df['fatal_incidents_ratio'], 0.5) else 0)

# %%
print('Number of cities with n_killed over second quantile: ', np.sum(is_killed))
print('Number of cities with n_killed in first or second quantile: ', len(is_killed) - np.sum(is_killed))

# %% [markdown]
# Split train and test set:

# %%
X = incidents_by_city_df.values
y = is_killed

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

# %% [markdown]
# #### ShapeletModel Classifier

# %%
n_ts, ts_sz = X_train.shape[0], X_train.shape[1]
n_classes = len(set(y))

shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz,
    n_classes=n_classes, l=0.1, r=1)
print('n_ts', n_ts)
print('ts_sz', ts_sz)
print('n_classes', n_classes)
print('shapelet_sizes', shapelet_sizes)

# %% [markdown]
# Performe grid search to find best model:

# %%
"""param_grid = {
    'optimizer': ['sgd'],
    'scale': [False], # scale time series
    'weight_regularizer': [0.01, 0.1, 0.2],
    'max_iter': [200],
    'batch_size': [16, 32, 64, 128, 256, 512],
    'verbose': [1],
    'random_state': [42]
}

gs = GridSearchCV(
    ShapeletModel(),
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(accuracy_score),
    verbose=10,
    cv=3, # uses a stratified 3-fold cv to validate the models
    refit=False
)
gs.fit(X_train, y_train)"""

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(20).style.background_gradient(
    subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %% [markdown]
# Train best model and make predictions on test set:

# %%
best_model_params = cv_results_df.loc[gs.best_index_]['params']
shp_clf = ShapeletModel(**best_model_params)

# %%
shp_clf.fit(X_train, y_train)

# %%
predicted_labels = shp_clf.predict(X_test)
print("Correct classification rate:", accuracy_score(y_test, predicted_labels))
predicted_locations = shp_clf.locate(X_train)

# %%
print(classification_report(y_test, predicted_labels))

# %%
ts_id = 0
plt.figure(figsize=(20, 3))
n_shapelets = sum(shapelet_sizes.values())
plt.title(f"Example locations of shapelet matches ({n_shapelets} shapelets extracted)")

plt.plot(X[ts_id].ravel())
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[ts_id, idx_shp]
    plt.plot(np.arange(t0, t0 + len(shp)), shp, linewidth=2)

# %%
ConfusionMatrixDisplay(confusion_matrix(y_test, predicted_labels)).plot(cmap=plt.cm.Blues);

# %% [markdown]
# ### Shaplet Discovery with LearningShapelets

# %% [markdown]
# **Shapelet Transform for Time Series**
# 
# The Shapelet model consists of a logistic regression layer on top of this transform. The shapelet coefficients as well as logistic regression weights are optimized by gradient descent on a L2-penalized cross-entropy loss.

# %%
X = incidents_by_city_df.values
#y_sh = cluster_df['cluster_kmeans_features'].values
y = is_killed.values

# split data into train and test sets, considering the stratification for classes with 1 data point
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

# %%
n_ts, ts_sz = X_train.shape[0], X_train.shape[1]
n_classes = len(set(y))

shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz, n_classes=n_classes, l=0.15, r=1)
print('n_ts', n_ts)
print('ts_sz', ts_sz)
print('n_classes', n_classes)
print('shapelet_sizes', shapelet_sizes)

# %%
learn_sh_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes, max_iter=3000, batch_size=64, verbose=0, 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), weight_regularizer=0.001, shapelet_length=0.1, 
    total_lengths=24, random_state=42)

# %%
learn_sh_clf.fit(X_train, y_train)

X_sh_train = learn_sh_clf.transform(X_train)
X_sh_test = learn_sh_clf.transform(X_test)

# %%
predicted_labels_shap = learn_sh_clf.predict(X_test)
print("Correct classification rate:", accuracy_score(y_test, predicted_labels_shap))

# %%
# Show the shapelets that are as many as we calculated in the dictionary
n_shapelet_sizes = len(shapelet_sizes.keys())
max_shapelet_size = max(shapelet_sizes.keys())

plt.figure(figsize=(20, 6))
for i, sz in enumerate(shapelet_sizes.keys()):
    plt.subplot(n_shapelet_sizes, 1, i + 1)
    plt.title("Shaplet learned, %d shapelets of size %d" % (shapelet_sizes[sz], sz))
    for sh_idx in range(shapelet_sizes[sz]):
        plt.plot(learn_sh_clf.shapelets_as_time_series_[sh_idx + sum(list(shapelet_sizes.values())[:i])].ravel())
        plt.tight_layout()
        plt.ylabel('Value')

# %% [markdown]
# ### DecisionTreeClassifier on shapelet

# %%
param_grid = {
    # values to try
    'criterion': ['entropy', 'gini'],
    'class_weight': ['balanced', None],
    'min_samples_split': [2, 0.01, 0.02, 0.025, 0.05, 0.1],
    'min_samples_leaf': [1, 0.01, 0.02, 0.025, 0.05, 0.075, 0.1],
    'splitter': ['best', 'random'],
    #'max_depth': [None],
    #'max_leaf_nodes': [None],
    #'min_impurity_decrease': [0],
    'ccp_alpha': [0, 0.001, 0.01, 0.1, 0.2, 0.5],
    #'min_weight_fraction_leaf': [0],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'random_state': [42]
}

gs = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5, # uses a stratified 5-fold cv to validate the models
    refit=False
)
gs.fit(X_sh_train, y_train)

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(10).style.background_gradient(
    subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
dt = DecisionTreeClassifier(**best_model_params)

# %%
dt.fit(X_sh_train, y_train)

# %%
predicted_labels_dt = dt.predict(X_sh_test)
print("Correct classification rate:", accuracy_score(y_test, predicted_labels_dt))

print(classification_report(y_test, predicted_labels_dt))

# %%
# complementarity of the two models
print('Complementarity of the two models:', accuracy_score(predicted_labels_shap, predicted_labels_dt))

complementary_predicted_labels_dt = [1 if x==0 else 0 for x in predicted_labels_dt]
print('Correct classification rate:', accuracy_score(y_test, complementary_predicted_labels_dt))

# %% [markdown]
# ### DecisionTreeClassifier on shapelet

# %%
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# %%
knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('knn', knn)])
param_grid = [
    {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 21, 31, 51, 73, 77, 79, 103],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['brute', 'kd_tree', 'ball_tree'],
        'knn__metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
        'knn__p': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
]

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    n_jobs=-1,
    scoring=make_scorer(f1_score),
    verbose=10,
    cv=5,
    refit=False
)
gs.fit(X_sh_train, y_train)

# %%
cv_results_df = pd.DataFrame(gs.cv_results_)
params = [col for col in cv_results_df.columns if 'param_' in col and 'random' not in col]
cv_results_df.sort_values(
    by='mean_test_score',
    ascending=False)[params+['std_test_score', 'mean_test_score']].head(10).style.background_gradient(
    subset=['std_test_score', 'mean_test_score'], cmap='Blues')

# %%
best_index = gs.best_index_
best_model_params = cv_results_df.loc[best_index]['params']
best_model_params = {k.replace('knn__', ''): v for k, v in best_model_params.items()}
knn = KNeighborsClassifier(**best_model_params)

# %%
knn.fit(X_sh_train, y_train)

# %%
predicted_labels_knn = knn.predict(X_sh_test)
print("Correct classification rate:", accuracy_score(y_test, predicted_labels_knn))

print(classification_report(y_test, predicted_labels_knn))

# %% [markdown]
# ### Rocket Classifier on time series

# %% [markdown]
# [ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](https://arxiv.org/pdf/1910.13051.pdf), Dempster et al.

# %%
rocket = RocketClassifier(num_kernels=500, rocket_transform='minirocket', random_state=42)

# %%
rocket.fit(X_train, y_train)

# %%
predicted_labels_rocket = rocket.predict(X_test)
print("Correct classification rate:", accuracy_score(y_test, predicted_labels_rocket))

# %%
print(classification_report(y_test, predicted_labels_rocket))

# %%
ConfusionMatrixDisplay(confusion_matrix(y_test, predicted_labels_rocket)).plot(cmap=plt.cm.Blues);


