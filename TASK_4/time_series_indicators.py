# %% [markdown]
# # Time Series Analysis

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=0,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)
incidents_df.drop_duplicates(inplace=True)
# keep only incidents between 2014 and 2017
incidents_df = incidents_df[incidents_df['year'].between(2014, 2017)]
incidents_df['year'].unique()

# %%
incidents_df['date'].min()

# %%
pd.to_datetime('2014-1-1', format='%Y-%m-%d').day_name()

# %%
pd.to_datetime('2014-1-6', format='%Y-%m-%d').day_name()

# %%
pd.to_datetime('2017-12-31', format='%Y-%m-%d').day_name()

# %%
incidents_df['week'] = (((incidents_df['date'] - pd.to_datetime('2014-1-1')).dt.days)+2) // 7

# %%
incidents_df[incidents_df['date'] == pd.to_datetime('2014-1-1')]['week'].unique()

# %% [markdown]
# Weeks are numbered from 0. The first week has 2 days less than the others.

# %%
incidents_df[incidents_df['date'] == pd.to_datetime('2014-1-6')]['week'].unique()

# %% [markdown]
# 6th January 2014 is Monday and belongs to week 1 (the second).

# %%
# number of weeks in the dataset
number_of_weeks = incidents_df['week'].max()+1
number_of_weeks

# %%
incidents_df['week'].unique().shape # all weeks are present

# %%
# group by week and count incidents
plt.figure(figsize=(20, 5))
plt.bar(
    incidents_df.groupby('week').size().index,
    incidents_df.groupby('week').size().values
)
plt.title('Number of incidents per week');

# %%
incidents_df.groupby(['city', 'state']).size().shape[0] # number of cities

# %% [markdown]
# We consider only cities with a number of weeks with incidents greater than 15% of the total number of the weeks of the 4 years, i.e. those having at least an incident in the following number of weeks

# %%
0.15 * number_of_weeks

# %%
incidents_df.groupby(['city', 'state'])['week'].count()

# %%
# list of index of incidents in city with incidents in more than 30 weeks
index_list = np.where(incidents_df.groupby(['city', 'state'])['week'].transform('count') > number_of_weeks*0.15)

# %%
# create a df with incidents_df where index is in index_list
incidents_df = incidents_df.iloc[index_list]
incidents_df.head(2)

# %%
incidents_df.groupby(['city', 'state']).size().shape[0] # number of cities left

# %%
incidents_df['state'].unique().shape # all states are present

# %%
# gruop by week and count incidents
plt.figure(figsize=(20, 5))
plt.bar(
    incidents_df.groupby('week').size().index,
    incidents_df.groupby('week').size().values
)
plt.title('Number of incidents per week');

# %% [markdown]
# ## Create Time series

# %%
n_weeks_per_year = 52

# %%
incidents_df.shape[0]

# %%
incidents_killed_df = incidents_df.dropna(subset=['n_killed'])
incidents_killed_df.shape[0]

# %%
killed_index_list = np.where(incidents_killed_df.groupby(['city', 'state'])['week'].transform('count') > number_of_weeks*0.15)
incidents_killed_df = incidents_killed_df.iloc[killed_index_list]
incidents_killed_df.shape[0]

# %%


# %%
incidents_killed_df.groupby(['city', 'state']).size().shape[0] # number of cities

# %%
incidents_killed_by_city_df = incidents_killed_df.groupby(['city', 'state', 'week'])['n_killed'].mean().reset_index()
incidents_killed_by_city_df = incidents_killed_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_killed')
#incidents_killed_by_city_df = incidents_killed_by_city_df.fillna(0)
incidents_killed_by_city_df

# %%
plt.figure(figsize=(20, 5))
plt.plot(np.nanmean(incidents_killed_by_city_df.values, axis=0), '.--')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*2, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*3, color='k', linestyle='--')
plt.title('Average number of killed people per week (mean over all cities)');


# %%
plt.figure(figsize=(20, 5))
new_york_killed_ts = incidents_killed_by_city_df[(incidents_killed_by_city_df.index.get_level_values('city') == 'City of New York')].values[0]
los_angeles_killed_ts = incidents_killed_by_city_df[(incidents_killed_by_city_df.index.get_level_values('city') == 'Los Angeles')].values[0]
chicago_killed_ts = incidents_killed_by_city_df[(incidents_killed_by_city_df.index.get_level_values('city') == 'Chicago')].values[0]
plt.plot(new_york_killed_ts, '.--', label='New York')
plt.plot(los_angeles_killed_ts, '.--', label='Los Angeles')
plt.plot(chicago_killed_ts, '.--', label='Chicago')
plt.title('Average numbr of killed people per week');
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*2, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*3, color='k', linestyle='--')
plt.legend();

# %%
from tslearn.utils import to_time_series_dataset
X = [row[~np.isnan(row)] for row in incidents_killed_by_city_df.values]
X = to_time_series_dataset(X)
# se passo a kmeans nan non funziona (non da errori ma setta inertia a inf)

# %%
km = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=100, random_state=42)
pred = km.fit_predict(X)

# %%
incidents_females_df = incidents_df.dropna(subset=['n_females'])
incidents_females_df.shape[0]

# %%
females_index_list = np.where(incidents_females_df.groupby(['city', 'state'])['week'].transform('count') > number_of_weeks*0.15)
incidents_females_df = incidents_females_df.iloc[females_index_list]
incidents_females_df.shape[0]

# %%
incidents_females_df.groupby(['city', 'state']).size().shape[0] # number of cities

# %%
incidents_females_by_city_df = incidents_females_df.groupby(['city', 'state', 'week'])['n_females'].mean().reset_index()
incidents_females_by_city_df = incidents_females_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_females')
#incidents_females_by_city_df = incidents_females_by_city_df.fillna(0)
incidents_females_by_city_df

# %%
plt.figure(figsize=(20, 5))
plt.plot(np.nanmean(incidents_females_by_city_df.values, axis=0), '.--')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*2, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*3, color='k', linestyle='--')
plt.title('Average number of females involved in incidents per week (mean over all cities)');

# %%
plt.figure(figsize=(20, 5))
new_york_females_ts = incidents_females_by_city_df[(incidents_females_by_city_df.index.get_level_values('city') == 'City of New York')].values[0]
los_angeles_females_ts = incidents_females_by_city_df[(incidents_females_by_city_df.index.get_level_values('city') == 'Los Angeles')].values[0]
chicago_females_ts = incidents_females_by_city_df[(incidents_females_by_city_df.index.get_level_values('city') == 'Chicago')].values[0]
plt.plot(new_york_females_ts, '.--', label='New York')
plt.plot(los_angeles_females_ts, '.--', label='Los Angeles')
plt.plot(chicago_females_ts, '.--', label='Chicago')
plt.title('Average number of females involved in incidents per week');
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*2, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*3, color='k', linestyle='--')
plt.legend();

# %%
incidents_females_df.dropna(subset=['n_killed'], inplace=True)

incidents_fatal_females_by_city_df = incidents_females_df[incidents_females_df['n_killed']>0].groupby(['city', 'state', 'week'])['n_females'].mean().reset_index()
incidents_fatal_females_by_city_df = incidents_fatal_females_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_females')
#incidents_fatal_females_by_city_df = incidents_fatal_females_by_city_df.fillna(0)

incidents_nonfatal_females_by_city_df = incidents_females_df[incidents_females_df['n_killed']==0].groupby(['city', 'state', 'week'])['n_females'].mean().reset_index()
incidents_nonfatal_females_by_city_df = incidents_nonfatal_females_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_females')
#incidents_nonfatal_females_by_city_df = incidents_nonfatal_females_by_city_df.fillna(0)

plt.figure(figsize=(20, 5))
plt.plot(np.nanmean(incidents_fatal_females_by_city_df.values, axis=0), '.--', label='fatal')
plt.plot(np.nanmean(incidents_nonfatal_females_by_city_df.values, axis=0), '.--', label='non fatal')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*2, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*3, color='k', linestyle='--')
plt.title('Average number of females involved in incidents per week (mean over all cities)');
plt.legend();

# %%
incidents_df['n_young'] = incidents_df['n_participants_child'] + incidents_df['n_participants_teen']
incidents_young_df = incidents_df.dropna(subset=['n_young'])
incidents_young_df.shape[0]

# %%
young_index_list = np.where(incidents_young_df.groupby(['city', 'state'])['week'].transform('count') > number_of_weeks*0.15)
incidents_young_df = incidents_young_df.iloc[young_index_list]
incidents_young_df.shape[0]

# %%
incidents_young_df.groupby(['city', 'state']).size().shape[0] # number of cities

# %%
incidents_young_by_city_df = incidents_young_df.groupby(['city', 'state', 'week'])['n_young'].mean().reset_index()
incidents_young_by_city_df = incidents_young_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_young')
#incidents_young_by_city_df = incidents_young_by_city_df.fillna(0)
incidents_young_by_city_df

# %%
plt.figure(figsize=(20, 5))
plt.plot(np.nanmean(incidents_young_by_city_df.values, axis=0), '.--')
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*2, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*3, color='k', linestyle='--')
plt.title('Average number of young people per week (mean over all cities)');

# %%
plt.figure(figsize=(20, 5))
new_york_young_ts = incidents_young_by_city_df[(incidents_young_by_city_df.index.get_level_values('city') == 'City of New York')].values[0]
los_angeles_young_ts = incidents_young_by_city_df[(incidents_young_by_city_df.index.get_level_values('city') == 'Los Angeles')].values[0]
chicago_young_ts = incidents_young_by_city_df[(incidents_young_by_city_df.index.get_level_values('city') == 'Chicago')].values[0]
plt.plot(new_york_young_ts, '.--', label='New York')
plt.plot(los_angeles_young_ts, '.--', label='Los Angeles')
plt.plot(chicago_young_ts, '.--', label='Chicago')
plt.title('Average number of young people per week (amplitude scaling)');
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*2, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*3, color='k', linestyle='--')
plt.legend();

# %% [markdown]
# Time series: mean number of participants per incident per week in each city
# 
# 0 if we have no incidents in the week or NaN values (i.e. incidents where we don not know the nember of participants)

# %%
# create a dataset with series of mean number of participants per incident per week in each city
incidents_df['n_participants'] = incidents_df['n_participants'].fillna(0) # substitute NaN with 0
incidents_by_city_df = incidents_df.groupby(['city', 'state', 'week'])['n_participants'].mean().reset_index()
incidents_by_city_df = incidents_by_city_df.pivot(index=['city', 'state'], columns='week', values='n_participants')
incidents_by_city_df = incidents_by_city_df.fillna(0) # substitute NaN with 0
incidents_by_city_df

# %%
incidents_by_city_df.groupby('state')[0].count().sort_values(ascending=False) # number of cities per state

# %%
# plot time series for big cities
new_york_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'City of New York')].values[0]
los_angeles_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'Los Angeles')].values[0]
chicago_ts = incidents_by_city_df[(incidents_by_city_df.index.get_level_values('city') == 'Chicago')].values[0]
plt.figure(figsize=(20, 5))
plt.plot(new_york_ts, '.--', label='New York')
plt.plot(los_angeles_ts, '.--', label='Los Angeles')
plt.plot(chicago_ts, '.--', label='Chicago')
plt.title('Number of participants per incident per week');
plt.legend();

# %%
# offset translation
plt.figure(figsize=(20, 5))
plt.plot(new_york_ts - new_york_ts.mean(), '.--', label='New York')
plt.plot(los_angeles_ts - los_angeles_ts.mean(), '.--', label='Los Angeles')
plt.plot(chicago_ts - chicago_ts.mean(), '.--', label='Chicago')
plt.title('Number of participants per incident per week (offset translation)');
plt.legend();

# %%
# amplitude scaling
plt.figure(figsize=(20, 5))
plt.plot((new_york_ts-new_york_ts.mean()) / new_york_ts.std(), '.--', label='New York')
plt.plot((los_angeles_ts-los_angeles_ts.mean()) / los_angeles_ts.std(), '.--', label='Los Angeles')
plt.plot((chicago_ts-chicago_ts.mean()) / chicago_ts.std(), '.--', label='Chicago')
plt.title('Number of participants per incident per week (amplitude scaling)');
plt.legend();

# %%
# plot time series for city in ALASKA state
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.T, '.--')
plt.title('Number of participants per incident per week')
plt.legend(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'
    ].index.get_level_values('city'), loc='upper left', bbox_to_anchor=(1, 1));

# %%
# Offset translation
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.T - 
    incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.mean(axis=1), '.--')
plt.title('Number of participants per incident per week, offset translation')
plt.legend(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].index.get_level_values('city'), 
    loc='upper left', bbox_to_anchor=(1, 1));

# %%
# Amplitude translation
plt.figure(figsize=(20, 5))
plt.plot((incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.T - 
    incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.mean(axis=1)) /
    incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].values.std(axis=1), '.--')
plt.title('Number of participants per incident per week, amplitude translation')
plt.legend(incidents_by_city_df[incidents_by_city_df.index.get_level_values('state') == 'ALASKA'].index.get_level_values('city'), 
    loc='upper left', bbox_to_anchor=(1, 1));

# %%
# mean of all time series
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df.values.mean(axis=0), '.--')
plt.title('Mean of number of participants per incident per week');
n_weeks_per_year = 52
plt.axvline(x=n_weeks_per_year-1, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*2, color='k', linestyle='--')
plt.axvline(x=(n_weeks_per_year-1)*3, color='k', linestyle='--')

# %% [markdown]
# Grafico coerente, molti 0 nel dataset e la maggior parte degli incidenti aveva 1 solo partecipante

# %% [markdown]
# ## Clustering

# %% [markdown]
# ### Shape-based clustering: k-means

# %% [markdown]
# Choose best k:

# %%
X = TimeSeriesScalerMeanVariance().fit_transform(incidents_by_city_df.values) # scale time series
k_list = [2, 5, 10, 15, 20, 50]
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
best_k = 11
km = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=100, random_state=42)
km.fit(X)

# %%
plt.figure(figsize=(20, 5))
plt.plot(km.cluster_centers_.reshape(incidents_by_city_df.values.shape[1], best_k))
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
cluster_df['cluster'] = cluster
cluster_df.head(2)

# %%
fig = px.scatter_mapbox(
    lat=cluster_df['latitude'],
    lon=cluster_df['longitude'],
    zoom=2, 
    color=cluster_df['cluster'],
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
    cluster_df.groupby('cluster').size().index,
    cluster_df.groupby('cluster').size().values
)
plt.title('Number of cities per cluster');

# %%
# visualize time series for each cluster (mean)
plt.figure(figsize=(20, 5))
plt.plot(incidents_by_city_df.groupby(cluster).mean().values.T, '.--')
plt.title('Number of participants per incident per week, mean for each cluster')
plt.legend(incidents_by_city_df.groupby(cluster).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %% [markdown]
# ### Compression-based clustering

# %%
from sklearn.metrics import pairwise_distances
import zlib
from sklearn.cluster import DBSCAN
from tslearn.piecewise import PiecewiseAggregateApproximation
from sklearn import metrics 
from scipy.spatial.distance import pdist, squareform

# %% [markdown]
# #### DBSCAN measuring the distance between each pair of points in a dataset via Pairwise Distances

# %%
def cdm_dist(x, y):
    # compounding dissimilarity measure
    x_str = (' '.join([str(v) for v in x.ravel()])).encode('utf-8')
    y_str = (' '.join([str(v) for v in y.ravel()])).encode('utf-8')
    return len(zlib.compress(x_str + y_str)) / (len(zlib.compress(x_str)) + len(zlib.compress(y_str)))

X = incidents_by_city_df.values
M = pairwise_distances(X.reshape(X.shape[0], X.shape[1]), metric=cdm_dist)

# %%
#TODO: hierarchical con M

# %%
def find_best_eps(X, min_samples_range=[3, 5, 9, 15]):
    dist = pdist(X, 'euclidean') # pair wise distance
    dist = squareform(dist) # distance matrix given the vector dist
    
    # Calculate sorted list of distances for points for each k in k_list
    # and plot the graph of distance from k-th nearest neighbour
    fig, ax = plt.subplots(int(np.ceil(len(min_samples_range)/3)), 3, figsize=(20, 8))

    for i, k in enumerate(min_samples_range):
        kth_distances = list()
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])

        # Plot the graph of distance from k-th nearest neighbour
        ax[int(i/3), int(i%3)].plot(range(0, len(kth_distances)), sorted(kth_distances))
        ax[int(i/3), int(i%3)].set_ylabel('%sth near neighbor distance' %k)
        ax[int(i/3), int(i%3)].set_xlabel('Point Sorted according to distance of %sth near neighbor' %k)
        ax[int(i/3), int(i%3)].tick_params(axis='both', which='major', labelsize=8)
        ax[int(i/3), int(i%3)].grid(linestyle='--', linewidth=0.5, alpha=0.6)

    plt.show()

def dbscan(X, eps=0.1, min_samples=10):
    # Compute DBSCAN      
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    return {'eps': eps, 'min_samples': min_samples, 
        '#clusters': len(set(labels)) - (1 if -1 in labels else 0),
        '#noise': list(labels).count(-1),  '%noise': list(labels).count(-1)/X.shape[0]*100,
        'silhouette_coef': metrics.silhouette_score(X, labels) if n_clusters_ > 1 else None,
        '#cluster0': list(labels).count(0), '#cluster1': list(labels).count(1), 
        '#cluster2': list(labels).count(2), '#cluster3': list(labels).count(3), 
        '#cluster4': list(labels).count(4), '#cluster5': list(labels).count(5),
        '#cluster6': list(labels).count(6), '#cluster7': list(labels).count(7)}

# %%
find_best_eps(M, min_samples_range=[3, 5, 9, 15, 20, 30])

# %%
eps = [0.55, 0.6, 0.65]
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
min_samples = [2, 5, 7]

dbscan_df = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in eps:
    for k in min_samples:
        db = dbscan(M, eps=e, min_samples=k)
        dbscan_df = pd.concat([dbscan_df, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_df

# %%
dbscan = DBSCAN(eps=0.63, min_samples=2, metric='precomputed')
dbscan.fit(M)

# %%
cluster = dbscan.labels_

cluster_df = incidents_df.groupby(['city', 'state'])[['latitude', 'longitude']].mean().reset_index()
cluster_df['cluster'] = cluster
cluster_df.head(2)

# %%
fig = px.scatter_mapbox(
    lat=cluster_df['latitude'],
    lon=cluster_df['longitude'],
    zoom=2, 
    color=cluster_df['cluster'],
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
    cluster_df.groupby('cluster').size().index,
    cluster_df.groupby('cluster').size().values
)
plt.title('Number of cities per cluster');

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
X_paa = paa.fit_transform(X) # PAA transformation

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
plt.legend(incidents_by_city_df.groupby(cluster).mean().index, loc='upper left', bbox_to_anchor=(1, 1));

# %%
cluster = km.labels_

cluster_df = incidents_df.groupby(['city', 'state'])[['latitude', 'longitude']].mean().reset_index()
cluster_df['cluster'] = cluster
cluster_df.head(2)

# %%
fig = px.scatter_mapbox(
    lat=cluster_df['latitude'],
    lon=cluster_df['longitude'],
    zoom=2, 
    color=cluster_df['cluster'],
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
    cluster_df.groupby('cluster').size().index,
    cluster_df.groupby('cluster').size().values
)
plt.title('Number of cities per cluster');

# %% [markdown]
# ## Matrix profile

# %%
#from matrixprofile-ts.matrixprofile import matrixProfile
#from matrixprofile import *

# %%
# w = 3
# mp, mpi = matrixProfile.stomp(incidents_by_city_df[0].values, w)

# plt.plot(mp)
# plt.title('Matrix Profile');


