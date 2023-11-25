# -*- coding: utf-8 -*-
# %% [markdown]
# # Density clustering

# %% [markdown]
# # Import library and dataset

# %%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_scattermap_plotly

# %%
incidents_df = pd.read_csv(
    '../data/incidents_indicators.csv',
    index_col=0,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

f = open('../data/indicators_names.json')
ind_names_list = json.loads(f.read())
# %%
incidents_df.head(2)

# %% [markdown]
# # Prepare dataset and indices for choosen state

# %%
illinois_df = incidents_df[incidents_df['state']=='ILLINOIS'][ind_names_list].dropna()
illinois_df.head(2)

# %% [markdown]
# # Density clustering

# %% [markdown]
# DBSCAN: density-based cluster, define a cluster as a dense region of objects.
#
# Partitional clustering, number of clester automatically detected from algorithm.
# Points in low-density region are classified as noise.
#
# Pros: can handle irregular clusters and with arbitrary shape and size, works well when noise or oulier are present.
# an find many cluster that K-means could not find.
#
# Contro: not able to classified correctly whan the clusters have widley varing density, and have trouble with high dimensional data because density is difficult to define.
#
#

# %% [markdown]
# ## Indices correlation

# %%
corr_matrix_illinois = illinois_df.dropna().corr('kendall')

import seaborn as sns

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_illinois, annot=True, cmap=plt.cm.Reds, mask=np.triu(corr_matrix_illinois))
plt.show()

# %%
ind_names_list = [
    # geo
    'location_imp',
    'entropy_address_type',
    # age
    'avg_age',
    # participants
    'severity',
    'n_males_prop',
    'n_arrested_prop',
    'n_participants',
    ]

# %%
corr_matrix_illinois = illinois_df[ind_names_list].corr('kendall')

import seaborn as sns

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_illinois, annot=True, cmap=plt.cm.Reds, mask=np.triu(corr_matrix_illinois))
plt.show()

# %%
illinois_df[ind_names_list].describe()

# %% [markdown]
# ## Utilities

# %%
from sklearn.preprocessing import StandardScaler

def standardization(df, columns):
    std_scaler = StandardScaler()
    std_scaler.fit(df[columns].values)
    return std_scaler.transform(df[columns].values)

# %%
def plot_dbscan(X, db): 
    labels = db.labels_ 
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True # create an array of booleans where True = core point
    # core point = point that has at least min_samples in its eps-neighborhood (punto interno al cluster)

    plt.figure(figsize=(20, 8))

    colors = [plt.cm.rainbow_r(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k # array of booleans where True = point in cluster k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=10,
            label=f'Cluster {k}'
        )

        # plot noise points
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor=col,
            markersize=6,
            label=f'Cluster {k}'
        )

    plt.grid()
    plt.legend()
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

# %%
from sklearn.cluster import DBSCAN
from sklearn import metrics 

def dbscan(X, eps=0.1, min_samples=10, plot_clusters=False):
    # Compute DBSCAN      
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    if plot_clusters:
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        if len(np.unique(labels)) > 1:
            print("Silhouette Coefficient: %0.3f"
                % metrics.silhouette_score(X, labels))
        plot_dbscan(X, db)
    
    return {'eps': eps, 'min_samples': min_samples, 
        '#clusters': len(set(labels)) - (1 if -1 in labels else 0),
        '#noise': list(labels).count(-1),  '%noise': list(labels).count(-1)/X.shape[0]*100,
        'silhouette_coef': metrics.silhouette_score(X, labels), 
        '#cluster0': list(labels).count(0), '#cluster1': list(labels).count(1), 
        '#cluster2': list(labels).count(2), '#cluster3': list(labels).count(3), 
        '#cluster4': list(labels).count(4), '#cluster5': list(labels).count(5),
        '#cluster6': list(labels).count(6), '#cluster7': list(labels).count(7)}

# %% [markdown]
# The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

# %% [markdown]
# ### Find best EPS

# %%
from scipy.spatial.distance import pdist, squareform

def find_best_eps(X, k_list=[3, 5, 9, 15]):
    dist = pdist(X, 'euclidean') # pair wise distance
    dist = squareform(dist) # distance matrix given the vector dist
    
    # Calculate sorted list of distances for points for each k in k_list
    # and plot the graph of distance from k-th nearest neighbour
    fig, ax = plt.subplots(int(np.ceil(len(k_list)/3)), 3, figsize=(20, 8))

    for i, k in enumerate(k_list):
        kth_distances = list()
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])

        # Plot the graph of distance from k-th nearest neighbour
        ax[int(i/3), int(i%3)].plot(range(0, len(kth_distances)), sorted(kth_distances))
        ax[int(i/3), int(i%3)].set_ylabel('%sth near neighbor distance' %k)
        ax[int(i/3), int(i%3)].set_xlabel('Point Sorted according to distance of %sth near neighbor' %k)
        #ax[int(i/3), int(i%3)].set_yticks(np.linspace(0, 5, 12))
        #ax[int(i/3), int(i%3)].set_ylim(0, 3)
        ax[int(i/3), int(i%3)].tick_params(axis='both', which='major', labelsize=8)
        ax[int(i/3), int(i%3)].grid(linestyle='--', linewidth=0.5, alpha=0.6)

    plt.show()

# %% [markdown]
# ## Clustering: Illinois

# %% [markdown]
# ### Std data

# %%
X_std_illinois = standardization(illinois_df, columns=ind_names_list)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
plt.boxplot(X_std_illinois, vert=True, labels=ind_names_list)
plt.xticks(rotation=90, ha='right')
plt.show()

# %%
#kneed_algorithm(X_std_illinois, neighbors=5)

# %%
find_best_eps(X_std_illinois, k_list=[3, 5, 9, 15, 20, 30]) # altro metodo per kneed point

# %%
eps = [0.75, 1, 1.25, 1.5, 1.75, 2]
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
min_samples = [3, 5, 10, 15, 20]

dbscan_illinois = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in eps:
    for k in min_samples:
        db = dbscan(X_std_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois = pd.concat([dbscan_illinois, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois

# %%
dbscan_illinois_second = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in [1.8, 2, 2.2]:
    for k in [5, 10, 15, 20]:
        db = dbscan(X_std_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois_second = pd.concat([dbscan_illinois_second, pd.DataFrame(db, index=[0])], ignore_index=True)

# %% [markdown]
# ### Visualize results

# %%
db = DBSCAN(eps=1.75, min_samples=5).fit(X_std_illinois) #21 dati stadardizzati, eps=1.75, min_samples=5
plot_dbscan(X_std_illinois, db)

# %%
df = illinois_df[ind_names_list]

fig, ax = plt.subplots(6, 4, figsize=(20, 30))
i = 0
for i in range(7):
    for j in range(i+1, 7):
        ax[int(i/4), i%4].scatter(df.values[:, i], df.values[:, j], c=db.labels_, cmap='plasma', s=6)
        ax[int(i/4), i%4].set_xlabel(df.columns[i], fontsize=8)
        ax[int(i/4), i%4].set_ylabel(df.columns[j], fontsize=8)
        ax[int(i/4), i%4].tick_params(axis='both', which='major', labelsize=6)
        ax[int(i/4), i%4].grid(linestyle='--', linewidth=0.5, alpha=0.6)
        i = i + 1
#plt.suptitle('DBSCAN Clustering', fontsize=16)
plt.show()

# %%
columns = ['n_males', 'n_adult', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed']
df = incidents_df.loc[illinois_df.index][columns]

fig, ax = plt.subplots(4, 4, figsize=(20, 15))
i = 0
for i in range(6):
    for j in range(i+1, 6):
        ax[int(i/4), i%4].scatter(df.values[:, i], df.values[:, j], c=db.labels_, cmap='plasma', s=6)
        ax[int(i/4), i%4].set_xlabel(df.columns[i], fontsize=8)
        ax[int(i/4), i%4].set_ylabel(df.columns[j], fontsize=8)
        ax[int(i/4), i%4].tick_params(axis='both', which='major', labelsize=6)
        ax[int(i/4), i%4].grid(linestyle='--', linewidth=0.5, alpha=0.6)
        i = i + 1
#plt.suptitle('DBSCAN Clustering', fontsize=16)
plt.show()

# %%
# bar plot of number of incidents per cluster
cluster_counts = pd.Series(db.labels_).value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(cluster_counts.index, cluster_counts.values, edgecolor='black', linewidth=0.8, alpha=0.5)
plt.xlabel('Cluster')
plt.xticks(cluster_counts.index)
plt.ylabel('Number of incidents')
plt.yscale('log')
for i, v in enumerate(cluster_counts.values):
    plt.text(x=i-1, y=v, s=str(v), horizontalalignment='center', verticalalignment='bottom', fontsize=8)
plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
plt.title('Number of incidents per cluster')
plt.show()


# %%
illinois_df['cluster'] = db.labels_
sns.pairplot(illinois_df, hue='cluster', palette=sns.color_palette(
    n_colors=illinois_df['cluster'].unique().shape[0]), vars=ind_names_list)
plt.show()

# %%
fig, ax = plt.subplots(4, 2, figsize=(20, 15), sharex=False, sharey=False)
i = 0
for i in range(7):
    for cluster in np.unique(db.labels_):
        ax[int(i/2), i%2].hist(illinois_df.values[db.labels_==cluster, i], 
            bins=int(1+3.3*np.log(X_std_illinois[db.labels_==cluster, i].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(i/2), i%2].set_xlabel(illinois_df.columns[i], fontsize=8)
    ax[int(i/2), i%2].set_yscale('log')
    ax[int(i/2), i%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(i/2), i%2].legend()
    ax[int(i/2), i%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)
    i = i + 1


# %%
columns = ['n_males', 'n_adult', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed']
df = incidents_df.loc[illinois_df.index][columns]
df['cluster'] = db.labels_

fig, ax = plt.subplots(3, 2, figsize=(20, 10), sharex=False, sharey=False)
i = 0
for i in range(6):
    for cluster in np.unique(db.labels_):
        ax[int(i/2), i%2].hist(df[df['cluster']==cluster][columns[i]], 
            bins=int(1+3.3*np.log(df[df['cluster']==cluster].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(i/2), i%2].set_xlabel(df.columns[i], fontsize=8)
    ax[int(i/2), i%2].set_yscale('log')
    ax[int(i/2), i%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(i/2), i%2].legend()
    ax[int(i/2), i%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)
    i = i + 1

# %%
#illinois_df['cluster'] = db.labels_
illinois_df[['latitude', 'longitude', 'county', 'city']] = incidents_df.loc[illinois_df.index, [
    'latitude', 'longitude', 'county', 'city']]

illinois_df.head(2)

# %%
plot_scattermap_plotly(illinois_df, 'cluster', zoom=5, title='Incidents clustered by DBSCAN')

# %%
plot_scattermap_plotly(illinois_df[illinois_df['county']=='Cook County'], 'cluster', zoom=8, 
    title='Incidents clustered by DBSCAN in Cook county')

# %% [markdown]
# ### MinMax Scale Data

# %%
from sklearn.preprocessing import MinMaxScaler

def minmax_scaler(df, columns):
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(df[columns].values)
    return minmax_scaler.transform(df[columns].values)


# %%
X_minmax_illinois = minmax_scaler(illinois_df, columns=ind_names_list)

# %%
find_best_eps(X_minmax_illinois, k_list=[3, 5, 9, 15, 20, 30])

# %%
eps = [0.1, 0.15, 0.2, 0.25]
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
min_samples = [3, 5, 10, 15, 20]

dbscan_illinois = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in eps:
    for k in min_samples:
        db = dbscan(X_minmax_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois = pd.concat([dbscan_illinois, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois

# %% [markdown]
# ### Visualize data

# %%
db = DBSCAN(eps=0.2, min_samples=10).fit(X_minmax_illinois) #12
plot_dbscan(X_std_illinois, db)

# %%
# bar plot of number of incidents per cluster
cluster_counts = pd.Series(db.labels_).value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(cluster_counts.index, cluster_counts.values, edgecolor='black', linewidth=0.8, alpha=0.5)
plt.xlabel('Cluster')
plt.xticks(cluster_counts.index)
plt.ylabel('Number of incidents')
plt.yscale('log')
for i, v in enumerate(cluster_counts.values):
    plt.text(x=i-1, y=v, s=str(v), horizontalalignment='center', verticalalignment='bottom', fontsize=8)
plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
plt.title('Number of incidents per cluster')
plt.show()

# %%
illinois_df['cluster'] = db.labels_
sns.pairplot(illinois_df, hue='cluster', palette=sns.color_palette(
    n_colors=illinois_df['cluster'].unique().shape[0]), vars=ind_names_list)
plt.show()

# %%
fig, ax = plt.subplots(7, 1, figsize=(20, 30), sharex=False, sharey=False)
i = 0
for i in range(7):
    for cluster in np.unique(db.labels_):
        ax[i].hist(illinois_df.values[db.labels_==cluster, i], 
            bins=int(1+3.3*np.log(X_std_illinois[db.labels_==cluster, i].shape[0])), 
            stacked=True, fill=True, histtype='step',
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[i].set_xlabel(illinois_df.columns[i], fontsize=8)
    ax[i].set_yscale('log')
    ax[i].tick_params(axis='both', which='major', labelsize=6)
    ax[i].legend(fontsize=8, )
    ax[i].grid(linestyle='--', linewidth=0.5, alpha=0.6)
    i = i + 1

# %%
columns = ['n_males', 'n_adult', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 'year', 'poverty_perc', 'congd']
df = incidents_df.loc[illinois_df.index][columns]

fig, ax = plt.subplots(9, 4, figsize=(20, 30))
i = 0
for i in range(9):
    for j in range(i+1, 9):
        ax[int(i/4), i%4].scatter(df.values[:, i], df.values[:, j], c=db.labels_, cmap='plasma', s=20)
        ax[int(i/4), i%4].set_xlabel(df.columns[i], fontsize=8)
        ax[int(i/4), i%4].set_ylabel(df.columns[j], fontsize=8)
        ax[int(i/4), i%4].tick_params(axis='both', which='major', labelsize=6)
        ax[int(i/4), i%4].grid(linestyle='--', linewidth=0.5, alpha=0.6)
        i = i + 1
#plt.suptitle('DBSCAN Clustering', fontsize=16)
plt.show()

# %%
columns = ['n_males', 'n_adult', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 'year', 'poverty_perc']
df = incidents_df.loc[illinois_df.index][columns]
df['cluster'] = db.labels_

fig, ax = plt.subplots(4, 2, figsize=(20, 15), sharex=False, sharey=False)
for i in range(8):
    for cluster in np.unique(db.labels_):
        ax[int(i/2), i%2].hist(df[df['cluster']==cluster][columns[i]], 
            bins=int(1+3.3*np.log(df[df['cluster']==cluster].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(i/2), i%2].set_xlabel(df.columns[i], fontsize=8)
    ax[int(i/2), i%2].set_yscale('log')
    ax[int(i/2), i%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(i/2), i%2].legend(fontsize=8)
    ax[int(i/2), i%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)

# %%
illinois_df[['latitude', 'longitude', 'county', 'city']] = incidents_df.loc[illinois_df.index, [
    'latitude', 'longitude', 'county', 'city']]

illinois_df.head(2)

# %%
plot_scattermap_plotly(illinois_df, 'cluster', zoom=5, title='Incidents clustered by DBSCAN')