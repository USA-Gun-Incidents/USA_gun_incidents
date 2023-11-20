# -*- coding: utf-8 -*-
# %% [markdown]
# # Density clustering

# %% [markdown]
# # Import library and dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_scattermap_plotly

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=False,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

indicators_df = pd.read_csv(
    '../data/incidents_cleaned_indicators.csv', 
    index_col=0
)
# %% [markdown]
# # Prepare dataset and indices for choosen state

# %%
columns = ['location_importance', 'avg_age_participants', 'age_range', 
    'n_participants_child_prop', 'n_participants_teen_prop', 'n_males_pr', 
    'n_killed_pr', 'n_arrested_pr']

# %%
illinois_df = indicators_df[indicators_df['state']=='ILLINOIS'][columns].dropna()
illinois_df.info()
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
corr_matrix_illinois = illinois_df[columns].corr()

import seaborn as sns

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_illinois, annot=True, cmap=plt.cm.Reds, mask=np.triu(corr_matrix_illinois))
plt.show()

# %%
illinois_df[columns].describe()

# %%
# show Nan values in illinois_df[columns]
illinois_df[columns].isna().sum()

# %% [markdown]
# ## Utilities

# %%
from sklearn.preprocessing import StandardScaler

def standardization(df, columns):
    std_scaler = StandardScaler()
    std_scaler.fit(df[columns].values)
    return std_scaler.transform(df[columns].values)

# %%
def plot_dbscan(X_std, db): 
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

        xy = X_std[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=6,
            label=f'Cluster {k}'
        )

        # plot noise points
        xy = X_std[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "x",
            markerfacecolor=tuple(col),
            markeredgecolor=col,
            markersize=8,
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
        ax[int(i/3), int(i%3)].set_ylim(0, 3)
        ax[int(i/3), int(i%3)].tick_params(axis='both', which='major', labelsize=8)
        ax[int(i/3), int(i%3)].grid(linestyle='--', linewidth=0.5, alpha=0.6)

    plt.show()

# %% [markdown]
# ## Clustering: Illinois

# %% [markdown]
# ### Std data

# %%
X_std_illinois = standardization(illinois_df, columns) #TODO: sono già standardizzati

# %%
#kneed_algorithm(X_std_illinois, neighbors=5)

# %%
find_best_eps(X_std_illinois, k_list=[3, 5, 9, 15, 20, 30]) # altro metodo per kneed point

# %%
eps = [0.5, 1, 1.5, 2]
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
min_samples = [5, 10, 15, 20]

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

for e in [1.7, 2.3, 2.5]:
    for k in [5, 10, 15, 20]:
        db = dbscan(X_std_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois_second = pd.concat([dbscan_illinois_second, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois_second

# %%
dbscan_illinois_third = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in [2.1, 2.2, 2.3]:
    for k in [5, 7, 10]:
        db = dbscan(X_std_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois_third = pd.concat([dbscan_illinois_third, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois_third

# %% [markdown]
# ### Visualize results

# %% [markdown]
# scegliamo config 
# eps = 2.3, k = 10
#
# ultima

# %%
db = DBSCAN(eps=2.3, min_samples=10).fit(X_std_illinois)
plot_dbscan(X_std_illinois, db)

# %%
fig, ax = plt.subplots(7, 4, figsize=(20, 30))
index = 0
for i in range(8):
    for j in range(i+1, 8):
        ax[int(index/4), index%4].scatter(illinois_df.values[:, i], illinois_df.values[:, j], c=db.labels_, cmap='plasma_r', s=6)
        ax[int(index/4), index%4].set_xlabel(illinois_df.columns[i], fontsize=8)
        ax[int(index/4), index%4].set_ylabel(illinois_df.columns[j], fontsize=8)
        ax[int(index/4), index%4].tick_params(axis='both', which='major', labelsize=6)
        ax[int(index/4), index%4].grid(linestyle='--', linewidth=0.5, alpha=0.6)
        index = index + 1
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
fig, ax = plt.subplots(4, 2, figsize=(20, 15), sharex=False, sharey=False)
index = 0
for i in range(8):
    for cluster in np.unique(db.labels_):
        ax[int(index/2), index%2].hist(illinois_df.values[db.labels_==cluster, i], 
            bins=int(1+3.3*np.log(X_std_illinois[db.labels_==cluster, i].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(index/2), index%2].set_xlabel(illinois_df.columns[i], fontsize=8)
    ax[int(index/2), index%2].set_yscale('log')
    ax[int(index/2), index%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(index/2), index%2].legend()
    ax[int(index/2), index%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)
    index = index + 1
        

# %%
illinois_df['cluster'] = db.labels_
illinois_df[['latitude', 'longitude', 'county', 'city']] = incidents_df.loc[illinois_df.index, [
    'latitude', 'longitude', 'county', 'city']]

illinois_df.head(2)

# %%
plot_scattermap_plotly(illinois_df, 'cluster', zoom=5, title='Incidents clustered by DBSCAN')

# %%
plot_scattermap_plotly(illinois_df[illinois_df['county']=='Cook'], 'cluster', zoom=8, 
    title='Incidents clustered by DBSCAN in Cook county')
