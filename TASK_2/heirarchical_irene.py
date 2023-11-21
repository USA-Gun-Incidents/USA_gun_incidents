# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cut_tree, cophenet
from sklearn.preprocessing import MinMaxScaler
import utm
from clustering_utils import *
# %matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %%
# TODO: si leggerà un solo file che contiene tutto
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv'
)
indicators_df = pd.read_csv(
    '../data/incidents_cleaned_indicators.csv', index_col=0
)
features_to_cluster = [
    'latitude_proj', 'longitude_proj', # TODO: escludendole cambia molto...
    'location_importance', 'city_entropy', 'address_entropy',
    'avg_age_participants', 'age_range', 'log_avg_age_mean_SD', 'avg_age_entropy',
    'n_participants', 'n_participants_child_prop', 'n_participants_teen_prop', 'n_participants_adult_entropy',
    'n_males_pr', 'log_n_males_n_males_mean_semest_congd_ratio',
    'n_killed_pr', 'n_injured_pr', 'n_arrested_pr', 'n_unharmed_pr',
    'tags_entropy'
]

# %%
# drop nan
indicators_df = indicators_df.dropna()
incidents_df = incidents_df.loc[indicators_df.index]

# %%
# restrict to ILLINOIS
indicators_df = indicators_df.loc[incidents_df['state']=='ILLINOIS']
incidents_df = incidents_df.loc[indicators_df.index]

# %%
latlong_projs = utm.from_latlon(indicators_df['latitude'].to_numpy(), indicators_df['longitude'].to_numpy())
scaler= MinMaxScaler()
latlong = scaler.fit_transform(np.stack([latlong_projs[0], latlong_projs[1]]).reshape(-1, 2))
indicators_df['latitude_proj'] = latlong[:,0]
indicators_df['longitude_proj'] = latlong[:,1]

# %%
X = indicators_df[features_to_cluster].values

# %%
dm = pdist(X, metric='euclidean')

# %%
methods = [
    'single',
    'complete',
    'average',
    'weighted', # as average but does not take into account the number of points in each cluster to decide how to merge
    'centroid',
    'median',
    'ward'
]
linkages = []
default_distance_thresholds = []

# %%
nrows = 2
ncols = 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 16))
for i, method in enumerate(methods):
    dl = linkage(dm, method=method, metric='euclidean', optimal_ordering=False) # optimal_ordering=False otherwise it takes too long
    linkages.append(dl)
    distance_threshold = 0.7*max(dl[:,2]) # 70% of the maximum distance between two clusters that are merged
    default_distance_thresholds.append(distance_threshold)
    dendrogram(dl, truncate_mode='lastp', p=50, ax=axs[int(i/ncols)][i%ncols])
    axs[int(i/ncols)][i%ncols].axhline(distance_threshold, ls='--', color='k', label='threshold')
    axs[int(i/ncols)][i%ncols].legend()
    axs[int(i/ncols)][i%ncols].set_xlabel('Number of samples if between parenthesis, sample index otherwise', fontsize=6)
    axs[int(i/ncols)][i%ncols].set_ylabel('Euclidean Distance')
    axs[int(i/ncols)][i%ncols].set_title(f'{method} linkage\n default threshold at {distance_threshold:.2f}')
    i += 1
fig.suptitle('Dendrograms with different linkage methods', fontweight='bold')
for ax in axs[nrows-1, i%ncols+1:]:
    ax.remove()


# %% [markdown]
# Inversions in the centroid linkage and median linkage dendograms are due to the fact that the distance between cluster centroids or medoids can diminish in later agglomeration steps.

# %%
cophenetic_coefs = []
for i, method in enumerate(methods):
    cophenetic_matrix = cophenet(linkages[i])
    cophenetic_coef = np.corrcoef(cophenetic_matrix, dm)[0][1]
    cophenetic_coefs.append(cophenetic_coef)

cophenetic_df = pd.DataFrame()
cophenetic_df['method'] = methods
cophenetic_df['cophenetic correlation coefficient'] = cophenetic_coefs
cophenetic_df.set_index(['method'], inplace=True)
cophenetic_df

# %%
def plot_cluster_map(
        X,
        method,
        xlabels
    ):
    '''
    This function plots a cluster map of the data X, using the specified method.

    :param X: data to cluster
    :param method: method to use for clustering
    :param xlabels: list of labels for the features
    '''

    g = sns.clustermap(X, method=method, metric='euclidean')
    g.fig.suptitle(f'Cluster map of {method} linkage clustering', fontweight='bold')
    xlabels = [xlabels[int(xtick.get_text())] for xtick in g.ax_heatmap.get_xmajorticklabels()]
    g.ax_heatmap.set_xticklabels(labels=xlabels, rotation=90);
    g.ax_heatmap.set_yticklabels(labels=[]);
    g.ax_heatmap.set_yticks([]);

# %%
plot_cluster_map(X, 'average', indicators_df.columns)

# %%
start_iteration = 13100
nrows = 2
ncols = 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 16))
for i, method in enumerate(methods):
    axs[int(i/ncols)][i%ncols].plot(range(start_iteration, linkages[i].shape[0]), linkages[i][start_iteration:, 2], 'o')
    axs[int(i/ncols)][i%ncols].axhline(default_distance_thresholds[i], ls='--', color='k', label='default threshold')
    axs[int(i/ncols)][i%ncols].legend()
    axs[int(i/ncols)][i%ncols].set_title(f'{method} linkage')
    axs[int(i/ncols)][i%ncols].set_xlabel('Iteration')
    axs[int(i/ncols)][i%ncols].set_ylabel('Merge Distance')
    i += 1
fig.suptitle('Distance between merged clusters', fontweight='bold')
for ax in axs[nrows-1, i%ncols:]:
    ax.remove()

# %%
ncuts = 5
clusters_info = {}
for i, method in enumerate(methods):
    merge_dist = linkages[i][:,2]
    merge_dist_diff = np.array([merge_dist[j + 1] - merge_dist[j] for j in range(len(merge_dist) - 1)])
    sorted_merge_dist_diff_it = np.argsort(-merge_dist_diff)
    clusters_info[method] = {}
    clusters_info[method]['thresholds'] = []
    clusters_info[method]['distance_diff'] = []
    clusters_info[method]['clusters'] = []
    clusters_info[method]['n_clusters'] = []
    clusters_info[method]['clusters_sizes'] = []
    for j in range(ncuts):
        clusters_info[method]['thresholds'].append(merge_dist[sorted_merge_dist_diff_it[j]])
        clusters_info[method]['distance_diff'].append(merge_dist_diff[sorted_merge_dist_diff_it[j]])
        clusters = np.array(cut_tree(linkages[i], height=merge_dist[sorted_merge_dist_diff_it[j]]+np.finfo(float).eps)).reshape(-1)
        clusters_info[method]['clusters'].append(clusters)
        clusters_info[method]['n_clusters'].append(np.unique(clusters).shape[0])
        counts = np.bincount(clusters) # TODO: serve per centroid e medoid che assegnano come label id non contigui
        clusters_info[method]['clusters_sizes'].append(counts[counts!=0])

# %%
for method in methods:
    print(method)
    display(pd.DataFrame(clusters_info[method])[['thresholds', 'distance_diff', 'n_clusters', 'clusters_sizes']])

# %% [markdown]
# Gli 0 in centroi-median forse sono dovuti alle inversioni...

# %%
threshold_num = 0
nrows = 2
ncols = 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 16))
clusters = []
avg_c = []
for i, method in enumerate(methods):
    plot_clusters_size(clusters_info[method]['clusters'][threshold_num], axs[int(i/ncols)][i%ncols], color_palette=sns.color_palette('deep'), title=method)
fig.suptitle('Number of points in each cluster (first threshold)', fontweight='bold')
plt.show()

# %%
start_iteration = 13100
nrows = 2
ncols = 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 16))
for i, method in enumerate(methods):
    axs[int(i/ncols)][i%ncols].plot(range(start_iteration, linkages[i].shape[0]), linkages[i][start_iteration:, 2], 'o')
    for j, th in enumerate(clusters_info[method]['thresholds']):
        axs[int(i/ncols)][i%ncols].axhline(th, ls='--', color='C'+str(j), label=f'threshold {j}')
    axs[int(i/ncols)][i%ncols].legend()
    axs[int(i/ncols)][i%ncols].set_title(f'{method} linkage')
    axs[int(i/ncols)][i%ncols].set_xlabel('Iteration')
    axs[int(i/ncols)][i%ncols].set_ylabel('Merge Distance')
    i += 1
fig.suptitle('Distance between merged clusters', fontweight='bold')
for ax in axs[nrows-1, i%ncols:]:
    ax.remove()

# %%
threshold_num=2
nrows = 2
ncols = 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 16))
for i, method in enumerate(methods):
    dendrogram(linkages[i], truncate_mode='lastp', p=50, ax=axs[int(i/ncols)][i%ncols], color_threshold=clusters_info[method]['thresholds'][threshold_num]+np.finfo(float).eps)
    axs[int(i/ncols)][i%ncols].axhline(clusters_info[method]['thresholds'][threshold_num], ls='--', color='k', label='chosen threshold')
    axs[int(i/ncols)][i%ncols].legend()
    axs[int(i/ncols)][i%ncols].set_xlabel('Number of samples if between parenthesis, sample index otherwise', fontsize=6)
    axs[int(i/ncols)][i%ncols].set_ylabel('Euclidean Distance')
    axs[int(i/ncols)][i%ncols].set_title(f'{method} linkage')
    i += 1
fig.suptitle('Dendrograms with different linkage methods', fontweight='bold')
for ax in axs[nrows-1, i%ncols+1:]:
    ax.remove()

# %%
# TODO: sul libro si parla di heirarchical f-measure (nell'ambito di supervised validation)
# di internal metrics SSE rispetto al centroide non ha senso, forse silhouette è l'unico sensato
# compare singleton clusters found by heirarchical with noise points found by DBSCAN
# inspect singleton clusters?
# clusters


