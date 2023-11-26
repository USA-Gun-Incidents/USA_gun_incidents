# -*- coding: utf-8 -*-
# %% [markdown]
# # Self organizing maps
#
# From the [Pyclustering](https://github.com/annoviko/pyclustering/) library we already used the X-means algorithm, comparing it with K-means. In this notebook we will experiment with Self Organizing Maps (SOM). SOM is an unsupervised neural network model that could be used both for dimensionality reduction, visualization of high-dimensional data and clustering. This model organizes the input data into a lower-dimensional (typically 2 dimensional) grid of neurons (numerical vectors of 'weights'), where each neuron represents a cluster prototype. Neurons weights are randomly initialized and, during training, the network adjusts its weights to map similar input patterns to neighboring locations on the grid, preserving the topological relationships of the input data.
#
# First, we import the libraries:

# %%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from pyclustering.nnet.som import som_parameters, som, type_conn
from clustering_utils import *

# %% [markdown]
# We load the data and prepare it for the clustering. We will use the same dataset used with K-means and X-means.

# %%
# load the data
incidents_df = pd.read_csv('../data/incidents_indicators.csv', index_col=0)
# load the names of the features to use for clustering
features_to_cluster = json.loads(open('../data/indicators_names.json').read())
# FIXME: da fare in indicators
features_to_cluster = [feature for feature in features_to_cluster if feature not in ['lat_proj', 'lon_proj']]
# drop nan
incidents_df = incidents_df.dropna(subset=features_to_cluster).reset_index(drop=True)
# initialize a colum for the clustering labels
incidents_df['cluster'] = None
# project on the indicators
indicators_df = incidents_df[features_to_cluster]
# apply MinMaxScaler
minmax_scaler = MinMaxScaler()
X = minmax_scaler.fit_transform(indicators_df.values)

# %% [markdown]
# Scrivere che c'è codice in libreria per clustering che wrappa quello che facciamo sotto ma è meno customizzabile...
#
# In the library [Pyclustering](https://github.com/annoviko/pyclustering/) there is ...
#
# Below we define the parameters of the algorithm:
# - we use the default parameters for the SOM algorithm, that are:
#     - adaptation_threshold=0.001, used if autostop=True, determines the threshold for stopping the training
#     - init_learn_rate=0.1
#     - init_radius=2 if cols+rows>4
#     - init_type=    distributed in line with uniform grid
# - we use a 3x3 grid
# - each grid cell has at most 4 neighbors

# %%
som_params = som_parameters()
rows = 3
cols = 3
n_clusters = rows*cols
structure = type_conn.grid_four
network = som(rows, cols, structure, som_params)

# %% [markdown]
# We train the model using the autostop criterion, which stops the training when ... TODO: commentare

# %%
network.train(X, autostop=True, epochs=100000)

# %% [markdown]
# We save the clustering results:

# %%
for i in range(n_clusters):
    incidents_df.loc[network.capture_objects[i], 'cluster'] = i
prototypes = np.array(network.weights)
clusters = np.array(incidents_df['cluster'].astype(int))

# %% [markdown]
# We visualize the size of each cluster on the grid:

# %%
fig, axs = plt.subplots(1, figsize=(6,4))
award_mtx = np.array(network._award).reshape(network._rows, network._cols)
sns.heatmap(award_mtx, annot=True, ax=axs, fmt='.0f')
axs.set_xticks([])
axs.set_yticks([])
plt.title("Number of point per cluster")

# %% [markdown]
# This operation was also implemented in the library, but we notice that the colors associated to the grid cells are not in line with the labels. The implementation of the library is probably missing a transposition of the labels.

# %%
network.show_winner_matrix()

# %% [markdown]
# We also visualize the cluster sizes as a bar plot:

# %%
fig, axs = plt.subplots(1, figsize=(25,5))
plot_clusters_size(
    clusters=incidents_df['cluster'],
    ax=axs,
    title='SOM clusters size',
    color_palette=sns.color_palette('tab10')
)


# %% [markdown]
# There are two big clusters, 2 clusters with medium size and 5 smaller clusters. Cluster labels are assigned in the order of the grid cells, from left to right and from top to bottom.

# %% [markdown]
# To explore the clustering result and the topology of the grid, we will color each cell according to the mean or mode - depending on the type of the variable - of the features of the corresponding cluster. The following function computes the mean or the mode of the given feature for the incidents in the same cluster:

# %%
def agg_feature_by_cluster(network, df, feature, agg_fun):
    agg_feature_per_cluster = []
    for i in range(n_clusters):
        agg_feature = df.iloc[network.capture_objects[i]][feature].agg(agg_fun)
        if agg_fun=='mode':
            agg_feature = agg_feature.values[0]
        agg_feature_per_cluster.append(agg_feature)
    return agg_feature_per_cluster

# %% [markdown]
# First, we explore numerical features:

# %%
ncols = 3
nplots = len(features_to_cluster)
nrows = int(nplots/ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30,25))
for i, feature in enumerate(features_to_cluster):
    avg_feature_per_cluster = agg_feature_by_cluster(network, indicators_df, feature=feature, agg_fun='mean')
    avg_feature_mtx = np.array(avg_feature_per_cluster).reshape(network._rows, network._cols)
    sns.heatmap(avg_feature_mtx, ax=axs[int(i/ncols)][i%ncols], annot=True)
    axs[int(i/ncols)][i%ncols].set_title(feature)
    axs[int(i/ncols)][i%ncols].set_xticks([])
    axs[int(i/ncols)][i%ncols].set_yticks([])
for ax in axs[nrows-1, i%ncols+1:]:
    ax.remove()
f.suptitle("Average value of each feature per cluster", fontweight='bold');

# %% [markdown]
# We notice that:
# - the up-left corner of the grid groups incidents with higher values of n_injured_prop
# - the up-right corner groups incidents with higher values of n_arrested_prop
# - the down-left corner groups incidents with higher values of n_unharmed_prop
# - the down-right corner groups incidents with higher values of n_killed_prop
# - the center of the grid groups incidents with higher values of suprisal_age_groups, surprisal_n_males, suprisal_characteristics and n_participants; and lower values of n_males_prop
#
# The groups discovered are similar to the ones discovered with K-means (despite the fact that here we are grouping points in 9 clusters).

# %% [markdown]
# Now we explore the distribution of categorical features on the grid:

# %%
incidents_df['unharmed'] = incidents_df['n_unharmed'] > 0
incidents_df['arrested'] = incidents_df['n_arrested'] > 0
features_to_mode = [
    'children',
    'shots',
    'aggression',
    'suicide',
    'injuries',
    'death',
    'drugs',
    'illegal_holding',
    'unharmed',
    'arrested'
]

nplots = len(features_to_mode)
ncols = 3
nrows = int(nplots / ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,8))
for i, feature in enumerate(features_to_mode):
    mode_per_cluster = agg_feature_by_cluster(network, incidents_df, feature=feature, agg_fun='mode')
    unique_values = incidents_df[feature].unique()
    unique_values_map = {unique_value: i for i, unique_value in enumerate(unique_values)}
    num_mode_per_cluster = [unique_values_map[mode] for mode in mode_per_cluster]
    num_mode_per_cluster_mtx = np.array(num_mode_per_cluster).reshape(network._rows, network._cols)
    mode_per_cluster_mtx = np.array(mode_per_cluster).reshape(network._rows, network._cols)
    sns.heatmap(num_mode_per_cluster_mtx, ax=axs[int(i/ncols)][i%ncols], annot=mode_per_cluster_mtx, cmap='tab10', cbar=False, fmt='')
    axs[int(i/ncols)][i%ncols].set_title(feature)
    axs[int(i/ncols)][i%ncols].set_xticks([])
    axs[int(i/ncols)][i%ncols].set_yticks([])
for ax in axs[nrows-1, i%ncols+1:]:
    ax.remove()
f.suptitle("Most frequent value for each feature in each cluster", fontweight='bold');

# %% [markdown]
# The distribution of the variables is in line with the one observations made regarding the numerical features.
#
# We now visualize the U-matrix, where each cell is colored according to the average distance between the weights of the cell and its neighbors:

# %%
network.show_distance_matrix()

# %% [markdown]
# We notice that the cell in the middle of the grid is similar to all the adjacent cells (it has not a unique identity). Instead, the prototipe of the cluster in the down-left corner is well separated from the adjacent prototipes.

# %% [markdown]
# We visualize clusters in the principal components space:

# %%
pca = PCA()
X_pca = pca.fit_transform(X)
palette = [sns.color_palette('tab10')[i] for i in range(n_clusters)]
scatter_pca_features_by_cluster(
    X_pca=X_pca,
    n_components=6,
    clusters=incidents_df['cluster'],
    palette=palette,
    hue_order=None,
    title='Clusters in PCA space'
)

# %% [markdown]
# Clusters are quite separated in the feature space of the first principal components.

# %%
for j in range(0, len(prototypes)):
    plt.plot(prototypes[j], marker='o', label='Cluster %s' % j, c=sns.color_palette('tab10')[j])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster)), features_to_cluster, rotation=90)
plt.legend(fontsize=10)
plt.title(f'Prototypes of SOM clusters');

# %% [markdown]
# The features that differentiate the most the clusters prototipes are, as expected, the following:
# - n_killed_prop
# - n_injured_prop
# - n_unharmed_prop
# - n_arrested_prop

# %%
plot_boxes_by_cluster(
    df=incidents_df,
    features=features_to_cluster,
    cluster_column='cluster',
    figsize=(15, 35),
    title='Box plots of features by cluster'
)

# %%
clustering_scores = {}
se_per_point = compute_se_per_point(X=X, clusters=clusters, centroids=prototypes)
clustering_scores['SSE'] = se_per_point.sum()
clustering_scores['BSS'] = compute_bss_per_cluster(X=X, clusters=incidents_df['cluster'], centroids=prototypes, weighted=True).sum()
clustering_scores['davies_bouldin_score'] = davies_bouldin_score(X=X, labels=incidents_df['cluster'])
clustering_scores['calinski_harabasz_score'] = calinski_harabasz_score(X=X, labels=incidents_df['cluster'])
clustering_scores['silhouette_score'] = silhouette_score(X=X, labels=incidents_df['cluster'])
pd.DataFrame(clustering_scores, index=['SOM'])

# %% [markdown]
# TODO: COMMENTARE

# %%
fig, axs = plt.subplots(1, figsize=(8,5))
silhouette_per_point = silhouette_samples(X=X, labels=incidents_df['cluster'])
plot_scores_per_point(
    score_per_point=silhouette_per_point,
    clusters=incidents_df['cluster'],
    score_name='Silhouette score', ax=axs,
    title=f'Silhouette score for SOM clustering',
    color_palette=sns.color_palette('tab10'),
    minx=-0.02
)

# %% [markdown]
# The majority of points in cluster 4 have a negative silhouette score.

# %%
sse_feature = []
for i in range(X.shape[1]):
    sse_feature.append(compute_se_per_point(X=X[:,i], clusters=clusters, centroids=prototypes[:,i]).sum())

plt.figure(figsize=(15, 5))
sse_feature_sorted, clustering_features_sorted = zip(*sorted(zip(sse_feature, features_to_cluster)))
plt.bar(range(len(sse_feature_sorted)), sse_feature_sorted)
plt.xticks(range(len(sse_feature_sorted)), clustering_features_sorted)
plt.xticks(rotation=90)
plt.ylabel('SSE')
plt.xlabel('Feature')
plt.title('SSE per feature');

# %%
se_per_point = compute_se_per_point(X=X, clusters=clusters, centroids=prototypes)
indices_of_top_contributors = np.argsort(se_per_point)[-5:]
incidents_df.iloc[indices_of_top_contributors]

# %%
fig, axs = plt.subplots(1, figsize=(10,5))
plot_scores_per_point(
    score_per_point=se_per_point,
    clusters=clusters,
    score_name='SE',
    ax=axs,
    color_palette=sns.color_palette('tab10'),
    minx=-0.1
)

# %%
# compute cohesion for each cluster
se_per_cluster = np.zeros(n_clusters)
sizes = np.ones(prototypes.shape[0])
for i in range(n_clusters):
    se_per_cluster[i] = np.sum(se_per_point[np.where(clusters == i)[0]])/sizes[i]
# compute separation for each cluster
bss_per_cluster = compute_bss_per_cluster(X, clusters, prototypes, weighted=True)
# compute average silhouette score for each cluster
silhouette_per_cluster = np.zeros(n_clusters)
for i in range(n_clusters):
    silhouette_per_cluster[i] = silhouette_per_point[np.where(clusters == i)[0]].mean()
# visualize the result
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
axs[0].bar(range(n_clusters), se_per_cluster, color=sns.color_palette('tab10'))
axs[0].set_ylim(15000, 0)
axs[0].set_title('Cohesion')
axs[0].set_ylabel('SSE')
axs[1].bar(range(n_clusters), bss_per_cluster, color=sns.color_palette('tab10'))
axs[1].set_title('Separation')
axs[1].set_ylabel('BSS')
axs[2].bar(range(n_clusters), silhouette_per_cluster, color=sns.color_palette('tab10'))
axs[2].set_title('Silhouette')
axs[2].set_ylabel('Silhouette score')

for i in range(3):
    axs[i].set_xlabel('Cluster')
    axs[i].set_xticks(range(n_clusters))
    axs[i].set_xticklabels(range(n_clusters))

plt.suptitle('Cohesion and separation measures for each cluster', fontweight='bold')

# %%
clusters = incidents_df['cluster'].to_numpy()
plot_distance_matrices(X, n_samples=5000, clusters=clusters)

# %%
incidents_df['unharmed'] = incidents_df['n_unharmed'] > 0
incidents_df['arrested'] = incidents_df['n_arrested'] > 0
compute_permutation_invariant_external_metrics(
    incidents_df,
    'cluster',
    ['shots', 'aggression', 'suicide', 'injuries', 'death', 'drugs', 'illegal_holding', 'unharmed', 'arrested']
)

# %%
write_clusters_to_csv(clusters, f'./SOM_clusters.csv')

# %%

# prova anche con std?


