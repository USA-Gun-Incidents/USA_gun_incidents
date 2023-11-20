# %%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cut_tree, cophenet
from sklearn.preprocessing import MinMaxScaler
import utm
from clustering_utils import *
%matplotlib inline
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
    'latitude_proj', 'longitude_proj', 'location_importance', 'city_entropy', 'address_entropy',
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
    'centroid', # could lead to strange results: https://stats.stackexchange.com/questions/26769/cluster-analysis-in-r-produces-reversals-on-dendrogram
    'median', # could lead to strange results (see above)
    'ward'
]
linkages = []
default_distance_thresholds = []

# %%
nrows = 2
ncols = 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 16))
id = 0
for method in methods:
    dl = linkage(dm, method=method, metric='euclidean', optimal_ordering=False) # optimal_ordering=False otherwise it takes too long
    linkages.append(dl)
    distance_threshold = 0.7*max(dl[:,2]) # 70% of the maximum distance between two clusters that are merged
    default_distance_thresholds.append(distance_threshold)
    dendrogram(dl, truncate_mode='lastp', p=50, ax=axs[int(id/ncols)][id%ncols])
    axs[int(id/ncols)][id%ncols].axhline(distance_threshold, ls='--', color='k', label='threshold')
    axs[int(id/ncols)][id%ncols].legend()
    axs[int(id/ncols)][id%ncols].set_xlabel('Number of samples if between parenthesis, sample index otherwise', fontsize=6)
    axs[int(id/ncols)][id%ncols].set_ylabel('Euclidean Distance')
    axs[int(id/ncols)][id%ncols].set_title(f'{method} linkage\n default threshold at {distance_threshold:.2f}')
    id += 1
fig.suptitle('Dendrograms with different linkage methods', fontweight='bold')
for ax in axs[nrows-1, id%ncols+1:]:
    ax.remove()


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
plot_cluster_map(X, 'single', indicators_df.columns)

# %%
start_iteration = 13000
nrows = 2
ncols = 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 16))
id = 0
for i, method in enumerate(methods):
    axs[int(id/ncols)][id%ncols].plot(range(start_iteration, linkages[i].shape[0]), linkages[i][start_iteration:, 2], 'o')
    axs[int(id/ncols)][id%ncols].axhline(default_distance_thresholds[i], ls='--', color='k', label='default threshold')
    axs[int(id/ncols)][id%ncols].legend()
    axs[int(id/ncols)][id%ncols].set_title(f'{method} linkage')
    axs[int(id/ncols)][id%ncols].set_xlabel('Iteration')
    axs[int(id/ncols)][id%ncols].set_ylabel('Merge Distance')
    id += 1
fig.suptitle('Distance between merged clusters', fontweight='bold')
for ax in axs[nrows-1, id%ncols+1:]:
    ax.remove()

# %%
for i, method in enumerate(methods):
    merge_dist = linkages[i][:,2]
    merge_dist_diff = np.array([merge_dist[j + 1] - merge_dist[j] for j in range(len(merge_dist) - 1)])
    sorted_diff = np.argsort(-merge_dist_diff)
    print(f'{method} method')
    print('Higher merge distances')
    print(merge_dist_diff[sorted_diff][:10])
    print('Iteration')
    print(sorted_diff[:10]+2) # +2 right?
    print('-----')

# %%
clusters_default_threshold = []
clusters_inconsistent = []
for i, method in enumerate(methods):
    clusters_default_threshold.append(cut_tree(linkages[i], height=default_distance_thresholds[i])) # TODO: try other thresholds
    clusters_inconsistent.append(fcluster(linkages[i], t=default_distance_thresholds[i], criterion='inconsistent'))
    # TODO: try (and understand) other fcluster criterion, can also set the number of desired clusters

# thresholds are typically identified via: silhouette plot, Dunn’s validity index, Hubert's gamma, G2/G3 coefficient, corrected Rand index, cophenetic distance

# %%
# inspect singleton clusters from single linkage method
unique, counts = np.unique(clusters_default_threshold[0], return_counts=True)
singleton_clusters = unique[counts==1]


# %%
for i in singleton_clusters:
    display(incidents_df.loc[clusters_default_threshold[0]==i])

# %%
# TODO: sul libro si parla di heirarchical f-measure (nell'ambito di supervised validation)
# di internal metrics SSE rispetto al centroide non ha senso, forse silhouette è l'unico sensato
# compare singleton clusters found by heirarchical with noise points found by DBSCAN


