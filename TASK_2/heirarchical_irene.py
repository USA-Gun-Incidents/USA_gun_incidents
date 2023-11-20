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
categorical_features = [
    'year', 'month', 'day_of_week', 'party', #'state', 'address_type', 
    'firearm', 'air_gun', 'shots', 'aggression', 'suicide',
    'injuries', 'death', 'road', 'illegal_holding', 'house',
    'school', 'children', 'drugs', 'officers', 'organized', 'social_reasons',
    'defensive', 'workplace', 'abduction', 'unintentional'
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
for ax in axs[nrows-1, id%ncols:]:
    ax.remove()


# %%
dm = pdist(X)
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
clusters_default_threshold = []
clusters_inconsistent = []
for i, method in enumerate(methods):
    clusters_default_threshold.append(cut_tree(linkages[i], height=default_distance_thresholds[i])) # TODO: try other thresholds
    clusters_inconsistent.append(fcluster(linkages[i], t=default_distance_thresholds[i], criterion='inconsistent'))
    # TODO: try (and understand) other fcluster criterion

# thresholds are typically identified via: silhouette plot, Dunn’s validity index, Hubert's gamma, G2/G3 coefficient, corrected Rand index, cophenetic distance

# %%
# inspect singleton clusters from single linkage method
unique, counts = np.unique(clusters_default_threshold[0], return_counts=True)
singleton_clusters = unique[counts==1]


# %%
for i in singleton_clusters:
    display(incidents_df.loc[clusters_default_threshold[0]==i])


