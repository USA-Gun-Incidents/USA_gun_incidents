# %% [markdown]
# # Hierarchical Clustering

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage, cophenet, cut_tree
from clustering_utils import *
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
import seaborn as sns

# %%
# import dataset
incidents_df = pd.read_csv('../data/incidents_cleaned_indicators.csv', index_col=False)
incidents_df.drop("Unnamed: 0", axis=1, inplace=True)

# %%
incidents_df.info()

# %% [markdown]
# ## Dataset preparation
# We already have normalized data, so we just select the better state for clustering analysis

# %%
incidents_df_full = pd.read_csv('../data/incidents_cleaned.csv')
incidents_df_full.head(2)

# %%
# select a subset of records regarding a certain state
incidents_df['state'] = incidents_df_full['state']

state = "ILLINOIS"
incidents_df = incidents_df[incidents_df['state'] == state]
incidents_df.drop('state', axis=1, inplace=True)

# %%
incidents_df.isna().sum()

# %%
incidents_df.dropna(inplace=True)

incidents_df.shape

# %%
# print all indexes for clustering
incidents_df.columns

# %% [markdown]
# ## Clustering

# %%
# clustering
algorithms = ["single", "complete", "average", "ward"]
linkages = []
distance_thresholds = []

for algorithm in algorithms:
    #models.append(AgglomerativeClustering(linkage=algorithm, compute_distances=True).fit(incidents_df))
    linkage_res = linkage(pdist(incidents_df, metric='euclidean'), method=algorithm, metric='euclidean', optimal_ordering=False)

    linkages.append(linkage_res)
    distance_thresholds.append(0.7 * max(linkage_res[:,2]))

# %%
f, axs = plt.subplots(ncols=len(linkages), figsize=(32,7))

for i in range(len(linkages)):    
    axs[i].set_title(algorithms[i])
    axs[i].set_xlabel('IncidentID or (Cluster Size)')
    axs[i].set_ylabel('Distance')
    axs[i].axhline(distance_thresholds[i], ls='--', color='k', label='threshold')
    color_threshold = distance_thresholds[i]

    # Plot the corresponding dendrogram
    dendrogram(linkages[i], truncate_mode="lastp", p=30, leaf_rotation=60, leaf_font_size=8,
               show_contracted=True, ax=axs[i], color_threshold=color_threshold)

plt.suptitle(('Hierarchical Clustering Dendograms'), fontsize=18, fontweight='bold')
plt.show()

# %%
ncuts = 10 # valutare se aumentarli, possiamo ottenere migliori prestazioni su complete
clusters_info_df = pd.DataFrame([],  columns =  ['method', 'threshold', 'distance_diff', 'cluster_labels', 'n_clusters', 'clusters_sizes'])
silhouette_scores = []
for i, algorithm in enumerate(algorithms):
    merge_dist = linkages[i][:,2]
    merge_dist_diff = np.array([merge_dist[j + 1] - merge_dist[j] for j in range(len(merge_dist) - 1)])
    sorted_merge_dist_diff_it = np.argsort(-merge_dist_diff)

    best_threshold = 0
    best_distance_diff = 0
    best_labels = []
    best_n_clusters = 0
    best_cluster_sizes = []
    best_silhouette_score = 0
    for j in range(ncuts):
        clusters = np.array(cut_tree(linkages[i], height=merge_dist[sorted_merge_dist_diff_it[j]] + np.finfo(float).eps)).reshape(-1)
        n_clusters = np.unique(clusters).shape[0]
        # check silhouette score
        if(n_clusters > 1):
            silhouette_avg = silhouette_score(incidents_df, clusters)
            if silhouette_avg > best_silhouette_score:
                best_threshold = merge_dist[sorted_merge_dist_diff_it[j]]
                best_distance_diff = merge_dist_diff[sorted_merge_dist_diff_it[j]]
                best_labels = clusters
                best_n_clusters = n_clusters
                counts = np.bincount(clusters) # TODO: serve per centroid e medoid che assegnano come label id non contigui
                best_cluster_sizes = counts[counts!=0]
                best_silhouette_score = silhouette_avg
    
    clusters_info_df.loc[len(clusters_info_df)] = [algorithm, best_threshold, best_distance_diff, best_labels, best_n_clusters, best_cluster_sizes]
    silhouette_scores.append(best_silhouette_score)

# %%
clusters_info_df

# %%
dm = pdist(incidents_df)
cophenetic_coefs = []
for linkage in linkages:
    cophenetic_matrix = cophenet(linkage)
    cophenetic_coef = np.corrcoef(cophenetic_matrix, dm)[0][1]
    cophenetic_coefs.append(cophenetic_coef)


results_df = pd.DataFrame()
results_df['method'] = clusters_info_df['method']
results_df['optimal n_clusters'] = clusters_info_df['n_clusters']
results_df['distance threshold'] = clusters_info_df['threshold']
results_df['silhouette score'] = silhouette_scores
results_df['cophenetic correlation coefficient'] = cophenetic_coefs
results_df.set_index(['method'], inplace=True)
results_df

# %% [markdown]
# # Clustering evaluation

# %% [markdown]
# MIN
# - pro: gestisce bene forme non ellittiche
# - contro: molto suscettibile agli outliers
# 
# MAX
# - pro: robusto contro noise e outliers
# - contro: funziona bene principalmente su cluster globulari
#           se ho un cluster (o più di uno) più grande degli altri, max tende a separarlo
# 
# AVG
# - pro: più robusto contro noise e outilers rispetto a min
# - contro: funziona bene principalmente su cluster globulari
# 
# WARD
# - pro: più robusto contro noise e outilers rispetto a min
# - contro: funziona bene principalmente su cluster globulari
# 
# TODO: confrontare k-means con WARD (stesso k)

# %%
# correlation matrix for each algorithm
RANDOM_STATE = 42
X = incidents_df.values
for i in range(len(algorithms)):
    plot_distance_matrices(X=X, n_samples=500, clusters=clusters_info_df.loc[i]['cluster_labels'], random_state=RANDOM_STATE)

# %%
for i in range(len(algorithms)):
    fig, axs = plt.subplots(1)
    plot_clusters_size(clusters_info_df.loc[i]['cluster_labels'], ax=axs, title='Clusters size')
    fig.show()

# %%
start_iteration = 1
fig, axs = plt.subplots(ncols=4, figsize=(28, 8))
for i, algorithm in enumerate(algorithms):
    axs[i].plot(range(start_iteration, linkages[i].shape[0]), linkages[i][start_iteration:, 2], 'o')
    # axs[i].axhline(optimal_heights[i], ls='--', color='k', label='best cut threshold')
    axs[i].axhline(distance_thresholds[i], ls='--', color='k', label='default threshold')
    axs[i].legend()
    axs[i].set_title(f'{algorithm} linkage')
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel('Merge Distance')
fig.suptitle('Distance between merged clusters', fontweight='bold')

# %%
# FIXME: non mi dice così tanto in realtà...

cluster_column_name = algorithms
incidents_df_cluster = incidents_df.copy()

for i in range(len(algorithms)):
    incidents_df_cluster[algorithms[i]] = clusters_info_df.loc[i]['cluster_labels']
    scatter_by_cluster(incidents_df_cluster,
                    ['latitude',
                     'longitude',
                     'avg_age_participants',
                     'age_range',
                     'n_participants'
                    ],
                    algorithms[i],
                    figsize=(15, 10),
                    color_palette=sns.color_palette(n_colors=clusters_info_df.loc[i]['n_clusters']))

# %%
for i in range(len(algorithms)):
    plot_boxes_by_cluster(incidents_df_cluster,
                        incidents_df.columns,
                        algorithms[i],
                        figsize=(15, 35),
                        title=('Box plots of features by cluster - ' + algorithms[i]))

# %%
for i in range(len(algorithms)):
    plot_violin_by_cluster(
        incidents_df_cluster,
        incidents_df.columns,
        algorithms[i],
        figsize=(15, 20),
        title=('Violin plots of features by cluster- ' + algorithms[i])
    )

# %%
for feature in incidents_df.columns:
    plot_hists_by_cluster(
        df=incidents_df_cluster,
        feature=feature,
        cluster_column='ward',
        title=f'Distribution of {feature} in each cluster',
        color_palette=sns.color_palette(n_colors=int(results_df.loc['ward']['optimal n_clusters']) + 1)
    )


