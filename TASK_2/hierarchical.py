# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
#  
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
# 
# # Hierarchical Clustering
# 
# Hierarchical clustering methods generate a series of nested clusters arranged in a hierarchical tree, making them particularly effective for data exhibiting a nested or hierarchical structure. One notable advantage is their flexibility, as they eliminate the need to predefine a fixed number of clusters. However, their time and space efficiency may not be optimal, and they can exhibit sensitivity to noise and outliers.
# 
# Agglomerative clustering starts with the points as individual clusters and, at each iteration, merges the closest pair of clusters. The cluster proximity could be determined using different policies, such as:
# - **single linkage**: defines cluster proximity as the proximity between the closest two points that are in different clusters. This method is good at handling non-elliptical shapes, but is sensitive to noise and outliers.
# 
# - **complete linkage**: defines cluster proximity as the proximity between the farthest two points that are in different clusters. This approach is less susceptible to noise and outliers, but it is biased towards globular clusters. Moreover it tends to break large clusters.
# 
# - **average linkage**: is an intermediate approach between the single and complete link approaches, it defines cluster proximity as the distance between all the points in the clusters.
# 
# - **ward**: defines cluster proximity as the increase in squared error when two clusters are merged. It is robust to noise and outliers but it's biased towards globular clusters.
# 
# We import the libraries:

# %%
import pandas as pd
import numpy as np
import json
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage, cophenet, cut_tree
from scipy.spatial.distance import pdist
from clustering_utils import *
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns
RESULTS_DIR = "../data/clustering_labels/"
RANDOM_STATE = 42

# %% [markdown]
# We load the data and prepare it for the clustering (we will restrict the clustering to the incidents happened in Illinois):

# %%
# load the data
incidents_df = pd.read_csv('../data/incidents_indicators.csv', index_col=0)
# load the names of the features to use for clustering
features_to_cluster = json.loads(open('../data/indicators_names.json').read())
# select only the incidents happened in Illinois
incidents_df = incidents_df[incidents_df['state'] == 'ILLINOIS']
# for clustering we will use all the extracted indicators except the projected coordinates
features_to_cluster = [feature for feature in features_to_cluster if feature not in ['lat_proj', 'lon_proj']]
# drop nan
incidents_df = incidents_df.dropna(subset=features_to_cluster)
# project on the indicators
indicators_df = incidents_df[features_to_cluster]

# %% [markdown]
# We scale the data applying MinMaxScaler:

# %%
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(indicators_df.values)

# %% [markdown]
# We apply the clustering algorithms saving the results:

# %%
algorithms = ["single", "complete", "average", "ward"]
linkages = []
distance_thresholds = []
dm = pdist(X_minmax, metric='euclidean')
for algorithm in algorithms:
    linkage_res = linkage(dm, method=algorithm, metric='euclidean', optimal_ordering=False)
    linkages.append(linkage_res)
    distance_thresholds.append(0.7 * max(linkage_res[:,2]))

# %% [markdown]
# We define a function to plot dendograms:

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

# %% [markdown]
# We display the dendograms (using as 'cut height' the default threshold, i.e. 0.7 times the maximum distance between merged clusters):

# %%
plot_dendrograms(linkages, algorithms, distance_thresholds)

# %% [markdown]
# Each proximity measure leads to very different shaped hierarchies.
# 
# Using **single-linkage** the result we get is similar to what we could achieve adding incrementally to a starting clusters points in closest clusters.
# 
# Using **complete-linkage** the tree is more balanced, however a cluster is significantly bigger than others (the one containing 4553 points).
# 
# Using **average-linkage** we find, as expected, a compromise between the results obtained using single and complete linkage.
# 
# Using **ward** we get the most balanced results, both in terms of tree structure and in terms of clusters size.

# %% [markdown]
# ## Clustering evaluation
# 
# ### Internal indices

# %% [markdown]
# To evaluate the clustering results with the different methods we compute the cophenetic correlation coefficient. The cophenetic distance between two objects is the proximity at which an agglomerative hierarchical clustering technique puts the objects in the same cluster for the first time. In a cophenetic distance matrix, the entries are the cophenetic distances between each pair of objects. The cophenetic correlation coefficient measures the correlation between the entries of the cophenetic distance metric and the dissimilarity matrix.

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
# In the following, we search for the best cut computing the silhouette score for each clustering obtained by cutting the tree at the 10 merging steps with the greatest distance between merged clusters.

# %%
ncuts = 10
clusters_info = {}
clusters_info['method'] = []
clusters_info['cut_height'] = []
clusters_info['merging_difference'] = []
clusters_info['cluster_labels'] = []
clusters_info['n_clusters'] = []
clusters_info['clusters_sizes'] = []
clusters_info['silhouette_score'] = []
silhouette_scores = []

for i, algorithm in enumerate(algorithms):
    print(f"Method: {algorithm}-linkage")
    clusters_info['method'].append(algorithm)
    merge_dist = linkages[i][:,2]
    merge_dist_diff = np.array([merge_dist[j + 1] - merge_dist[j] for j in range(len(merge_dist) - 1)])
    sorted_merge_dist_diff_it = np.argsort(-merge_dist_diff)

    best_threshold = 0
    best_distance_diff = 0
    best_labels = []
    best_n_clusters = 0
    best_cluster_sizes = []
    best_silhouette_score = -1
    
    for j in range(ncuts):
        print(f'Computing silhouette score for cut {j+1}/{ncuts}...')
        clusters = np.array(cut_tree(linkages[i], height=merge_dist[sorted_merge_dist_diff_it[j]])).reshape(-1)
        n_clusters = np.unique(clusters).shape[0]
        if (n_clusters > 1):
            silhouette_avg = silhouette_score(X_minmax, clusters)
            if silhouette_avg > best_silhouette_score:
                best_threshold = merge_dist[sorted_merge_dist_diff_it[j]]
                best_distance_diff = merge_dist_diff[sorted_merge_dist_diff_it[j]]
                best_labels = clusters
                best_n_clusters = n_clusters
                counts = np.bincount(clusters)
                best_cluster_sizes = counts[counts!=0]
                best_silhouette_score = silhouette_avg
    
    clusters_info['cut_height'].append(best_threshold)
    clusters_info['merging_difference'].append(best_distance_diff)
    clusters_info['cluster_labels'].append(best_labels)
    clusters_info['n_clusters'].append(best_n_clusters)
    clusters_info['clusters_sizes'].append(best_cluster_sizes)
    clusters_info['silhouette_score'].append(best_silhouette_score)
    silhouette_scores.append(best_silhouette_score)

clusters_info_df = pd.DataFrame(clusters_info)
clusters_info_df.set_index(['method'], inplace=True)
clusters_info_df['cophenetic_correlation_coefficient'] = cophenetic_coefs
clusters_info_df

# %% [markdown]
# Ward's method achieves the highest silhouette score.

# %% [markdown]
# We plot the dendograms displaying the best cut heights:

# %%
plot_dendrograms(linkages, algorithms, clusters_info_df['cut_height'])

# %% [markdown]
# We display the distance between merged clusters at each iteration:

# %%
fig, axs = plt.subplots(ncols=4, figsize=(28, 8))
for i, method in enumerate(clusters_info_df.index):
    axs[i].plot(range(0, linkages[i].shape[0]-1), linkages[i][0:-1, 2], 'o')
    axs[i].axhline(distance_thresholds[i], ls='--', color='k', label='default threshold')
    axs[i].axhline(clusters_info_df.loc[method]['cut_height'], ls='--', color='r', label='best cut threshold')
    axs[i].legend()
    axs[i].set_title(f'{method} linkage')
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel('Merge Distance')
fig.suptitle('Distance between merged clusters', fontweight='bold');

# %% [markdown]
# The best cuts according to the silhouette score do not correspond to the largest distances between merged clusters.

# %% [markdown]
# We plot the silhouette scores for each point:

# %%
fig, axs = plt.subplots(2, 2, figsize=(20,15))
x_axs = 0
y_axs = 0
for i, method in enumerate(clusters_info_df.index):
    silhouette_per_point = silhouette_samples(X=X_minmax, labels=clusters_info_df.loc[method]['cluster_labels'])
    if i != 0 and i%2 == 0:
        x_axs += 1
        y_axs = 0
    plot_scores_per_point(
        score_per_point=silhouette_per_point,
        clusters=clusters_info_df.loc[method]['cluster_labels'],
        score_name='Silhouette score',
        ax=axs[x_axs][y_axs],
        title=(f'Silhouette score for Hierachical Clustering - {method} (cpcc: {clusters_info_df.loc[method]["cophenetic_correlation_coefficient"]:.3f})'),
    )
    y_axs += 1

# %% [markdown]
# With complete linkage almost every cluster has some points with negative silhouette score. With ward linkage cluster 6, 5 and 2 don't have any points with negative silhouette score. Cluster has 0 the lowest values of silhouette score.
# 
# We visualize the size of the clusters:

# %%
fig, axs = plt.subplots(2, 2, figsize=(20,15))
x_axs = 0
y_axs = 0
for i, method in enumerate(clusters_info_df.index):
    if i != 0 and i%2 == 0:
        x_axs += 1
        y_axs = 0
    plot_clusters_size(clusters_info_df.loc[method]['cluster_labels'], ax=axs[x_axs][y_axs], title=f'{method} linkage')
    y_axs += 1
fig.suptitle("Cluster sizes", fontweight='bold');

# %% [markdown]
# With Ward's method clusters are more balanced.

# %% [markdown]
# We visualize the distance matrix sorted by cluster computed on a stratified subsample of 500 points for complete and Ward linkage:

# %%
plot_distance_matrices(X=X_minmax, n_samples=500, clusters=clusters_info_df.loc['complete']['cluster_labels'], random_state=RANDOM_STATE)
plot_distance_matrices(X=X_minmax, n_samples=500, clusters=clusters_info_df.loc['ward']['cluster_labels'], random_state=RANDOM_STATE)

# %% [markdown]
# This kind of evaluation is not very informative for hierarchical clustering, since the clusters could not be globular and may be intertwined with other clusters.
# Nevertheless, both the matrices have a block diagonal structure, meaning that clusters are well separated.
# 
# Since with Ward's method we get the best results in terms of silhouette score and cluster size, we will use this method for the following analysis.

# %%
best_method = 'ward'

# %% [markdown]
# We visualize the cluster in the bidimensional feature spaces obtained pairing some features used for the clustering:

# %%
incidents_df['cluster'] = clusters_info_df.loc[best_method]['cluster_labels']
features_to_scatter = [
    'n_child_prop',
    'n_teen_prop',
    'n_killed_prop',
    'n_injured_prop',
    'n_unharmed_prop',
    'n_males_prop',
    'n_arrested_prop',
    'n_participants'
]
scatter_by_cluster(
    df=incidents_df,
    features=features_to_scatter,
    cluster_column='cluster',
    figsize=(15, 34)
)

# %% [markdown]
# In some feature spaces clusters are almost well separated.

# %% [markdown]
# We apply the principal component analysis to the scaled matrix of indicators:

# %%
pca = PCA()
X_pca = pca.fit_transform(X_minmax)

# %% [markdown]
# We display the explained variance ratio of the components:

# %%
exp_var_pca = pca.explained_variance_ratio_

diff_var = []
for i, var in enumerate(exp_var_pca[:-1]):
    diff_var.append( var-exp_var_pca[i+1])
xtick = []
gap = 0
for i, var in enumerate(diff_var):
    xtick.append(i+gap)
    if i != 0 and diff_var[i-1] <= var:
        gap += 0.5
        if gap == 0.5:
            plt.axvline(x = i+gap+0.25, color = 'green', linestyle = '-.', alpha=0.5, label='possible cut')
        else:
            plt.axvline(x = i+gap+0.25, color = 'green', linestyle = '-.', alpha=0.5)
print(xtick)
xtick.append(xtick[-1]+1.5)

plt.bar(xtick, exp_var_pca, align='center')
plt.plot(xtick[1:], diff_var, label='difference from prevoius variance', color='orange')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component')
plt.title('Explained variance by principal component')
plt.xticks(xtick, range(exp_var_pca.shape[0]));
plt.legend();

# %% [markdown]
# The first 6 components contribute the most to the overall variance in the dataset.
# 
# We visualize the clusters in the feature spaces obtained by pairing the first 6 principal components:

# %%
n_clusters = len(np.unique(incidents_df['cluster']))
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
# In the feature spaces involving the third principal components clusters are almost well separated.

# %% [markdown]
# We now visualize the distributions of the features used to cluster the data in each cluster:

# %%
plot_boxes_by_cluster(
    df=incidents_df,
    features=indicators_df.columns,
    cluster_column='cluster',
    figsize=(15, 30),
    title=(f'Box plots of features by cluster - {best_method}')
)

# %% [markdown]
# We observe that:
# - Cluster 0 has higher values for age_range, n_unharmed_prop_, n_participants (mass shootings?)
# - Cluster 1 has higher values for n_teen_prop, avg_age below 40, n_injured_prop near 1
# - Clsuter 2 has n_injured_prop near 1, n_killed_prop near 0, n_child_prop near 0, n_unharmed_prop near 0, n_arrested_prop near 0 (incidents involving only injured people)
# - Cluster 3 has higher values for surprisal_adrres_type, surprisal_characteristics and n_arrested_prop, n_injured_prop near 0, n_killed_prop centered in 0 but with outliers (unusual location and characteristics, no injuries, sometimes deaths)
# - Cluster 4 has higher values for n_teen_prop, avg_age below 40, higher values for n_participants, n_killed_prop, n_unharmed_prop and n_arrested_prop centered in 0 but with a wide interquartile range (mass shootings?)
# - Cluster 5 has n_killed_prop near 1, n_injured_prop and n_unharmed_prop near 0, few n_participants (fatal incidents with few participants, suicides?)
# - Cluster 6 has n_injured_prop near 1, n_killed_prop near 0, n_unharmed_prop near 0, n_arrested_prop near 0, n_males_prop near 0, low values for n_participants (incidents involving an injured female)

# %% [markdown]
# Now we visualize the same information using histograms and comparing the distributions of features in each cluster with the distribution in the whole dataset:

# %%
for feature in indicators_df.columns:
    plot_hists_by_cluster(
        df=incidents_df,
        feature=feature,
        cluster_column='cluster',
        title=f'Distribution of {feature} in each cluster'
    )

# %% [markdown]
# This visualization confirms what was already observed.
# 
# The attributes with the most different distributions are:
# - n_teen_prop
# - surprisal_age_groups
# - n_killed_prop
# - n_injured_prop
# - n_unharmed_prop
# - suprisal_n_males
# - surprisal_characteristics

# %% [markdown]
# Now we inspect the distribution of the most relevant categorical features within the clusters:

# %%
plot_bars_by_cluster(df=incidents_df, feature='year', cluster_column='cluster')

# %% [markdown]
# Cluster 1 and 2 have less incidents happened in 2017, while cluster 0 and 3 have more incidents happened in 2017.

# %%
plot_bars_by_cluster(df=incidents_df, feature='day_of_week', cluster_column='cluster')

# %% [markdown]
# In cluster 2 the proportion of incidents happened in weekends is higher than in the whole dataset, while in cluster 3 and 4 it is lower.

# %%
plot_bars_by_cluster(df=incidents_df, feature='shots', cluster_column='cluster')

# %% [markdown]
# Most of the non-shooting incidents are in cluster 0, 2, 3 and 4.

# %%
plot_bars_by_cluster(df=incidents_df, feature='suicide', cluster_column='cluster')

# %% [markdown]
# Suicides are mostly in clutser 5 as previously hypothesized.

# %%
plot_bars_by_cluster(df=incidents_df, feature='injuries', cluster_column='cluster')

# %% [markdown]
# Cluster 1, 2 and 6 group incidents with injuries only.

# %%
plot_bars_by_cluster(df=incidents_df, feature='death', cluster_column='cluster')

# %% [markdown]
# Fata incidents are mostly in cluster 0,3 4 and 5. Cluster 5 groups only fatal incidents.

# %%
plot_bars_by_cluster(df=incidents_df, feature='illegal_holding', cluster_column='cluster')

# %% [markdown]
# Incidents tagged with 'illegal_holding' are mostly in cluster 0, 2 and 3.

# %%
plot_bars_by_cluster(df=incidents_df, feature='children', cluster_column='cluster')

# %% [markdown]
# Incidents involving children are mostly in cluster 0.

# %%
plot_bars_by_cluster(df=incidents_df, feature='drugs', cluster_column='cluster')

# %% [markdown]
# Incidents involving drugs are mostly in cluster 0 and 3.

# %%
plot_bars_by_cluster(df=incidents_df, feature='officers', cluster_column='cluster')

# %% [markdown]
# Incidnets involving officers are mostly in cluster 3.

# %%
plot_bars_by_cluster(df=incidents_df, feature='defensive', cluster_column='cluster')

# %% [markdown]
# Defensive incidents are mostly in cluster 0 and 2.

# %%
plot_bars_by_cluster(df=incidents_df, feature='unintentional', cluster_column='cluster')

# %% [markdown]
# Cluster 0 has less unintentional incidents compared to cluster 2.

# %% [markdown]
# ### External indices
# 
# We measure the extent to which the discovered clustering structure matches some categorical features of the dataset, using the following permutation invariant scores:
# - **Adjusted rand score**: this score computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings. It is 0.0 for random labeling, 1.0 when the clusterings are identical and is bounded below by -0.5 for especially discordant clusterings.
# - **Normalized mutual information**: is a normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation). Mutual Information is a function that measures the agreement of the two assignments, ignoring permutations.
# - **Homogeneity**: measure the degree to which each cluster contains only members of a single class; it ranges between 0 and 1, with 1 denoting perfectly homogeneous labeling.
# - **Completeness**: measure the degree to ewhich data points that are members of a given class are also elements of the same cluster; it ranges between 0 and 1, with 1 denoting perfectly complete labeling.

# %%
incidents_df['cluster'] = clusters_info_df.loc[best_method]['cluster_labels']
incidents_df['unharmed'] = incidents_df['n_unharmed'] > 0
incidents_df['arrested'] = incidents_df['n_arrested'] > 0
incidents_df['males'] = incidents_df['n_males'] > 0
incidents_df['females'] = incidents_df['n_females'] > 0
external_scores_df = compute_permutation_invariant_external_metrics(
    incidents_df,
    'cluster',
    ['shots', 'aggression', 'suicide', 'injuries', 'death', 'drugs', 'illegal_holding', 'unharmed', 'arrested', 'males', 'females']
)
external_scores_df.to_csv(RESULTS_DIR + "hierarchical_external_scores.csv")
external_scores_df

# %% [markdown]
# The most homogeneous category is 'arrested', however it is not complete. Completeness is quite low for all the categories.

# %% [markdown]
# We save the clustering results for later use:

# %%
pd.DataFrame(
    {'cluster': clusters_info_df.loc[best_method]['cluster_labels']}
).to_csv(RESULTS_DIR + "hierarchical_clusters.csv")
pd.DataFrame({
        'BBS': np.NaN,
        'SSE': np.NaN,
        'calinski_harabasz_score': np.NaN,
        'davies_bouldin_score': np.NaN,
        'model': best_method,
        'n_iter': np.NaN,
        'silhouette_score': clusters_info_df.loc['ward']['silhouette_score']
    },
    index=[0]
).to_csv(RESULTS_DIR + "hierarchical_internal_scores.csv")

# %% [markdown]
# ## Final considerations
# Advantages of hierarchical clustering:
# - Do not have to assume any particular number of clusters
# - Suitable for data with a nested or hierarchical structure
# 
# Disadvantages of hierarchical clustering:
# - No global objective function is directly minimized (once a decision is made to merge two clusters, it cannot be undone at a later time)
# - Is expensive in terms of computational and storage requirements
# 
# Furthermore - as outlined above - each different proximity measures has its own advantages and disadvantages (e.g. sensitivity to noise and outliers or difficulty in handling clusters of different sizes and non-globular shapes).


