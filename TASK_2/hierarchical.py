# %% [markdown]
# # Hierarchical Clustering
# 
# Hierarchical clustering is an approach that produces a set of nested clusters organized as a hierarchical tree. It works very well with nested or hierarchical structures of the data, moreover it's flexible, becouse you don't have to choose a priori a fixed number of clusters. On the other hand the time and space performance are not the best, and sometimes the algorithms are too much sensitive to noise and otliers.<br>
# (It's also used to determine the number of clusters before starting k-means.) <-- *vediamo se mettere questa frase, non l'abbiamo fatto*
# <br><br>
# Agglomerative clustering works using a proximity matrix, which shows the distance between each cluster. At each iteration the clusters with smallest distance are merged until only one cluster remains. The distance between each group is calculated using different policies, we will explain later the ones we choose for our analisys.

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
import utm

# %% [markdown]
# We start importing the dataset of the indicators we want to use to cluster incidents

# %%
incidents_df = pd.read_csv('../data/indicators.csv', index_col=False)
incidents_df.drop("Unnamed: 0", axis=1, inplace=True)

# %%
incidents_df.info()

# %% [markdown]
# ## Dataset preparation
# We get the original dataset to get complete info on each incident

# %%
incidents_df_full = pd.read_csv('../data/incidents_cleaned.csv')
incidents_df_full.head(2)

# %% [markdown]
# We restrict once again our analisys to the state with most records

# %%
# select a subset of records regarding a certain state
incidents_df['state'] = incidents_df_full['state']

state = "ILLINOIS"
incidents_df = incidents_df[incidents_df['state'] == state]
incidents_df.drop('state', axis=1, inplace=True)

# %% [markdown]
# TODO: quando sei sicuro, eliminare le successive 2 celle di codice

# %%
latlong_projs = utm.from_latlon(incidents_df['latitude'].to_numpy(), incidents_df['longitude'].to_numpy())
scaler= MinMaxScaler()
latlong = scaler.fit_transform(np.stack([latlong_projs[0], latlong_projs[1]]).reshape(-1, 2))
incidents_df['latitude_proj'] = latlong[:,0]
incidents_df['longitude_proj'] = latlong[:,1]

# %%
incidents_df.drop(columns=['latitude', 'longitude'], axis=1, inplace=True)
new_order = ['latitude_proj', 'longitude_proj', 'location_importance', 'avg_age_participants',
       'n_participants', 'age_range', 'n_participants_child_prop',
       'n_participants_teen_prop', 'n_killed_pr', 'n_injured_pr',
       'n_arrested_pr', 'n_unharmed_pr',
       'log_n_males_n_males_mean_semest_congd_ratio', 'log_avg_age_mean_SD',
       'avg_age_entropy', 'city_entropy', 'address_entropy',
       'n_adults_entropy', 'tags_entropy', 'severity']
incidents_df = incidents_df[new_order]

# %% [markdown]
# We delete latitude and longitude columns becouse, from previous experiments, we saw that they affect too much the clustering. In fact, in different algorithms, they result to be a criteria "too much weighted", leading in cluster subdivision based almost just on their distribution

# %%
incidents_df.drop(columns=['lat_proj', 'lon_proj'], axis=1, inplace=True)

# %%
incidents_df.isna().sum()

# %% [markdown]
# We drop all columns with NaN, otherwise we can't perform clustering. Since they're not too much, this action doesn't affect the analisys.

# %%
incidents_df.dropna(inplace=True)

incidents_df.shape

# %% [markdown]
# We scale all the data in order to avoid that the attributes with higher values affect too much the clustering. What can happen, in fact, is that if an attribute has bigger numbers in the range if its possible values, the overall distance between each points will be moslty affected by the difference calculated for that particular attribute. By doing scaling we restrict our values between 0 and 1, avoiding the domincance of one or a small group of features.

# %%
scaler_obj = MinMaxScaler()
incidents_df = pd.DataFrame(data=scaler_obj.fit_transform(incidents_df.values), columns=incidents_df.columns)

# %%
# print all indexes for clustering
incidents_df.columns

# %% [markdown]
# ## Clustering

# %% [markdown]
# There are different types of algorithms we can choose to perform hierrchical clustering, and they differ for the type of linkage they perform. We choose 4 algorithms: *single linkage*, *complete linkage*, *average linkage* and *ward*. Here we give a short description of them, pointing out theire strengths and weaknesses.
# - **single linkage**: the proximity of two clusters corrisponds to the distance to the nearest point in each cluster. This performs good when we want to capture cluster with non-elliptical shapes, but at the same time we lose robustness to noise and outliers.
# - **complete linkage**: the proximity of two clusters corrisponds to the distance to the furthest point in each cluster. The advantage here is that the algorithm is not susceptible to noise and outliers, but this strategy tends to be biased towards globular clusters. Moreover it doesn't work well when we have one or more clusters bigger than the others, since it may happen that it breaks them in different parts. 
# - **average linkage**: it combines the last two algorithms, in fact here the the proximity of two clusters corrisponds to the the average of the distance of all points in both clusters. It still remains robust to noise and outliers, but it's also biased towards globular clusters.
# - **ward**: it's the hierarchical analogue of k-means, the similarity of two clusters is based on the increase in squared error when two clusters are merged. Like average linkage is robust to noise and outliers but it's biased towards globular clusters.
# 

# %%
# clustering
algorithms = ["single", "complete", "average", "ward"]
linkages = []
distance_thresholds = []

for algorithm in algorithms:
    #models.append(AgglomerativeClustering(linkage=algorithm, compute_distances=True).fit(incidents_df))
    linkage_res = linkage(pdist(incidents_df, metric='euclidean'), method=algorithm, metric='euclidean', optimal_ordering=False)

    linkages.append(linkage_res)
    distance_thresholds.append(0.7 * max(linkage_res[:,2])) # the first threshold we use is the default

# %% [markdown]
# We define a function that plots dendrograms corresponding to a certain linkage. Each cluster is displayed in different colors (the choice of cluster separation is given by a certain threshold)

# %%
import sys
sys.setrecursionlimit(10000)

def plot_dendrograms(linkages, algorithms, thresholds):
    f, axs = plt.subplots(ncols=len(linkages), figsize=(32,7))

    for i in range(len(linkages)):
        axs[i].set_title(algorithms[i])
        axs[i].set_xlabel('IncidentID or (Cluster Size)')
        axs[i].set_ylabel('Distance')
        axs[i].axhline(thresholds[i], ls='--', color='k', label='threshold')
        color_threshold = thresholds[i]

        # Plot the corresponding dendrogram
        dendrogram(linkages[i], truncate_mode="lastp", p=30, leaf_rotation=60, leaf_font_size=8,
                show_contracted=True, ax=axs[i], color_threshold=color_threshold)

    plt.suptitle(('Hierarchical Clustering Dendograms'), fontsize=18, fontweight='bold')
    plt.show()

# %% [markdown]
# We check the results of clustering looking at the dendrograms. In the first analisys we set the threshold to its default value.

# %%
plot_dendrograms(linkages, algorithms, distance_thresholds)

# %% [markdown]
# From these dendrograms we see that each linkage lead to very different shaped hierarchies.<br>
# The **single-linkage** tends to add each point to a huge cluster that contains (almost) all the other points already chosen for merging. The result we get is similar to what we could archieve aligning some points with increasing distance to the previous in a straight line.<br>
# The **complete-linkage**, instead, we can observe a more balanced tree. We can see from the number between parenthesis (below in the graph) that there is one cluster significantly higher than others, but the result we have it's not the one we get from single-linkage where everything is absorbed by the same one.<br>
# In **average-linkage** we find, as expected, a compromise between single and complete linkage. In fact, we see that there is a cluster that tend to absorb all the other points, but not in the same way we see in single-linkage. Compared to the first one, as we can see, the tree is more balanced.<br>
# In **ward** we find the most balanced results, both in terms of tree structure and in terms of clusters size.

# %% [markdown]
# Here we try to find the **best cut**, we iterate until a certain point and we find the optimal number of clusters for each linkage. We compare the different cuts we have looking at the **average silhouette score**, assigning to the best cut the one which maximize this measure.
# <br><br>
# The **silhouette score** is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). It's in range [-1, 1] and we want to maximize it, but with a clustering that actually make sense and give us some extra info on our data.

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

# %% [markdown]
# We plot the results of our search.

# %%
clusters_info_df

# %% [markdown]
# The **Cophenetic correlation coefficient**  is a measure of how faithfully a dendrogram preserves the pairwise distances between the original unmodeled data points. It's calculated looking at the distancences beetween each points in the clusters and in the original dataset and their mean.

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
# Here we plot the same dendrograms but applying the threshold we found.

# %%
plot_dendrograms(linkages, algorithms, list(results_df['distance threshold']))

# %% [markdown]
# # Clustering evaluation

# %% [markdown]
# We evaluate the results we get using different plots that enligth different aspects of our analisys.

# %% [markdown]
# At first we plot the **proximity** and **ideal similarity matrices** to find out if the points in the same clusters are close to each other, in this way we inspect correlation between the points.<br>
# We plot only the matrices corresponding to complete linkage and ward, since the other two have only two cluster where the second one is very small (singletone in case of single-link).<br>
# We also compute the **Pearson's coefficient** to check how good is the correlation. It's calculated as the ratio between the covariance of two variables and the product of their standard deviations.

# %%
# correlation matrix
RANDOM_STATE = 42
X = incidents_df.values

# display only complete and ward
plot_distance_matrices(X=X, n_samples=500, clusters=clusters_info_df.loc[1]['cluster_labels'], random_state=RANDOM_STATE)
plot_distance_matrices(X=X, n_samples=500, clusters=clusters_info_df.loc[3]['cluster_labels'], random_state=RANDOM_STATE)

# %% [markdown]
# Here we plot, in a histogram, the size of each cluster for all the linkage methods we used.

# %%
for i in range(len(algorithms)):
    fig, axs = plt.subplots(1)
    plot_clusters_size(clusters_info_df.loc[i]['cluster_labels'], ax=axs, title='Clusters size')
    fig.show()

# %% [markdown]
# We plot the distance between the merged clusters. Here we can also visualize at wich heigth the standard threshold and the best cut threshold we found separate the clusters.

# %%
start_iteration = 1
fig, axs = plt.subplots(ncols=4, figsize=(28, 8))
for i, algorithm in enumerate(algorithms):
    axs[i].plot(range(start_iteration, linkages[i].shape[0]), linkages[i][start_iteration:, 2], 'o')
    axs[i].axhline(distance_thresholds[i], ls='--', color='k', label='default threshold')
    axs[i].axhline(results_df.loc[algorithms[i]]['distance threshold'], ls='--', color='r', label='best cut threshold')
    axs[i].legend()
    axs[i].set_title(f'{algorithm} linkage')
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel('Merge Distance')
fig.suptitle('Distance between merged clusters', fontweight='bold')

# %% [markdown]
# Here we display a **scatter plot** of all the features used for clustering combined two by two. We use these plots to understand better what are the criteria of separation and wich spaces of the data are included in each cluster.

# %%
cluster_column_name = algorithms
incidents_df_cluster = incidents_df.copy()

# we create a dataframe where we can easily retrieve the label of the clustering for each incident
for i in range(len(algorithms)):
    incidents_df_cluster[algorithms[i]] = clusters_info_df.loc[i]['cluster_labels']

scatter_by_cluster(incidents_df_cluster,
                   incidents_df.columns,
                   algorithms[3],
                   figsize=(20, 180),
                   color_palette=sns.color_palette(n_colors=clusters_info_df.loc[3]['n_clusters']))

# %% [markdown]
# Here we tell, for each cluster, which are the distinguishing attributes and their associated values:<br>
# 
# cluster 0: **surprisal n_injured** > 0.5 & **n_child_prop** > 0 (in the last one not all the points belong to cluster 0 but still the majority)
# 
# cluster 1: **n_temp_prop** > 0.5 (it's a small cluster, and for this attriute just half the points belong to the cluster) & (**avg age** = 0.2 & & **surprisal_age_groups** around the half, in particular when **n_injured_prop** = 1 or **n_arrested_prop** = 0)
# 
# cluster 2: **n_injured_prop** = 1 & **surprisal_age_groups** < 0.2 & **surprisal_n_males** < 0.2 & **surprisal_characteristics** < 0.2 (for the last 3 attributes, more than half of the points that lie in that space but non all)
# 
# cluster 3: **hunarmed_prop** > 0.5
# 
# cluster 4: **n_arrested_prop** > 0.2 with **surprisal_n_injured** < 0.5
# 
# cluster 5: **avg_age** = 0.2 & **n_teen_prop** < 0.5 but even in this case, since this is a small cluster, not all the points in those spaces belongs to cluster 5
# 
# cluster 6: **n_killed_prop** = 1 in particular with **surprisal_age_groups** < 0.4
# 
# cluster 7: it's the smallest cluster and there are no subspaces of the data where we found a clear distinction from other points, but the records in this cluster lie in 0 < **surprisal_age_groups** < 0.2 & 0.4 < **surprisal_n_males** < 0.6 & **n_males_prop** = 0

# %% [markdown]
# We display, for each feature, a **boxplot** and a **violin plot** in order to check the distribution of the data and get some statistical insights for that feature in a specific cluster.<br>
# We restrict to complete-link and ward, becouse for single and average link we have a big cluster which contains almost all records, so the distribuition of the features will be the same we have in the entire dataset.

# %%
for algorithm in ["complete", "ward"]:
    plot_boxes_by_cluster(incidents_df_cluster,
                        incidents_df.columns,
                        algorithm,
                        figsize=(15, 65),
                        title=('Box plots of features by cluster - ' + algorithm))

# %% [markdown]
# behaviur for each feature:
# - location_imp:
# - surprisal_address_type:
# - age_range:
# - avg_age:
# - surprisal_min_age:
# - n_child_prop:
# - n_teen_prop:
# - surprisal_age_groups:
# - n_killed_prop:
# - n_injured_prop:
# - surprisal_n_injured:
# - n_unharmed_prop:
# - n_males_prop:
# - surprisal_n_males:
# - surprisal_characteristics:
# - n_arrested_prop:
# - n_participants:
# - surprisal_day:
# 

# %%
for algorithm in ["complete", "ward"]:
    plot_violin_by_cluster(
        incidents_df_cluster,
        incidents_df.columns,
        algorithm,
        figsize=(15, 35),
        title=('Violin plots of features by cluster- ' + algorithm)
    )

# %% [markdown]
# Finally we check again but with an histogram, just for the algorithm "ward" (we take it as exemplification just to avoid huge and messy outputs), the distribution of the values in each cluster. This time we compare it with the distribution on all the dataset.

# %%
for feature in incidents_df.columns:
    plot_hists_by_cluster(
        df=incidents_df_cluster,
        feature=feature,
        cluster_column='ward',
        title=f'Distribution of {feature} in each cluster',
        color_palette=sns.color_palette(n_colors=int(results_df.loc['ward']['optimal n_clusters']) + 1)
    )

# %% [markdown]
# What we see it that when data is homogeneously distributed in the dataset, it will be the same on each cluster. Differences among groups arise when there are distributions with more than one local maximum (ex: n_arrested_prop); in these cases it may happen that clusters cover different and separated ranges of that data.


