# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
#  
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
#
# # Clustering comparison
#
# In this notebook, we compare the clustering results of the different methods. Since DBSCAN and Heirarchical clustering were applied only to the incidents happened in Illinois, we restrict the comparison to the incidents in this state. We remind that the state of Illinois was chosen because it had few null values and the distribution of the variables was similar to the distribution of the variables of the whole dataset.
#
# We import the libraries:

# %%
import pandas as pd
from clustering_utils import *

# %% [markdown]
# We define the paths to the saved clustering results:

# %%
PATH = '../data/clustering_labels/'
clustering_name = ['KMeans', 'KMeansPCA', 'DBSCAN', 'Hierarchical']
labels_files = [PATH+'/4-Means_clusters.csv', PATH+'4-Means_PCA_clusters.csv', PATH+'DBSCAN_clusters.csv', PATH+'hierarchical_clusters.csv']
external_scores_files = [PATH+'4-Means_external_scores.csv', PATH+'4-Means_PCA_external_scores.csv', PATH+'DBSCAN_external_scores.csv', PATH+'hierarchical_external_scores.csv']
internal_scores_files = [PATH+'4-Means_internal_scores.csv', PATH+'4-Means_PCA_internal_scores.csv', PATH+'DBSCAN_internal_scores.csv', PATH+'hierarchical_internal_scores.csv']

# %% [markdown]
# We concatenate the clustering results into a single dataframe:

# %%
nrows = pd.read_csv(labels_files[0], index_col=0).shape[0]
clusters_df = pd.DataFrame(index=[i for i in range(nrows)])
for name, labels_file, external_score_file, internal_score_file in zip(clustering_name, labels_files, external_scores_files, internal_scores_files):
    clusters_curr_df = pd.read_csv(labels_file, index_col=0)
    clusters_curr_df = clusters_curr_df.rename(columns={'cluster':'cluster'+name})
    clusters_df = clusters_df.join(clusters_curr_df)
clusters_df.dropna(inplace=True)

# %% [markdown]
# We visualize the clustering results using a sankey diagram:

# %%
sankey_plot(
    [clusters_df['clusterDBSCAN'], clusters_df['clusterKMeans'], clusters_df['clusterKMeansPCA'], clusters_df['clusterHierarchical']],
    labels_titles=['DBSCAN', 'KMeans', 'KMeansPCA', 'Heirarchical'],
    title='Clusterings comparison'
)

# %% [markdown]
# The clusters found by KMeans applied to the indicators and the clusters found by KMeansPCA applied to the first principal components of the indicators are very similar. Cluster 2 of DBSCAN groups almost all the points from cluster 0, 1 and 3 of KMeans; while cluster 1 of DBSCAN groups almost all the points from cluster 2 of KMeans. Cluster 2 of KMeansPCA groups points belonging mainly to cluster 1, 2 and 6 of the Heirachical clustering. There is an high overlap between cluster 0 and 3 of KMeansPCA and Heirarchical clustering. Cluster 1 of KMeansPCA graoups all the points in cluster 5 of Heirarchical clustering. 
#
# From this analysis, we can conclude that despite the differences in the methods, the clusters found are not so different.
#
# Now we compare the internal scores of KMeans and KMenasPCA:

# %%
internal_scores_df = pd.DataFrame()
for name, internal_scores_file in zip(clustering_name[:2], internal_scores_files[:2]):
    internal_scores_curr_df = pd.read_csv(internal_scores_file, index_col=0).T
    internal_scores_df = pd.concat([internal_scores_df, internal_scores_curr_df])
internal_scores_df.rename(columns={'0':'silhouette_score'}, inplace=True)
internal_scores_df.drop(columns=['model'], inplace=True)
internal_scores_df

# %% [markdown]
# BSS and SSE are not comparable because the feature space on which we run the algorithms is different. The other scores are comparable. As for the Calinski-Harabasz score and the Silhouette score, the best results are obtained by KMeans, while for the Davies-Bouldin score the best results are obtained by KMeansPCA.
#
# Now we compare the silhouette score of all the methods:

# %%
silhouette_df = internal_scores_df['silhouette_score'].to_frame()
DBSCAN_silhouette = pd.read_csv(PATH+'DBSCAN_internal_scores.csv', index_col=0)['silhouette_score'].values[0]
hierarchical_silhouette = pd.read_csv(PATH+'hierarchical_internal_scores.csv', index_col=0).T['silhouette_score'].values[0]
pd.concat([silhouette_df, pd.DataFrame({'silhouette_score': [DBSCAN_silhouette, hierarchical_silhouette]}, index=['DBSCAN', 'Hierarchical'])])

# %% [markdown]
# According to the silhouette score the best clustering results are obtained by Hierarchical clustering.

# %% [markdown]
# We finally visualize the external scores of all the methods:

# %%
scores_per_feature = {}
scores_per_metric = {}
algs_order = []
scores_order = []
features_order = []
external_scores_df = pd.DataFrame()
for name, external_score_file in zip(clustering_name, external_scores_files):
    scores_curr_df = pd.read_csv(external_score_file, index_col='feature')
    for feature in scores_curr_df.index:
        if feature not in scores_per_feature:
            scores_per_feature[feature] = []
        scores_per_feature[feature].append(scores_curr_df.loc[feature].to_list())
    for metric in scores_curr_df.columns:
        if metric not in scores_per_metric:
            scores_per_metric[metric] = []
        scores_per_metric[metric].append(scores_curr_df[metric].to_list())
    algs_order.append(name)
    scores_order = scores_curr_df.columns.to_list()
    features_order = scores_curr_df.index.to_list()

# %%
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i, key in enumerate(scores_per_metric):
    pd.DataFrame(np.array(scores_per_metric[key]), index=algs_order, columns=features_order).plot.bar(rot=0, title=key, ax=axs[i])

# %%
fig, axs = plt.subplots(2, 2, figsize=(30, 10))
for i, key in enumerate(scores_per_metric):
    pd.DataFrame(np.array(scores_per_metric[key]).T, index=features_order, columns=algs_order).plot.bar(rot=0, title=key, ax=axs[int(i/2)][i%2])

# %%
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i, key in enumerate(scores_per_feature):
    pd.DataFrame(np.array(scores_per_feature[key]), index=algs_order, columns=scores_order).plot.bar(rot=0, title=key, ax=axs[int(i/3)][i%3])

# %% [markdown]
# Regarding the external scores:
# - the class 'death' is better clustered by DBSCAN
# - KMeans has similar scores to KMeansPCA; KMeansPCA works better in identifying incidents from the class 'arrest'
# - Heirarchical clustering has the highest scores for the class 'arrested' and works also better than the other algorithms in identifying incidents from the classes 'aggression' and 'injuries'


