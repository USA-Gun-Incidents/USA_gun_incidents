# %% [markdown]
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
clusters_df = pd.DataFrame(index=[i for i in range(239379)])
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
# BSS and SSE are not comparable because the feature space on which we run the algorithms is different. The other scores are comparable. 

# %%
silhouette_df = internal_scores_df['silhouette_score'].to_frame()
silhouette_df

# %%
DBSCAN_silhouette = pd.read_csv(PATH+'DBSCAN_internal_scores.csv', index_col=0)['silhouette_score'].values[0]
DBSCAN_silhouette

# %%
pd.concat([silhouette_df, pd.DataFrame({'silhouette_score': [DBSCAN_silhouette]}, index=['DBSCAN'])])

# %%
external_scores_df = pd.DataFrame()
for name, external_score_file in zip(clustering_name[:2], external_scores_files[:2]):
    scores_curr_df = pd.read_csv(external_score_file, index_col='feature')
    for
    # columns = {}
    # for column in scores_curr_df.columns:
    #     columns[column] = column+' '+name
    # scores_curr_df = scores_curr_df.rename(columns=columns)
    # external_scores_df = pd.concat([external_scores_df, scores_curr_df], axis=1)

# %%
external_scores_files = [PATH+'4-Means_external_scores.csv', PATH+'4-Means_PCA_external_scores.csv', PATH+'DBSCAN_external_scores.csv']
external_scores_dfs = []
for name, external_score_file in zip(clustering_name, external_scores_files):
    scores_curr_df = pd.read_csv(external_score_file, index_col='feature')
    external_scores_dfs.append(scores_curr_df)
