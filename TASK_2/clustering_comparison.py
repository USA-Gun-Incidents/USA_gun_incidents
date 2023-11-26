# %%
import pandas as pd
import seaborn as sns
from clustering_utils import *

# %%
PATH = '../data/clustering_labels/'
clustering_name = ['KMeans', 'KMeansPCA', 'DBSCAN', 'Heirarchical']
labels_files = [PATH+'/4-Means_clusters.csv', PATH+'./4-Means_clusters-PCA.csv', PATH+'./DBSCAN_illinois.csv', PATH+'./heirarchical_clusters.csv']
external_scores_files = [PATH+'./4-Means_external_scores.csv', PATH+'./4-Means_external_scores.csv']#PATH+'./4-Means_external_scores-PCA.csv', PATH+'./DBSCAN_illinois_external_scores.csv', PATH+'./heirarchical_external_scores.csv']
internal_scores_files = [PATH+'./4-Means_internal_scores.csv', PATH+'./4-Means_internal_scores.csv']#PATH+'./4-Means_internal_scores-PCA.csv', PATH+'./DBSCAN_illinois_internal_scores.csv', PATH+'./heirarchical_internal_scores.csv']
indexes_files = [PATH+'./4-Means_indexes.csv', PATH+'./4-Means_indexes.csv']#PATH+'./4-Means_indexes-PCA.csv', PATH+'./DBSCAN_illinois_indexes.csv', PATH+'./heirarchical_indexes.csv'

labels = []
common_indexes = set()
internal_scores_df = pd.DataFrame()
indexes_df = pd.DataFrame()
for name, labels_file, external_score_file, internal_score_file, indexes_file in zip(clustering_name, labels_files, external_scores_files, internal_scores_files, indexes_files):
    labels.append(pd.read_csv(labels_file, index_col=0)['cluster'].to_numpy())
    internal_scores_curr_df = pd.read_csv(internal_score_file, index_col=0).T['silhouette_score']
    internal_scores_df = pd.concat([internal_scores_df, internal_scores_curr_df])
    indexes_curr_df = pd.read_csv(indexes_file, index_col=0)
    
    # indexes_df = pd.read_csv(indexes_file, index_col=0)
    # indexes_list = indexes_df['0'].to_list()
    # common_indexes = common_indexes.intersection(set(indexes_list)) if len(common_indexes) > 0 else set(indexes_list)

# %%
sankey_plot(
    [kmeans_clusters, kmeans_PCA_clusters],
    labels_titles=['Kmeans', 'Kmeans-PCA'],
    title='Clusterings comparison'
)
