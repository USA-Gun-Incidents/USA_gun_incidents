# %%
import pandas as pd
import seaborn as sns
from clustering_utils import *

# %%
labels_files = ['./4-Means_clusters.csv', './4-Means_clusters-PCA.csv', './dbscan_clusters.csv', './heirarchical_clusters.csv']

labels = []
for files in labels_files:
    labels.append(pd.read_csv(files, index_col=0)['cluster'].to_numpy())

# dbscan_clusters = pd.read_csv('./dbscan_clusters.csv', index_col=0)['cluster'].to_numpy()
# heirarchical_clusters = pd.read_csv('./heirarchical_clusters.csv', index_col=0)['cluster'].to_numpy()

# %%
sankey_plot(
    [kmeans_clusters, kmeans_PCA_clusters],
    labels_titles=['Kmeans', 'Kmeans-PCA'],
    title='Clusterings comparison'
)

# %%
# TODO: 
# - other metrics and plots to compare results in a single plot
# - align labels according to plot above
# - make other plots, e.g. mark points belonging to different cluster


