# %%
import pandas as pd
import seaborn as sns
from clustering_utils import *

# %%
kmeans_clusters = pd.read_csv('./kmeans_clusters.csv', index_col=0)['cluster'].to_numpy() # files written with write_clusters_to_csv in clustering utils
dbscan_clusters = pd.read_csv('./dbscan_clusters.csv', index_col=0)['cluster'].to_numpy()
heirarchical_clusters = pd.read_csv('./heirarchical_clusters.csv', index_col=0)['cluster'].to_numpy()

# %%
sankey_plot(
    [kmeans_clusters, dbscan_clusters, heirarchical_clusters],
    labels_titles=['Kmeans', 'DBSCAN', 'Heirarchical'],
    title='Clusterings comparison'
)

# %%
# TODO: 
# - other metrics and plots to compare results in a single plot
# - align labels according to plot above
# - make other plots, e.g. mark points belonging to different cluster


