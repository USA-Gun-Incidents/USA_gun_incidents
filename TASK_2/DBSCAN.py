# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
# 
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa

# %% [markdown]
# # Density clustering

# %% [markdown]
# Import library and dataset

# %%
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics 
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from plot_utils import plot_scattermap_plotly
from clustering_utils import plot_dbscan, plot_scores_per_point, plot_bars_by_cluster, compute_bss_per_cluster
from clustering_utils import plot_hists_by_cluster_dbscan
from clustering_utils import plot_distance_matrices, write_clusters_to_csv, compute_permutation_invariant_external_metrics
sys.path.append(os.path.abspath('..'))
from plot_utils import sankey_plot

# %%
incidents_df = pd.read_csv(
    '../data/incidents_indicators.csv',
    index_col=0,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

# %%
incidents_df.head(2)

# %% [markdown]
# Prepare dataset and a list of indices and numerical attributes for choosen state: Illinois.

# %%
illinois_df = incidents_df[incidents_df['state']=='ILLINOIS']
illinois_df.head(2)

# %%
ind_names_list = [
    # geographical
    'lat_proj', 'lon_proj', 'location_imp', 'surprisal_address_type',
    # age 
    'age_range', 'avg_age', 'surprisal_min_age',
    'n_child_prop', 'n_teen_prop', 'surprisal_age_groups', 
    # number of participants by group
    'severity', 'n_killed_prop', 'surprisal_n_killed', 'n_injured_prop',
    'surprisal_n_injured', 'n_unharmed_prop',
    # gender
    'n_males_prop', 'surprisal_n_males', 'n_arrested_prop',
    # characteristics
    'surprisal_characteristics', 
    # number of participantes
    'surprisal_n_participants', 'n_participants', 
    # date
    'surprisal_day']

# %% [markdown]
# ## DBSCAN

# %% [markdown]
# In our clustering analysis, we chose to use DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to perform density-based clustering. 
# DBSCAN defines a cluster as a dense region of objects, offering a robust solution for identifying clusters with diverse shapes and sizes within complex datasets. This algorithm provides a flexible and effective approach to cluster analysis. 
# Unlike traditional partitional clustering algorithms that necessitate pre-specification of the number of clusters, DBSCAN autonomously detects clusters during the analysis, alleviating the need for prior knowledge about the dataset.
# 
# DBSCAN excels in handling datasets with irregular clusters, arbitrary shapes, and varying sizes. It is particularly robust in the presence of noise or outliers, classifying points in low-density regions as noise. This adaptability allows DBSCAN to discover clusters that may be challenging for other clustering methods, such as K-means, to identify.
# 
# Pros:
# - Accommodates irregular clusters with arbitrary shapes and sizes.
# - Effective in the presence of noise or outliers.
# - Identifies clusters that may go unnoticed by traditional clustering methods like K-means.
# 
# Cons:
# - Faces challenges when clusters exhibit widely varying densities.
# - Encounters difficulties with high-dimensional data due to the nuanced definition of density.

# %% [markdown]
# ## Indices and numerical feauters correlation

# %% [markdown]
# Since DBSCAN encounters difficulties with high-dimensional data, we decided to address this challenge by plotting the correlation matrix of all possible attributes. The goal is to choose a subset of attributes that are not highly correlated with each other for the clustering analysis.
# 
# In our previous study, we thoroughly examined the features' correlation and distribution in the notebook containing the data understanding analysis state by state. Therefore, we are taking those observations into consideration to choose the subset of attributes for our clustering analysis.

# %%
corr_matrix_illinois = illinois_df[ind_names_list].dropna().corr('kendall')

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_illinois, annot=True, cmap=plt.cm.Reds, mask=np.triu(corr_matrix_illinois))
plt.show()

# %%
ind_names_list = [
    # geo
    'surprisal_address_type',
    # age
    'avg_age',
    # participants
    'n_killed',
    'n_males_prop',
    'n_arrested_prop',
    'n_participants',
    ]

# %% [markdown]
# Below, we have reported the correlation matrix for the selected features along with a brief description of each.

# %%
corr_matrix_illinois = illinois_df[ind_names_list].corr('kendall')

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_illinois, annot=True, cmap=plt.cm.Reds, mask=np.triu(corr_matrix_illinois))
plt.show()

# %%
illinois_df[ind_names_list].describe()

# %% [markdown]
# We observe that the selected features exhibit low correlation among themselves and display distinct value ranges.

# %% [markdown]
# ## Utilities

# %% [markdown]
# In this section, we have prepared several useful functions for conducting clustering analysis. Here is a brief documentation for each function:
# 
# 1. **Function: standardization**
#    - Description: Standardizes data in the input dataframe (*df*) within the specified columns using either the 'Z-score' or 'MinMax' standardization methods. Both methods are implemented using objects imported from the sklearn library.
#    - Input:
#      - *df*: Input dataframe containing columns to be standardized.
#      - *columns*: List of column names to be standardized.
#      - *method*: The method of standardization to be used, either 'Z-score' or 'MinMax'.
#    - Output:
#      - A *np.ndarray* with the specified columns standardized.
# 
# 2. **Function: find_best_eps**
#    - Description: Implements the knee method to select the best candidates as values of *eps* for all the *min_samples* tried. Euclidean distance is used for this selection.
#    - Input:
#      - *X*: Input data array.
#      - *min_samples_range*: Range of values for *min_samples* parameter.
#    - Output:
#      - Plot the graph of distance from k-th nearest neighbour
# 
# 3. **Function: dbscan**
#    - Description: Performs the DBSCAN algorithm using the **DBSCAN** library from sklearn.cluster. The function returns a dictionary containing *eps* and *min_samples* given as input, the estimated number of noise points, the estimated number of clusters, and the silhouette coefficient. This function is used to select the best parameters (*eps* and *min_samples*).
#    - Input:
#      - *X*: Input data array.
#      - *eps*: Maximum distance between two samples for them to be considered in the same neighborhood.
#      - *min_samples*: Minimum number of points needed to create a cluster.
#    - Output:
#      - A dictionary containing information about the clustering results.
# 
# We have chosen to use the silhouette coefficient as a measure to evaluate the algorithm's performance. The silhouette coefficient ranges from -1 to +1, where a higher value indicates that objects are well-matched to their own cluster and poorly matched to neighboring clusters.

# %% [markdown]
# ### Standardize Data

# %%
def standardization(df, columns, standardizer='Zscore'):
    if standardizer == 'Zscore':
        standardizer = StandardScaler()
    if standardizer == 'MinMax':
        standardizer = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(df[columns].values)
    return scaler.transform(df[columns].values)

# %% [markdown]
# ### Find best EPS

# %%
def find_best_eps(X, min_samples_range=[3, 5, 9, 15]):
    dist = pdist(X, 'euclidean') # pair wise distance
    dist = squareform(dist) # distance matrix given the vector dist
    
    # Calculate sorted list of distances for points for each k in k_list
    # and plot the graph of distance from k-th nearest neighbour
    fig, ax = plt.subplots(int(np.ceil(len(min_samples_range)/3)), 3, figsize=(20, 8))

    for i, k in enumerate(min_samples_range):
        kth_distances = list()
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])

        # Plot the graph of distance from k-th nearest neighbour
        ax[int(i/3), int(i%3)].plot(range(0, len(kth_distances)), sorted(kth_distances))
        ax[int(i/3), int(i%3)].set_ylabel('%sth near neighbor distance' %k)
        ax[int(i/3), int(i%3)].set_xlabel('Point Sorted according to distance of %sth near neighbor' %k)
        ax[int(i/3), int(i%3)].tick_params(axis='both', which='major', labelsize=8)
        ax[int(i/3), int(i%3)].grid(linestyle='--', linewidth=0.5, alpha=0.6)

    plt.show()

# %% [markdown]
# ### DBASCAN Algorithm

# %%
def dbscan(X, eps=0.1, min_samples=10):
    # Compute DBSCAN      
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    return {'eps': eps, 'min_samples': min_samples, 
        '#clusters': len(set(labels)) - (1 if -1 in labels else 0),
        '#noise': list(labels).count(-1),  '%noise': list(labels).count(-1)/X.shape[0]*100,
        'silhouette_coef': metrics.silhouette_score(X, labels), 
        '#cluster0': list(labels).count(0), '#cluster1': list(labels).count(1), 
        '#cluster2': list(labels).count(2), '#cluster3': list(labels).count(3), 
        '#cluster4': list(labels).count(4), '#cluster5': list(labels).count(5),
        '#cluster6': list(labels).count(6), '#cluster7': list(labels).count(7)}

# %% [markdown]
# ## DBSCAN Algorithm: Illinois

# %% [markdown]
# ### Prepare Data and Parameters selection

# %% [markdown]
# We prepare the data by selecting only the entries in the dataset corresponding to the Illinois state and where all the values corresponding to the selected attributes, on which the clustering algorithm operates, are not NaN. Since the data have different range values for the various attributes, we decided to standardize using the 'MinMax' scaler, which, from some preliminary trials, has proven to be the most effective.
# 
# We also save all the attribute values for the entries we use in the dataframe *illinois_df*.

# %%
X_minmax_illinois = standardization(
    df=incidents_df[incidents_df['state']=='ILLINOIS'][ind_names_list].dropna(), 
    columns=ind_names_list, standardizer='MinMax')

# %%
illinois_df = incidents_df.loc[incidents_df[incidents_df['state']=='ILLINOIS'][ind_names_list].dropna().index]

# %%
illinois_df.shape

# %% [markdown]
# We plot boxplots representing the various attributes of our standardized data, revealing the presence of numerous outliers.

# %%
fig, ax = plt.subplots(figsize=(15, 5))
plt.boxplot(X_minmax_illinois, vert=True, labels=ind_names_list)
plt.xticks(rotation=90, ha='right')
plt.show()

# %% [markdown]
# We utilize the Kneed algorithm to identify the range for the optimal epsilon *eps*.

# %%
find_best_eps(X_minmax_illinois, min_samples_range=[3, 5, 9, 15, 20, 30])

# %% [markdown]
# Upon examining the plot, we decide to explore epsilon values in the range between 1.25 and 2.

# %%
eps = [1.25, 1.5, 1.75, 2]
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
min_samples = [7, 12, 20]

dbscan_illinois = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in eps:
    for k in min_samples:
        db = dbscan(X_minmax_illinois, eps=e, min_samples=k)
        dbscan_illinois = pd.concat([dbscan_illinois, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois

# %% [markdown]
# The selection of *eps* and *min_samples* was based on the silhouette coefficient, the number of clusters, their size, and the noise detected. 
# 
# Preliminary trials were conducted to choose appropriate coefficients. 
# We decided not to use *min_samples* greater than 20 since the algorithm struggled to find a suitable number of clusters, either grouping everything into a single cluster or forming multiple clusters, one of which contained almost all the data. In these cases, the silhouette coefficient appeared higher, but the result was unsatisfactory and lacked significance. 
# 
# Therefore, we opted for *min_samples=20* and *eps=1.5*.

# %% [markdown]
# ### Perform Clustering

# %% [markdown]
# We selected *eps* = 1.5 and *min-samples* = 20

# %%
db = DBSCAN(eps=1.5, min_samples=20).fit(X_minmax_illinois) # 5 first trial

# %%
illinois_df['cluster'] = db.labels_ # save cluster labels in dataframe

# %%
print('Silhouette Coefficient: %0.6f' %
    silhouette_score(X_minmax_illinois, db.labels_))

# %% [markdown]
# ### Save cluster

# %% [markdown]
# We save the clusters labels in a csv file, in order to use them for a cluster comparison analysis.

# %%
illinois_df['cluster'].to_csv('../data/clustering_labels/DBSCAN_clusters.csv', index=True)

# %% [markdown]
# ## Metrics Visualization

# %% [markdown]
# We have plotted the distance matrix. The distance matrix shows a clear diagonal correlation, as indicated by the Pearson coefficient of 0.43.

# %%
plot_distance_matrices(X=X_minmax_illinois, n_samples=500, clusters=db.labels_);

# %% [markdown]
# We have plotted the silhouette score for each point in each cluster. It is noticeable that clusters 0 and 1 (the most populated clusters) have a minority of points for which the silhouette score assumes negative values, indicating potential misclassifications or less distinct clusters. On the other hand, higher silhouette values are achieved by clusters 5 and 6 (the least populated ones), suggesting a clearer separation of points within these clusters.

# %%
def plot_scores_per_point(score_per_point, clusters, score_name, ax, color_palette=sns.color_palette(), title=None, minx=-0.1):
    '''
    This function plots the clustering score for each point, grouped by cluster.

    :param score_per_point: clustering score for each point
    :param clusters: cluster labels
    :param score_name: name of the clustering score
    :param ax: axis to plot on
    :param color_palette: color palette to use
    :param title: title of the plot
    '''

    n_clusters = len(np.unique(clusters))
    y_lower = 0
    for i in range(n_clusters):
        ith_cluster_sse = score_per_point[np.where(clusters == i)[0]]
        ith_cluster_sse.sort()
        size_cluster_i = ith_cluster_sse.shape[0]
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sse,
            facecolor=color_palette[i],
            edgecolor=color_palette[i],
            alpha=0.7,
        )
        ax.text(minx+0.1*i, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper

    ax.axvline(x=score_per_point.mean(), color="k", linestyle="--", label='Average')
    if title is None:
        title = f"{score_name} for each point in each cluster"
    ax.set_title(title)
    ax.set_xlabel(score_name)
    ax.set_ylabel("Cluster label")
    ax.legend(loc='best')
    ax.set_yticks([])

# %%
fig, ax = plt.subplots(figsize=(15, 3))
plot_scores_per_point(score_per_point=silhouette_samples(X=X_minmax_illinois, labels=db.labels_), 
    clusters=db.labels_, score_name='Silhouette Coefficient', ax=ax)

# %% [markdown]
# We saved the silhouette score in a dataframe for later comparison with other clustering algorithms. It's important to note that, in the case of DBSCAN, the silhouette score is not an ideal metric for evaluating clustering since DBSCAN is not a partitioning algorithm. A more meaningful interpretation can be derived using the silhouette score per point, as demonstrated earlier.
# 
# In the dataframe, additional columns include other metrics to make it compatible with the metrics dataframe we will use for other clustering algorithms.

# %%
results_df = pd.DataFrame(columns=['BSS', 'SSE', 'calinski_harabasz_score',
    'calinski_harabasz_score', 'davies_bouldin_score', 'model', 'n_iter', 'silhouette_score'], index=range(1))


results_df['model'] = 'DBSCAN(eps=1.5, min_samples=20)'
results_df['silhouette_score'] = silhouette_score(X_minmax_illinois, db.labels_)

# %%
results_df

# %% [markdown]
# We computed the same external metrics as well. It's noteworthy that the metrics for the *death* category, which represents incidents involving at least one fatality, are quite promising. These external metrics have been calculated for a subsequent comparative analysis between clustering algorithms.

# %%
illinois_df['unharmed'] = illinois_df['n_unharmed'] > 0
illinois_df['arrested'] = illinois_df['n_arrested'] > 0
illinois_df['males'] = illinois_df['n_males'] > 0
illinois_df['females'] = illinois_df['n_females'] > 0

external_scores_df = compute_permutation_invariant_external_metrics(
    illinois_df,
    'cluster',
    ['shots', 'aggression', 'suicide', 'injuries', 'death', 'drugs', 'illegal_holding', 'unharmed', 'arrested','males', 'females']
)

external_scores_df

# %%
results_df.to_csv(f'../data/clustering_labels/DBSCAN_internal_scores.csv')
external_scores_df.to_csv(f'../data/clustering_labels/DBSCAN_external_scores.csv')

# %% [markdown]
# ## Results 

# %% [markdown]
# Below, we visualize how the algorithm divided the data into clusters.
# 
# From the barplot below, we can clearly see that the clusters are not balanced; the majority of entries (9941 over 13234) are contained in one cluster.

# %%
# bar plot of number of incidents per cluster
cluster_counts = pd.Series(db.labels_).value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(cluster_counts.index, cluster_counts.values, edgecolor='black', linewidth=0.8, alpha=0.5)
plt.xlabel('Cluster')
plt.xticks(cluster_counts.index)
plt.ylabel('Number of incidents')
plt.yscale('log')
for i, v in enumerate(cluster_counts.values):
    plt.text(x=i-1, y=v, s=str(v), horizontalalignment='center', verticalalignment='bottom', fontsize=8)
plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
plt.title('Number of incidents per cluster')
plt.show()

# %% [markdown]
# We visualize clusters by plotting the divisions between all dimensions using Seaborn's pairplot.

# %%
sns.pairplot(illinois_df, hue='cluster', palette=sns.color_palette(
    n_colors=illinois_df['cluster'].unique().shape[0]), vars=ind_names_list)
plt.show()

# %% [markdown]
# From this plot, we can observe that clusters seem to divide the data based on 'n_killed.'

# %% [markdown]
# We plot the data distribution between clusters for 'avg_age' and 'surprisal_address_type' use Plotly for the same visualization to obtain a clearer view and better understand how noise is distributed

# %%
plot_dbscan(X=X_minmax_illinois, db=db, columns=[0,1], axis_labels=ind_names_list[:2], figsize=(10, 5))
# 'surprisal_address_type': column 0 in X_minmax_illinois
# 'avg_age': column 1 in X_minmax_illinois

# %% [markdown]
# From the plot w can observe that class -1 in black represents the noise points, points with a black border represent the core points of the cluster, while points without a border are the border points.
# 
# From this plot, a precise division of the clusters does not appear. Cluster 5 seems to occupy points where 'surprisal_address_type' mostly takes negative values. A similar trend can be observed for cluster 3.
# 
# Clusters 0 and 1 (the most populated ones) seem to have a uniform distribution across the data, and a similar observation applies to the noise points.

# %% [markdown]
# Below, we plot some histograms to better visualize how data with the same values in each feature are divided into clusters.

# %%
fig, ax = plt.subplots(3, 2, figsize=(20, 10), sharex=False, sharey=False)
index = 0
for i in range(6):
    for cluster in np.unique(db.labels_):
        ax[int(index/2), index%2].hist(illinois_df[illinois_df['cluster']==cluster][ind_names_list[i]], 
            bins=int(1+3.3*np.log(illinois_df[illinois_df['cluster']==cluster].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(index/2), index%2].set_xlabel(ind_names_list[i], fontsize=8)
    ax[int(index/2), index%2].set_yscale('log')
    ax[int(index/2), index%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(index/2), index%2].legend(fontsize=8)
    ax[int(index/2), index%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)
    index += 1
fig.suptitle('Histograms of features per cluster')
fig.tight_layout()
plt.show()

# %% [markdown]
# Already, we can see the division by cluster for 'n_killed', the other features seem to be uniformly distributed across the cluster classes.

# %% [markdown]
# Below, we represent the same data as before, but in a histogram for each cluster and each feature, making it more comprehensible.

# %%
for column in ind_names_list:
    plot_hists_by_cluster_dbscan(df=illinois_df, db=db, column=column, figsize=(15, 5))

# %% [markdown]
# From this representation, we can better visualize the data distribution across clusters. For 'surprisal_address_type' and 'avg_age', the division of data into clusters appears homogeneous, and so does the noise. We can conclude that the cluster does not exhibit significant variations in these features.
# 
# A similar observation can be made for the number of participants, where we also notice that all data with a number of participants greater than 6 have been classified as noise, as they are indeed outliers in the distribution of data for the number of participants, both in the data used for clustering and in the entire dataset.
# 
# Regarding 'n_killed', as seen before, the clusters have divided the data based on this value. 
# 
# For 'n_arrested_prop', in clusters 2 and 3, there are only data with values close to 0 (left margin), in cluster 4, only data with values close to 1 (right margin), and in clusters 0 and 1 (the most populated ones), the values seem to be uniformly distributed.
# 
# Similarly, for 'n_males_prop', clusters 0 and 1 exhibit a distribution similar to 'n_arrested_prop', clusters 2 and 3 show marginal values near 0, and cluster 4 has marginal values near 1.
# 
# In conclusion, the algorithm has clustered the data, demonstrating clear divisions based on certain features such as 'n_killed,' 'n_arrested_prop,' and 'n_males_prop.' 
# However, for features like 'surprisal_address_type' and 'avg_age,' the clusters appear to exhibit homogeneity.
# The algorithm also effectively identifies outliers points, particularly for instances with a high number of participants.

# %% [markdown]
# ### Visualize Clusters on a Map

# %% [markdown]
# We plot the clusters on an Illinois map to visualize their geographical distribution.

# %%
plot_scattermap_plotly(illinois_df, 'cluster', zoom=5, title='Incidents clustered by DBSCAN')

# %% [markdown]
# Since the majority of incidents in our dataset in the state of Illinois are situated in Cook County, we also provide a zoomed-in view of that county for visualization purposes."

# %%
plot_scattermap_plotly(illinois_df[illinois_df['county']=='Cook County'], 'cluster', 
    zoom=8, title='Incidents clustered by DBSCAN')

# %% [markdown]
# No discernible patterns are observed; data points belonging to each cluster appear to be uniformly distributed across the territory.

# %% [markdown]
# We use a choropleth map to visualize the the most frequent clusters for incidents in each county in Illinois.

# %%
cluster_df = pd.DataFrame()
cluster_df['county'] = illinois_df['county']
cluster_df['county'] = cluster_df['county'].str.replace(' County', '')
cluster_df['county'] = cluster_df['county'].str.replace('Saint Clair', 'St. Clair')
cluster_df['county'] = cluster_df['county'].str.replace('DeWitt', 'De Witt')
cluster_df['cluster'] = illinois_df['cluster']

illinois_map = gpd.read_file('../cb_2018_us_county_500k')
illinois_merged = illinois_map.merge(cluster_df, left_on='NAME', right_on='county')
illinois_merged = illinois_merged[illinois_merged['STATEFP']=='17']

# %%
vmin, vmax = illinois_merged['cluster'].agg(['min', 'max'])
illinois_merged.plot(column='cluster', cmap='plasma', figsize=(10, 6), vmin=vmin, vmax=vmax,
    legend=True, legend_kwds={'label': 'cluster', 'shrink': 1}, edgecolor='black', linewidth=0.5)
plt.title('Most frequent cluster per county in Illinois')
plt.xticks([])
plt.yticks([])
plt.show()

# %% [markdown]
# We can observe that clusters 0 and 1 are the most frequent plot, as we expected, since they are the most populated and 
# the clusters distribuition sems to be uniform. 
# We can also observe that in 3 county the most frequent cluster is 2, while in 1 county the most frequent cluster is 5.

# %% [markdown]
# ## Analysis of Clustering Results Using External Features

# %% [markdown]
# In this section, we have examined the results of the clustering analysis, taking into account additional external features present in the dataset. Our goal is to explore how these clusters align with the identified external characteristics, providing a more comprehensive understanding of the overall dataset structure.

# %% [markdown]
# We considered the following features and indices:

# %%
features = [
    # number of participants
    'n_males', 'n_adult', 'n_injured', 'n_arrested', 'n_unharmed', 
    # age
    'n_child_prop', 'n_teen_prop', 'age_range', 
    # geographical
    'location_imp', 'congd',
    # characteristics
    'surprisal_characteristics', 
    # date
    'surprisal_day', 'year', 
    # external data
    'poverty_perc'
]

# %% [markdown]
# Below, we have plotted histograms for each of the selected features to visualize how incidents are distributed within the clusters based on feature values.

# %%
fig, ax = plt.subplots(7, 2, figsize=(15, 15), sharex=False, sharey=False)
for i in range(len(features)):
    for cluster in np.unique(db.labels_):
        ax[int(i/2), i%2].hist(illinois_df[illinois_df['cluster']==cluster][features[i]], 
            bins=int(1+3.3*np.log(illinois_df[illinois_df['cluster']==cluster].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(i/2), i%2].set_xlabel(features[i], fontsize=8)
    ax[int(i/2), i%2].set_yscale('log')
    ax[int(i/2), i%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(i/2), i%2].legend(fontsize=6)
    ax[int(i/2), i%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)
fig.tight_layout()
plt.show()

# %% [markdown]
# Despite the histograms not revealing a distinct pattern in the distribution of incidents within the clusters, we are exploring alternative visualization methods to uncover potential relationships or patterns that may not be evident in the current representations.

# %%
plot_bars_by_cluster(illinois_df[illinois_df['cluster']!=-1], feature='n_males', 
    cluster_column='cluster', figsize=(10, 5), log_scale=True)

# %%
plot_bars_by_cluster(illinois_df[illinois_df['cluster']!=-1], feature='n_child', 
    cluster_column='cluster', figsize=(10, 5), log_scale=True)

# %%
plot_bars_by_cluster(illinois_df[illinois_df['cluster']!=-1], feature='n_teen', 
    cluster_column='cluster', figsize=(10, 5), log_scale=True)

# %%
plot_bars_by_cluster(illinois_df[illinois_df['cluster']!=-1], feature='n_adult', 
    cluster_column='cluster', figsize=(10, 5), log_scale=True)

# %%
plot_bars_by_cluster(illinois_df[illinois_df['cluster']!=-1], feature='n_participants', 
    cluster_column='cluster', figsize=(10, 5), log_scale=True)

# %% [markdown]
# In the last 3 clusters (the least populated ones), there are no men or children, and they all belong to cluster 3. In cluster 3, there are few teenagers, and from the histograms, it can be observed that the majority of incidents clustered in clusters 3, 4, and 5 involve only one adult female participant.

# %%
plot_bars_by_cluster(illinois_df[illinois_df['cluster']!=-1], feature='year', 
    cluster_column='cluster', figsize=(10, 5), log_scale=True)

# %% [markdown]
# We are unable to find significant patterns among the clusters concerning the temporal division into years.

# %% [markdown]
# ## Noise Visualization

# %% [markdown]
# We present statistical metrics for all incidents in Illinois and specifically for those classified as noise by the DBSCAN algorithm. This analysis aims to provide insights into how the clustering algorithm identified noise points.

# %%
illinois_df.describe()[['min_age', 'max_age', 'avg_age', 'n_child', 'n_teen', 'n_adult', 'n_males',
       'n_females', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 'n_participants']]

# %%
illinois_df[illinois_df['cluster']==-1].describe()[['min_age', 'max_age', 'avg_age', 'n_child', 'n_teen', 'n_adult', 'n_males',
       'n_females', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 'n_participants']]

# %%
print('Number of incidents classified as noise:', illinois_df[illinois_df['cluster']==-1].shape[0])
print('Number of noise incidents with at least one woman: ', 
    illinois_df[(illinois_df['cluster']==-1) & (illinois_df['females']==True)].shape[0])
print('Number of noise incidents where average age is over 27: ', 
    illinois_df[(illinois_df['cluster']==-1) & (illinois_df['avg_age']>27)].shape[0])
print('Number of noise incidents where there are more than two participants: ', 
    illinois_df[(illinois_df['cluster']==-1) & (illinois_df['n_participants']>2)].shape[0])

# %% [markdown]
# We can observe that in the majority of incidents classified as noise, there are women present, and the average age is higher than 27, which happens to be the average age of participants in incidents. Additionally, these incidents involve more than two participants, corresponding to the fourth quartile of the number of participants in incidents. 
# 
# These results suggest that such incidents could have been considered outliers in the dataset, indicating that the DBSCAN algorithm correctly identified these cases as noise.

# %% [markdown]
# ## Compare DBSCAN with K-means

# %% [markdown]
# We also compare the clusters identified by the DBSCAN algorithm with those obtained from k-means clustering, setting k equal to the number of clusters identified by DBSCAN. This approach is suggested in the *'Cluster Evaluation'* chapter of *Pang-Ning Tan, Michael Steinbach, Vipin Kumar. Introduction to Data Mining.*

# %% [markdown]
# To obtain more information on the k-means algorithm, please refer to the notebook where the algorithm was thoroughly examined.

# %%
MAX_ITER = 300
N_INIT = 10
INIT_METHOD = 'k-means++'
MAX_K = 30
RANDOM_STATE = 42

def fit_kmeans(X, params):
    #print(f'Fitting KMeans with k={params['n_clusters']}')
    kmeans = KMeans(**params)
    kmeans.fit(X)
    results = {}
    results['model'] = kmeans
    results['SSE'] = kmeans.inertia_
    results['BSS'] = compute_bss_per_cluster(X=X, clusters=kmeans.labels_, centroids=kmeans.cluster_centers_, weighted=True).sum()
    results['davies_bouldin_score'] = davies_bouldin_score(X=X, labels=kmeans.labels_)
    results['calinski_harabasz_score'] = calinski_harabasz_score(X=X, labels=kmeans.labels_)
    results['silhouette_score'] = silhouette_score(X=X, labels=kmeans.labels_) 
    results['n_iter'] = kmeans.n_iter_
    return kmeans, results

# %%
results = {}
kmeans_params = {}
kmeans_params['random_state'] = RANDOM_STATE
kmeans_params['max_iter'] = MAX_ITER

kmeans_params['n_init'] = N_INIT
kmeans_params['n_clusters'] = 6
kmeans_params['init'] = INIT_METHOD

kmeans, results = fit_kmeans(X=X_minmax_illinois, params=kmeans_params)
results[str(k)+'means'] = result

# %%
results

# %% [markdown]
# To visualize the correspondence between clusters found by DBSCAN and k-means, we plotted the Sankey plot. It's important to note that in the Sankey plot for DBSCAN, Class 0 corresponds to the noise that we have referred to as -1. Additionally, all cluster labels are scaled by 1 for consistency.

# %%
sankey_plot(
    [kmeans.labels_, db.labels_],
    labels_titles=['Kmeans', 'DBSCAN'],
    title='Clusterings comparison'
)

# %% [markdown]
# From the plot, it is evident that clusters between 0 and 3 found by k-means are mapped into the most frequent cluster of DBSCAN. This suggests that we can map DBSCAN clusters to a k-means model with k=3 for a more concise representation.

# %%
kmeans_params['n_clusters'] = 3
kmeans = KMeans(**kmeans_params)
kmeans.fit(X_minmax_illinois)

# %%
sankey_plot(
    [kmeans.labels_, db.labels_],
    labels_titles=['Kmeans', 'DBSCAN'],
    title='Clusterings comparison'
)

# %% [markdown]
# The results of this latest attempt have shown significant improvement. We can observe that K-means Cluster 1 has been entirely mapped to the clusters identified as 2 and 4 in the DBSCAN plot, K-means Cluster 2 to cluster 2, and K-means Cluster 0 to DBSCAN Cluster 1 and clusters 0, 3, and 6, representing noise and the less numerous classes in DBSCAN. We can, therefore, see clear correspondences in the cluster creation between the two algorithms.


