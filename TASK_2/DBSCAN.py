# -*- coding: utf-8 -*-
# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
#
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa

# %% [markdown]
# # Density clustering

# %% [markdown]
# Import library and dataset

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics 
from scipy.spatial.distance import pdist, squareform
from plot_utils import plot_scattermap_plotly
from clustering_utils import plot_dbscan

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
min_samples = [6, 12, 20]

dbscan_illinois = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in eps:
    for k in min_samples:
        db = dbscan(X_minmax_illinois, eps=e, min_samples=k)
        dbscan_illinois = pd.concat([dbscan_illinois, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois

# %% [markdown]
# ### Perform Clustering

# %% [markdown]
# We selected *eps* = 1.5 and *min-samples* = 20

# %%
db = DBSCAN(eps=1.5, min_samples=20).fit(X_minmax_illinois) # 5 first trial

# %%
illinois_df['cluster'] = db.labels_

# %%
illinois_df.shape

# %% [markdown]
# ### Results 

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
# From this plot we can observe that clusters semms divided data for 'n_killed'

# %% [markdown]
# We plot the data distribution between clusters for 'avg_age' and 'surprisal_address_type' use Plotly for the same visualization to obtain a clearer view and better understand how noise is distributed

# %%
ind_names_list[:2][0]

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
# Belowed we plot 

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
plt.title('Histograms of features per cluster')
plt.show()

# %%
columns = ['n_males', 'n_adult', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 'year', 'poverty_perc', 'congd']
df = incidents_df.loc[illinois_df.index][columns]

fig, ax = plt.subplots(9, 4, figsize=(20, 30))
index = 0
for i in range(9):
    for j in range(i+1, 9):
        ax[int(index/4), index%4].scatter(df.values[:, i], df.values[:, j], c=db.labels_, cmap='plasma', s=20)
        ax[int(index/4), index%4].set_xlabel(df.columns[i], fontsize=8)
        ax[int(index/4), index%4].set_ylabel(df.columns[j], fontsize=8)
        ax[int(index/4), index%4].tick_params(axis='both', which='major', labelsize=6)
        ax[int(index/4), index%4].grid(linestyle='--', linewidth=0.5, alpha=0.6)
        index += 1
#plt.suptitle('DBSCAN Clustering', fontsize=16)
plt.show()

# %%
columns = ['n_males', 'n_adult', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 'poverty_perc', 'congd'] #'year'
df = incidents_df.loc[illinois_df.index][columns]
df['cluster'] = db.labels_

fig, ax = plt.subplots(4, 2, figsize=(20, 15), sharex=False, sharey=False)
for i in range(8):
    for cluster in np.unique(db.labels_):
        ax[int(i/2), i%2].hist(df[df['cluster']==cluster][columns[i]], 
            bins=int(1+3.3*np.log(df[df['cluster']==cluster].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(i/2), i%2].set_xlabel(df.columns[i], fontsize=8)
    ax[int(i/2), i%2].set_yscale('log')
    ax[int(i/2), i%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(i/2), i%2].legend(fontsize=8)
    ax[int(i/2), i%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)

# %%
# plot hist for poverty_perc for each cluster
fig, ax = plt.subplots(3, 2, figsize=(20, 10), sharex=True, sharey=True)
for i in range(6):
    ax[int(i/2), i%2].hist(illinois_df['poverty_perc'].values[db.labels_==i-1], 
        bins=int(1+3.3*np.log(illinois_df[illinois_df['cluster']==i-1].shape[0])), 
        label=f'Cluster {i-1}', edgecolor='black', linewidth=0.8, alpha=0.7,
        color=sns.color_palette(n_colors=illinois_df['cluster'].unique().shape[0])[i])
    ax[int(i/2), i%2].set_xlabel('poverty_perc', fontsize=8)
    ax[int(i/2), i%2].set_yscale('log')
    ax[int(i/2), i%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(i/2), i%2].legend(fontsize=8)
    ax[int(i/2), i%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)


# %%
# plot hist for congd for each cluster
fig, ax = plt.subplots(3, 2, figsize=(20, 10), sharex=True, sharey=True)
for i in range(6):
    ax[int(i/2), i%2].hist(illinois_df['congd'].values[db.labels_==i-1], 
        bins=int(1+3.3*np.log(illinois_df[illinois_df['cluster']==i-1].shape[0])), 
        label=f'Cluster {i-1}', edgecolor='black', linewidth=0.8, alpha=0.7,
        color=sns.color_palette(n_colors=illinois_df['cluster'].unique().shape[0])[i])
    ax[int(i/2), i%2].set_xlabel('congd', fontsize=8)
    ax[int(i/2), i%2].set_yscale('log')
    ax[int(i/2), i%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(i/2), i%2].legend(fontsize=8)
    ax[int(i/2), i%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)

# %% [markdown]
# ## External Data

# %%
mortality_data = pd.read_csv('../data/external_data/deaths.csv', index_col=False)
mortality_data['year'] = mortality_data['year'].astype('float')
mortality_data = mortality_data[mortality_data['state']=='Illinois']
mortality_data

# %%
illinois_df = illinois_df.merge(mortality_data, on=['year'], how='left')
illinois_df.head(2)

# %%
for attribute in ['male_child', 'male_teen', 'male_adult', 'female_child', 'female_teen', 'female_adult']:
    fig, ax = plt.subplots(2, 3, figsize=(20, 8), sharex=True, sharey=True)
    for i in range(6):
        ax[int(i/3), i%3].hist(illinois_df[attribute].values[db.labels_==i-1], 
            bins=int(1+3.3*np.log(illinois_df[illinois_df['cluster']==i-1].shape[0])), 
            label=f'Cluster {i-1}', edgecolor='black', linewidth=0.8, alpha=0.7,
            color=sns.color_palette(n_colors=illinois_df['cluster'].unique().shape[0])[i])
        ax[int(i/3), i%3].set_xlabel(attribute, fontsize=8)
        ax[int(i/3), i%3].set_yscale('log')
        ax[int(i/3), i%3].tick_params(axis='both', which='major', labelsize=6)
        ax[int(i/3), i%3].legend(fontsize=8)
        ax[int(i/3), i%3].grid(linestyle='--', linewidth=0.5, alpha=0.6)

# %% [markdown]
# ### Show cluster on a map

# %%
illinois_df[['latitude', 'longitude', 'county', 'city']] = incidents_df.loc[illinois_df.index, [
    'latitude', 'longitude', 'county', 'city']]

illinois_df.head(2)

# %%
plot_scattermap_plotly(illinois_df, 'cluster', zoom=5, title='Incidents clustered by DBSCAN')

# %%
plot_scattermap_plotly(illinois_df[illinois_df['county']=='Cook County'], 'cluster', 
    zoom=8, title='Incidents clustered by DBSCAN')

# %% [markdown]
# ## Proximity matrix

# %%
cluster_df = pd.DataFrame()
cluster_df['DBS_labels'] = db.labels_
cluster_df['data_point'] = range(len(cluster_df))
cluster_df.index = pd.RangeIndex(len(cluster_df)) # define an index from 0 to (len(data))
cluster_df = cluster_df.sort_values(by='DBS_labels')

# %%
from scipy.spatial import distance_matrix

dist_matrix = distance_matrix(cluster_df, cluster_df)
ax = sns.heatmap(dist_matrix)

# %% [markdown]
# ### compare with K-means

# %%
from sklearn.cluster import KMeans
from clustering_utils import compute_bss_per_cluster
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score


# %%
def fit_kmeans(X, params):
    print(f"Fitting KMeans with k={params['n_clusters']}")
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
    return results


# %%
MAX_ITER = 300
N_INIT = 10
INIT_METHOD = 'k-means++'
MAX_K = 30
RANDOM_STATE = 42

# %%
results = {}
kmeans_params = {}
kmeans_params['random_state'] = RANDOM_STATE
kmeans_params['max_iter'] = MAX_ITER

kmeans_params['n_init'] = N_INIT
kmeans_params['n_clusters'] = 6
kmeans_params['init'] = INIT_METHOD
result = fit_kmeans(X=X_minmax_illinois, params=kmeans_params)
results[str(k)+'means'] = result

# %%
results_df = pd.DataFrame(results).T
results_df.drop(columns=['model'])

# %%
kmeans = KMeans(**kmeans_params)
kmeans.fit(X_minmax_illinois)

# %%
kmeans.labels_

# %%
from clustering_utils import sankey_plot

# %%
sankey_plot(
    [kmeans.labels_, db.labels_],
    labels_titles=['Kmeans', 'DBSCAN'],
    title='Clusterings comparison'
)

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
