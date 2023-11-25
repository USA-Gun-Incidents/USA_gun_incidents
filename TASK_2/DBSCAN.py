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
from plot_utils import plot_scattermap_plotly

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
illinois_df = incidents_df[incidents_df['state']=='ILLINOIS'].dropna()
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
    'location_imp',
    'surprisal_address_type',
    # age
    'avg_age',
    # participants
    'severity',
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
# in this sedtion we prepare some usefull function in order to perform the clustering analisys
#
# Here a brief documentation:
#
# function *standardization*: standardize data in in the dataframe given in input contenuti nelle colonne columns, usando metodo 'Z-score' or 'MinMax' standardization, both using object import from sklearn
#
# function *plot_dbscan*: this function visualize the results of the clustering analisys via DBSCAN, 
# plotta i punti appartenenti ai cluster in colori diversi,
# questa funzione fornisce una rappresentazione visuale del risultato del clustering effettuato da DBSCAN sui dati forniti.

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardization(df, columns, standardizer='Zscore'):
    if standardizer == 'Zscore':
        standardizer = StandardScaler()
    if standardizer == 'MinMax':
        standardizer = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(df[columns].values)
    return scaler.transform(df[columns].values)

# %%
def plot_dbscan(X, db, columns): 
    labels = db.labels_ 
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True # create an array of booleans where True = core point
    # core point = point that has at least min_samples in its eps-neighborhood

    plt.figure(figsize=(20, 8))

    colors = [plt.cm.rainbow_r(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k # array of booleans where True = point in cluster k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=10,
            label=f'Cluster {k}'
        )

        # plot noise points
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor=col,
            markersize=6,
            label=f'Cluster {k}'
        )

    plt.grid()
    plt.legend()
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

# %%
from sklearn.cluster import DBSCAN
from sklearn import metrics 

def dbscan(X, eps=0.1, min_samples=10, plot_clusters=False):
    # Compute DBSCAN      
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    if plot_clusters:
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        if len(np.unique(labels)) > 1:
            print("Silhouette Coefficient: %0.3f"
                % metrics.silhouette_score(X, labels))
        plot_dbscan(X, db)
    
    return {'eps': eps, 'min_samples': min_samples, 
        '#clusters': len(set(labels)) - (1 if -1 in labels else 0),
        '#noise': list(labels).count(-1),  '%noise': list(labels).count(-1)/X.shape[0]*100,
        'silhouette_coef': metrics.silhouette_score(X, labels), 
        '#cluster0': list(labels).count(0), '#cluster1': list(labels).count(1), 
        '#cluster2': list(labels).count(2), '#cluster3': list(labels).count(3), 
        '#cluster4': list(labels).count(4), '#cluster5': list(labels).count(5),
        '#cluster6': list(labels).count(6), '#cluster7': list(labels).count(7)}

# %% [markdown]
# The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

# %% [markdown]
# ### Find best EPS

# %%
from scipy.spatial.distance import pdist, squareform

def find_best_eps(X, k_list=[3, 5, 9, 15]):
    dist = pdist(X, 'euclidean') # pair wise distance
    dist = squareform(dist) # distance matrix given the vector dist
    
    # Calculate sorted list of distances for points for each k in k_list
    # and plot the graph of distance from k-th nearest neighbour
    fig, ax = plt.subplots(int(np.ceil(len(k_list)/3)), 3, figsize=(20, 8))

    for i, k in enumerate(k_list):
        kth_distances = list()
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])

        # Plot the graph of distance from k-th nearest neighbour
        ax[int(i/3), int(i%3)].plot(range(0, len(kth_distances)), sorted(kth_distances))
        ax[int(i/3), int(i%3)].set_ylabel('%sth near neighbor distance' %k)
        ax[int(i/3), int(i%3)].set_xlabel('Point Sorted according to distance of %sth near neighbor' %k)
        #ax[int(i/3), int(i%3)].set_yticks(np.linspace(0, 5, 12))
        #ax[int(i/3), int(i%3)].set_ylim(0, 3)
        ax[int(i/3), int(i%3)].tick_params(axis='both', which='major', labelsize=8)
        ax[int(i/3), int(i%3)].grid(linestyle='--', linewidth=0.5, alpha=0.6)

    plt.show()

# %% [markdown]
# ## Clustering: Illinois

# %%
#kneed_algorithm(X_std_illinois, neighbors=5)
# DBSCAN(eps=1.75, min_samples=5).fit(X_std_illinois) #dati stadardizzati, eps=1.75, min_samples=5, 5 clusters

# %% [markdown]
# ### MinMax Scale Data

# %%
X_minmax_illinois = (illinois_df[ind_names_list], columns=ind_names_list)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
plt.boxplot(X_std_illinois, vert=True, labels=ind_names_list)
plt.xticks(rotation=90, ha='right')
plt.show()

# %%
find_best_eps(X_minmax_illinois, k_list=[3, 5, 9, 15, 20, 30])

# %%
eps = [0.1, 0.15, 0.2, 0.25]
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
min_samples = [3, 5, 10, 15, 20]

dbscan_illinois = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in eps:
    for k in min_samples:
        db = dbscan(X_minmax_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois = pd.concat([dbscan_illinois, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois

# %% [markdown]
# ### Visualize data

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_dbscan_subplots(X, db):
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    n_dimensions = X.shape[1]

    fig, axs = plt.subplots(n_dimensions, n_dimensions, figsize=(15, 15))

    colors = [plt.cm.rainbow_r(each) for each in np.linspace(0, 1, len(unique_labels))]

    for i in range(n_dimensions):
        for j in range(n_dimensions):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])

            if i == j:
                ax.text(0.5, 0.5, f'Dimension {i + 1}', ha='center', va='center', fontsize=10, color='black')
            else:
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        col = [0, 0, 0, 1]

                    class_member_mask = labels == k
                    xy = X[class_member_mask & core_samples_mask]
                    ax.plot(
                        xy[:, i],
                        xy[:, j],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor='k',
                        markersize=10,
                        label=f'Cluster {k}'
                    )

                    xy = X[class_member_mask & ~core_samples_mask]
                    ax.plot(
                        xy[:, i],
                        xy[:, j],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor=col,
                        markersize=6,
                        label=f'Cluster {k}'
                    )

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()

# Esempio di utilizzo
# plot_dbscan_subplots(X, db)



# %%
db = DBSCAN(eps=0.2, min_samples=10).fit(X_minmax_illinois) #12
#plot_dbscan(X_std_illinois, db)

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

# %%
illinois_df['cluster'] = db.labels_
sns.pairplot(illinois_df, hue='cluster', palette=sns.color_palette(
    n_colors=illinois_df['cluster'].unique().shape[0]), vars=ind_names_list)
plt.show()

# %%
df = incidents_df.loc[illinois_df.index][ind_names_list]
df['cluster'] = db.labels_

fig, ax = plt.subplots(3, 2, figsize=(20, 10), sharex=False, sharey=False)
index = 0
for i in range(6):
    for cluster in np.unique(db.labels_):
        ax[int(index/2), index%2].hist(df[df['cluster']==cluster][ind_names_list[i]], 
            bins=int(1+3.3*np.log(df[df['cluster']==cluster].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(index/2), index%2].set_xlabel(df.columns[i], fontsize=8)
    ax[int(index/2), index%2].set_yscale('log')
    ax[int(index/2), index%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(index/2), index%2].legend(fontsize=8)
    ax[int(index/2), index%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)
    index += 1

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
fig, ax = plt.subplots(6, 2, figsize=(20, 10), sharex=True, sharey=True)
for i in range(12):
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
fig, ax = plt.subplots(6, 2, figsize=(20, 10), sharex=True, sharey=True)
for i in range(12):
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
    fig, ax = plt.subplots(4, 3, figsize=(20, 8), sharex=True, sharey=True)
    for i in range(12):
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
