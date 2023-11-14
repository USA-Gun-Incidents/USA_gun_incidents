# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
np.warnings = warnings # altrimenti numpy da problemi con pyclustering, TODO: è un problema solo mio?
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans, BisectingKMeans
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
import utm
import os
import sys
sys.path.append(os.path.abspath('..'))
from plot_utils import *
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv'
)
indicators_df = pd.read_csv(
    '../data/incidents_cleaned_indicators.csv'
)
incidents_df[indicators_df.columns] = indicators_df

# %%
features_to_cluster = indicators_df.columns
categorical_features = [
    'year', 'month', 'day_of_week', #'state', 'address_type', 
    'firearm', 'air_gun', 'shots', 'aggression', 'suicide',
    'injuries', 'death', 'road', 'illegal_holding', 'house',
    'school', 'children', 'drugs', 'officers', 'organized', 'social_reasons',
    'defensive', 'workplace', 'abduction', 'unintentional', 'party']

# %%
incidents_df = incidents_df.dropna(subset=features_to_cluster)
#incidents_df.replace([np.inf, -np.inf], 0, inplace=True)

# %%
plt.figure(figsize=(15, 12))
corr_matrix = incidents_df[features_to_cluster].corr() # TODO: different coor coefficients
sns.heatmap(corr_matrix, mask=np.triu(corr_matrix))

# %%
incidents_df[features_to_cluster].boxplot()
plt.xticks(rotation=90);

# %%
scaler= StandardScaler()
X = scaler.fit_transform(incidents_df[features_to_cluster].values) # TODO: come scegliere?

# %%
def plot_k_elbow(X, kmeans_params, metric, start_k, max_k): # TODO: plottare nello stesso plot curva che parte da 1 e da 2 se sono diverse
    if metric == 'distortion':
        metric_descr = 'SSE'
    elif metric == 'calinski_harabasz':
        metric_descr = 'Calinski Harabasz Score'
    elif metric == 'silhouette':
        metric_descr = 'Silhouette Score'
    else:
        raise ValueError('Metric not supported')
    
    _, axs = plt.subplots(nrows=1, ncols=len(max_k), figsize=(30,5))

    best_k = []

    for i in range(len(max_k)):
        kmeans_params['n_clusters'] = i
        kmeans = KMeans(**kmeans_params)
        elbow_vis = KElbowVisualizer(kmeans, k=(start_k, max_k[i]), metric=metric, timings=False, ax=axs[i])
        elbow_vis.fit(X)
        axs[i].set_title(f'{metric_descr} elbow for K-Means clustering (K = [{str(start_k)}, {str(max_k[i])}])')
        axs[i].set_ylabel(metric_descr)
        axs[i].set_xlabel('K')
        axs[i].legend([
            f'{metric_descr} for K',
            f'elbow at K = {str(elbow_vis.elbow_value_)}, {metric_descr} = {elbow_vis.elbow_score_:0.2f}'
        ])
        if elbow_vis.elbow_value_ != None and elbow_vis.elbow_value_ not in best_k:
            best_k.append(elbow_vis.elbow_value_)
    
    plt.show()
    return best_k

# %%
kmeans_params = {'n_init': 10, 'max_iter': 100}
max_k = [10, 20, 30] # + di 30 è difficile da interpretare
best_k = []

# %%
ks = plot_k_elbow(X, kmeans_params, 'distortion', 1, max_k) # start_k=1 or start_k=2 ?
best_k += ks

# %%
# k = plot_k_elbow(X, kmeans_params, 'silhouette', 1, max_k)
# best_k += ks

# %%
# k = plot_k_elbow(X, kmeans, 'calinski_harabasz', 1, max_k)
# best_k += ks

# %%
initial_centers = kmeans_plusplus_initializer(X, 1).initialize()
xmeans_obj = xmeans(X, initial_centers, kmax=max_k[-1])
xmeans_obj.process()
n_xmeans_BIC_clusters = len(xmeans_obj.get_clusters())
print('Number of clusters found by xmeans using BIC score: ', n_xmeans_BIC_clusters)
if n_xmeans_BIC_clusters < max_k[-1] and n_xmeans_BIC_clusters not in best_k:
    best_k.append(n_xmeans_BIC_clusters)

# %%
xmeans_obj = xmeans(X, initial_centers, kmax=max_k[-1], splitting_type=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH)
xmeans_obj.process()
n_xmeans_MDL_clusters = len(xmeans_obj.get_clusters())
print('Number of clusters found by xmeans using MDL score: ', n_xmeans_MDL_clusters)
if n_xmeans_MDL_clusters < max_k[-1] and n_xmeans_MDL_clusters not in best_k:
    best_k.append(n_xmeans_MDL_clusters)

# %%
def bss(X, labels, centroids):
    centroid = X.mean(axis=0)
    sizes = np.bincount(labels)
    return np.sum(np.sum(np.square((centroids - centroid)), axis=1)*sizes)

def sse_per_point(X, labels, centroids):
    return np.sum(np.square((X - centroids[labels])), axis=(1 if X.ndim > 1 else 0))

def fit_kmeans(X, params):
    kmeans = KMeans(**params)
    kmeans.fit(X)
    results = {}
    results['model'] = kmeans
    results['SSE'] = kmeans.inertia_
    results['davies_bouldin_score'] = davies_bouldin_score(X, kmeans.labels_)
    results['calinski_harabasz_score'] = calinski_harabasz_score(X, kmeans.labels_)
    #results['silhouette_score'] = silhouette_score(X, kmeans.labels_) 
    results['BSS'] = bss(X, kmeans.labels_, kmeans.cluster_centers_)
    results['n_iter'] = kmeans.n_iter_
    return results

# %%
results = {}
kmeans_params = {}
kmeans_params['max_iter'] = 100
for k in best_k:
    kmeans_params['n_init'] = 10
    kmeans_params['n_clusters'] = k
    kmeans_params['init'] = 'k-means++'
    result = fit_kmeans(X, kmeans_params)
    results[str(k)+'means'] = result

    bisect_kmeans = BisectingKMeans(n_clusters=k, n_init=5).fit(X) # TODO: salvare i risultati anche di questo?
    kmeans_params['n_init'] = 1
    kmeans_params['init'] = bisect_kmeans.cluster_centers_
    result = fit_kmeans(X, kmeans_params)
    results[str(k)+'means_bis_init'] = result

# %%
results_df = pd.DataFrame(results).T
results_df.drop(columns=['model']) # only for printing

# %%
k = 5
kmeans = results[f'{k}means']['model']
labels = results[f'{k}means']['model'].labels_
centroids = results[f'{k}means']['model'].cluster_centers_
centroids_inverse = scaler.inverse_transform(centroids)
incidents_df['cluster'] = labels

# %%
plt.figure(figsize=(8, 4))
for i in range(0, len(centroids_inverse)):
    plt.plot(centroids_inverse[i], marker='o', label='Cluster %s' % i)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster)), features_to_cluster, rotation=90)
plt.legend(fontsize=10)
plt.title('Centroids (original features)')
plt.show()

# %%
plt.figure(figsize=(8, 4))
for i in range(0, len(centroids)):
    plt.plot(centroids[i], marker='o', label='Cluster %s' % i)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster)), features_to_cluster, rotation=90)
plt.legend(fontsize=10)
plt.title('Centroids (scaled features)')
plt.show()

# %%
df = pd.DataFrame()
for i, center in enumerate(centroids):
    tmp_df = pd.DataFrame(dict(r=center, theta=features_to_cluster))
    tmp_df['Centroid'] = f'Centroid {i}'
    df = pd.concat([df,tmp_df], axis=0)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, color='Centroid', color_discrete_sequence=sns.color_palette().as_hex())
fig.show()

# %%
sse_feature = []
for i in range(X.shape[1]):
    sse_feature.append(sse_per_point(X[:,i], kmeans.labels_, kmeans.cluster_centers_[:,i]).sum())

# %%
plt.figure(figsize=(15, 5))
sse_feature, features_to_cluster_sorted = zip(*sorted(zip(sse_feature, features_to_cluster)))
plt.bar(range(len(sse_feature)), sse_feature)
plt.xticks(range(len(sse_feature)), features_to_cluster_sorted)
plt.xticks(rotation=90)
plt.ylabel('SSE')
plt.xlabel('Feature')
plt.title('SSE for each feature')

# %%
counts = np.bincount(labels)
plt.bar(range(len(counts)), counts)
plt.xticks(range(len(counts)))
plt.ylabel('Number of points')
plt.xlabel('Cluster')
plt.title('Number of points per cluster');

# %%
# print top 5 points with highest SSE
sse_points = sse_per_point(X, labels, centroids)
indices_of_top_contributors = np.argsort(sse_points)[-5:]
incidents_df.iloc[indices_of_top_contributors]

# %%
plot_scattermap_plotly(incidents_df, 'cluster', zoom=2, title='Kmeans clustering (with standardized data)')

# %%
def plot_distribution_categorical_attribute(df, attribute):
    attribute_str = attribute.replace('_', ' ').capitalize()
    _, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 2]})
    df[attribute].value_counts().sort_index().plot(kind='bar', ax=axs[0], color='gray')
    axs[0].set_title(f'{attribute_str} distribution in the whole dataset')
    axs[0].set_xlabel(attribute_str)
    axs[0].set_ylabel('Number of incidents')
    day_xt = pd.crosstab(labels, df[attribute])
    day_xt.plot(
        kind='bar',
        stacked=False,
        figsize=(15, 7),
        ax=axs[1],
        color=sns.color_palette('hls').as_hex()
        )
    axs[1].set_title(f'{attribute_str} distribution in each cluster')
    axs[1].set_xlabel('Cluster')
    axs[1].set_ylabel('Number of incidents')
    plt.show()

# %%
for attribute in categorical_features:
    plot_distribution_categorical_attribute(incidents_df, attribute)

# %%
features_to_scatter = features_to_cluster
ncols = 3
nplots = len(features_to_scatter)*(len(features_to_scatter)-1)/2
nrows = int(nplots / ncols)
if nplots % ncols != 0:
    nrows += 1

colors = [sns.color_palette()[c%6] for c in incidents_df['cluster']] # FIXME: assumes having max 6 clusters
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36,36))
id = 0
for i in range(len(features_to_scatter)):
    for j in range(i+1, len(features_to_scatter)):
        x, y = incidents_df[features_to_scatter].columns[i], incidents_df[features_to_scatter].columns[j]
        axs[int(id/ncols)][id%ncols].scatter(incidents_df[x], incidents_df[y], s=20, c=colors)
        for c in range(len(centroids)):
            axs[int(id/ncols)][id%ncols].scatter(
                centroids[c][incidents_df[features_to_scatter].columns.get_loc(x)],
                centroids[c][incidents_df[features_to_scatter].columns.get_loc(y)],
                marker='o', c='white', alpha=1, s=200, edgecolor='k')
            axs[int(id/ncols)][id%ncols].scatter(
                centroids[c][incidents_df[features_to_scatter].columns.get_loc(x)],
                centroids[c][incidents_df[features_to_scatter].columns.get_loc(y)],
                marker='$%d$' % c, c='black', alpha=1, s=50, edgecolor='k')
        axs[int(id/ncols)][id%ncols].set_xlabel(x)
        axs[int(id/ncols)][id%ncols].set_ylabel(y)
        id += 1
for ax in axs[nrows-1, id%ncols:]:
    ax.remove()

legend_elements = []
clusters_ids = incidents_df['cluster'].unique()
for c in sorted(clusters_ids):
    legend_elements.append(Line2D(
        [0], [0], marker='o', color='w', label=f'Cluster {c}', markerfacecolor=sns.color_palette()[c%6]))
f.legend(handles=legend_elements, loc='lower center', ncols=len(clusters_ids))

plt.suptitle(("Clusters in different feature spaces"), fontsize=20)
plt.show()

# %%
pca = PCA()
X_pca = pca.fit_transform(incidents_df[features_to_cluster])

# %%
plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40, c=colors)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA')
plt.legend(handles=legend_elements)

# %%
silhouette_vis = SilhouetteVisualizer(kmeans, title='Silhouette plot') # FIXME: use same color
silhouette_vis.fit(X)
silhouette_vis.show()
# TODO: colorare negli scatter i punti con silhouette negativa?

# %%
visualizer = InterclusterDistance(kmeans)
visualizer.fit(X)
visualizer.show()

# %%
# TODO: similarity matrix (need to do it on a subsample)

# %%
ncols = 3
nplots = len(features_to_cluster)
nrows = int(nplots/ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36,36))
id = 0
for feature in features_to_cluster:
    incidents_df.boxplot(column=feature, by='cluster', ax=axs[int(id/ncols)][id%ncols])
    id += 1
for ax in axs[nrows-1, id%ncols:]:
    ax.remove()

# %%
for feature in features_to_cluster:
    axes = incidents_df[feature].hist(by=incidents_df['cluster'], bins=20, layout=(1,k), figsize=(20, 5))
    plt.suptitle(f'Distribution of {attr} in each cluster', fontweight='bold')
    for i, ax in enumerate(axes):
        ax.set_title(f'Cluster {i}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Number of incidents')

# %%
# metti altrove
# guarda indicatori
# guarda clustering altri


