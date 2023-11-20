# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.offline as pyo
import networkx as nx
import warnings
np.warnings = warnings # altrimenti numpy da problemi con pyclustering, TODO: è un problema solo mio?
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import KMeans, BisectingKMeans
from scipy.spatial.distance import pdist, squareform
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
import utm
import os
import sys
sys.path.append(os.path.abspath('..'))
from plot_utils import *
from clustering_utils import *
# %matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
RANDOM_STATE = 42 # to get reproducible results

# %%
# TODO: si leggerà un solo file che contiene tutto
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv'
)
indicators_df = pd.read_csv(
    '../data/incidents_cleaned_indicators.csv', index_col=0
)
features_to_cluster = [
    'latitude_proj', 'longitude_proj', 'location_importance', 'city_entropy', 'address_entropy',
    'avg_age_participants', 'age_range', 'log_avg_age_mean_SD', 'avg_age_entropy',
    'n_participants', 'n_participants_child_prop', 'n_participants_teen_prop', 'n_participants_adult_entropy',
    'n_males_pr', 'log_n_males_n_males_mean_semest_congd_ratio',
    'n_killed_pr', 'n_injured_pr', 'n_arrested_pr', 'n_unharmed_pr',
    'tags_entropy'
]
categorical_features = [
    'year', 'month', 'day_of_week', 'party', #'state', 'address_type', 
    'firearm', 'air_gun', 'shots', 'aggression', 'suicide',
    'injuries', 'death', 'road', 'illegal_holding', 'house',
    'school', 'children', 'drugs', 'officers', 'organized', 'social_reasons',
    'defensive', 'workplace', 'abduction', 'unintentional'
    ]

# %%
indicators_df = indicators_df.dropna()

# %%
# TODO: da spostare nel file che fa gli indicatori
latlong_projs = utm.from_latlon(indicators_df['latitude'].to_numpy(), indicators_df['longitude'].to_numpy())
scaler= MinMaxScaler()
latlong = scaler.fit_transform(np.stack([latlong_projs[0], latlong_projs[1]]).reshape(-1, 2))
indicators_df['latitude_proj'] = latlong[:,0]
indicators_df['longitude_proj'] = latlong[:,1]

# %%
# scaler= StandardScaler() # TODO: fare in questo notebook
# X = scaler.fit_transform(indicators_df.values) # TODO: come scegliere?
X = indicators_df[features_to_cluster].values

# %% [markdown]
# ## Identification of the best value of k

# %%
def plot_score_varying_k(X, kmeans_params, metric, start_k, max_k):
    if metric == 'SSE':
        metric = 'distortion'
        metric_descr = 'SSE'
    elif metric == 'calinski_harabasz':
        metric_descr = 'Calinski Harabasz Score'
    elif metric == 'silhouette':
        metric_descr = 'Silhouette Score'
    else:
        raise ValueError('Metric not supported')
    
    _, axs = plt.subplots(nrows=1, ncols=len(max_k) if len(max_k)!= 1 else 2, figsize=(30,5))

    best_k = []

    for i in range(len(max_k)):
        kmeans_params['n_clusters'] = i
        kmeans = KMeans(**kmeans_params)

        elbow_vis = KElbowVisualizer(kmeans, k=(start_k, max_k[i]), metric=metric, timings=False, ax=axs[i])
        elbow_vis.fit(X)
        if elbow_vis.elbow_value_ != None and elbow_vis.elbow_value_ not in best_k:
            best_k.append(elbow_vis.elbow_value_)
        axs[i].set_title(f'{metric_descr} elbow for K-Means clustering (K in [{str(start_k)}, {str(max_k[i])}])')
        axs[i].set_ylabel(metric_descr)
        axs[i].set_xlabel('K')
        axs[i].legend([
            f'{metric_descr}',
            f'elbow at K = {str(elbow_vis.elbow_value_)}, {metric_descr} = {elbow_vis.elbow_score_:0.2f}'
        ])

        if len(max_k)==1:
            axs[1].remove()
    
    plt.show()
    return best_k

# %%
MAX_ITER = 300
N_INIT = 10
INIT_METHOD = 'k-means++'
kmeans_params = {'init': INIT_METHOD, 'n_init': N_INIT, 'max_iter': MAX_ITER, 'random_state': RANDOM_STATE}
max_k = 30
best_k = []

# %% [markdown]
# Ci permette anche di valutare la sensibilità all'inizializzazione dei centroidi. Più di 30 difficile da interpretare. Sia partendo da 1 che da 2. Come interpola.

# %%
ks = plot_score_varying_k(X=X, kmeans_params=kmeans_params, metric='SSE', start_k=1, max_k=[10, 20, 30])
best_k += ks

# %%
# k = plot_k_elbow(X=X, kmeans_params=kmeans_params, metric='SSE', start_k=2, max_k=[10, 20, 30])
# best_k += ks

# %%
# k = plot_k_elbow(X=X, kmeans_params=kmeans_params, metric='silhouette', start_k=1, max_k=[20])
# best_k += ks

# %%
# k = plot_k_elbow(X=X, kmeans_params=kmeans_params, metric='calinski_harabasz', start_k=1, max_k=[20])
# best_k += ks

# %%
initial_centers = kmeans_plusplus_initializer(data=X, amount_centers=1, random_state=RANDOM_STATE).initialize()
xmeans_MDL_instance = xmeans( # TODO: assicurarsi di starlo usando nel modo in cui vogliamo, si arresta prima in base al BIC?
    data=X,
    initial_centers=initial_centers,
    kmax=max_k,
    splitting_type=splitting_type.BAYESIAN_INFORMATION_CRITERION,
    random_state=RANDOM_STATE
)
xmeans_MDL_instance.process()
n_xmeans_BIC_clusters = len(xmeans_MDL_instance.get_clusters())
print('Number of clusters found by xmeans using BIC score: ', n_xmeans_BIC_clusters)
if n_xmeans_BIC_clusters < max_k and n_xmeans_BIC_clusters not in best_k:
    best_k.append(n_xmeans_BIC_clusters)

# %%
xmeans_MDL_instance = xmeans( # TODO: idem come sopra
    data=X,
    initial_centers=initial_centers,
    kmax=max_k,
    splitting_type=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH,
    random_state=RANDOM_STATE
)
xmeans_MDL_instance.process()
n_xmeans_MDL_clusters = len(xmeans_MDL_instance.get_clusters())
print('Number of clusters found by xmeans using MDL score: ', n_xmeans_MDL_clusters)
if n_xmeans_MDL_clusters < max_k and n_xmeans_MDL_clusters not in best_k:
    best_k.append(n_xmeans_MDL_clusters)

# %%
def fit_kmeans(X, params):
    kmeans = KMeans(**params)
    kmeans.fit(X)
    results = {}
    results['model'] = kmeans
    results['SSE'] = kmeans.inertia_
    results['BSS'] = compute_bss_per_cluster(X=X, clusters=kmeans.labels_, centroids=kmeans.cluster_centers_, weighted=True).sum()
    results['davies_bouldin_score'] = davies_bouldin_score(X=X, labels=kmeans.labels_)
    results['calinski_harabasz_score'] = calinski_harabasz_score(X=X, labels=kmeans.labels_)
    #results['silhouette_score'] = silhouette_score(X=X, labels=kmeans.labels_) 
    results['n_iter'] = kmeans.n_iter_
    return results

# %%
best_k=[4]

# %%
results = {}
kmeans_params = {}
kmeans_params['random_state'] = RANDOM_STATE
kmeans_params['max_iter'] = MAX_ITER
for k in best_k:
    kmeans_params['n_init'] = N_INIT
    kmeans_params['n_clusters'] = k
    kmeans_params['init'] = INIT_METHOD
    result = fit_kmeans(X=X, params=kmeans_params)
    results[str(k)+'means'] = result

    bisect_kmeans = BisectingKMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit(X) # TODO: salvare i risultati anche di questo?
    kmeans_params['n_init'] = 1
    kmeans_params['init'] = bisect_kmeans.cluster_centers_
    result = fit_kmeans(X=X, params=kmeans_params)
    results[str(k)+'means_bis_init'] = result

# %%
results_df = pd.DataFrame(results).T
results_df.drop(columns=['model'])

# %%
k = 4
kmeans = results[f'{k}means']['model']
clusters = results[f'{k}means']['model'].labels_
centroids = results[f'{k}means']['model'].cluster_centers_
indicators_df['cluster'] = clusters


# %% [markdown]
# ## Characterization of the clusters

# %% [markdown]
# ### Analysis of the centroids

# %%
def plot_parallel_coordinates(points, features, figsize=(8, 4), title=None): # TODO: va fatto sulle feature trasformate giusto? per algoritmi non centroid based non ha senso?
    plt.figure(figsize=figsize)
    for i in range(0, len(points)):
        plt.plot(points[i], marker='o', label='Cluster %s' % i)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(range(0, len(features)), features, rotation=90)
    plt.legend(fontsize=10)
    plt.title(title)
    plt.show()


# %%
plot_parallel_coordinates(points=centroids, features=features_to_cluster, title=f'Centroids of {k}-means clusters')


# %%
def plot_spider(points, features, title=None, palette=sns.color_palette()):
    df = pd.DataFrame()
    for i, center in enumerate(points):
        tmp_df = pd.DataFrame(dict(r=center, theta=features))
        tmp_df['Centroid'] = f'Centroid {i}'
        df = pd.concat([df,tmp_df], axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        fig = px.line_polar(df, r='r', theta='theta', line_close=True, color='Centroid', color_discrete_sequence=palette.as_hex())
    fig.update_layout(title=title)
    fig.show()
    pyo.plot(fig, filename=f'../html/centroids_spider.html', auto_open=False)


# %%
plot_spider(points=centroids, features=features_to_cluster, title=f'Centroids of {k}-means clusters')

# %% [markdown]
# ## Distribution of variables within the clusters (and in the whole dataset)

# %%
incidents_df = incidents_df.loc[indicators_df.index]
incidents_df['cluster'] = clusters

# %%
plot_scattermap_plotly(incidents_df, 'cluster', zoom=2, title='Incidents clustered by Kmeans')

# %%
for i in range(k): # TODO: fare subplot (è più complicato di quello che sembra, plotly non supporta subplots con mappe)
    plot_scattermap_plotly(
        incidents_df[incidents_df['cluster']==i],
        'cluster',
        zoom=2.5,
        height=400,
        title=f'Cluster {i}',
        color_sequence=sns.color_palette().as_hex()[i:],
        black_nan=False,
        showlegend=False
    )

# %%
for feature in categorical_features:
    plot_bars_by_cluster(df=incidents_df, feature=feature, cluster_column='cluster')

# %%
scatter_by_cluster(
    df=indicators_df,
    features=[
        'latitude_proj',
        'longitude_proj',
        'avg_age_participants',
        'age_range',
        'n_participants'
    ],
    cluster_column='cluster',
    centroids=centroids,
    figsize=(15, 10)
)

# %%
pca = PCA()
X_pca = pca.fit_transform(X)

# %%
exp_var_pca = pca.explained_variance_ratio_
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, align='center')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component')
plt.title('Explained variance by principal component')
plt.xticks(np.arange(0,len(exp_var_pca),1.0));

# %%
palette = [sns.color_palette()[i] for i in range(k)]
scatter_pca_features_by_cluster(
    X_pca=X_pca,
    n_components=4,
    clusters=clusters,
    palette=palette,
    hue_order=None,
    title='Clusters in PCA space'
)

# %%
plot_boxes_by_cluster(
    df=indicators_df,
    features=features_to_cluster,
    cluster_column='cluster',
    figsize=(15, 35),
    title='Box plots of features by cluster'
)

# %%
plot_violin_by_cluster(
    df=indicators_df,
    features=features_to_cluster,
    cluster_column='cluster',
    figsize=(15, 20),
    title='Violin plots of features by cluster'
)

# %%
for feature in features_to_cluster:
    plot_hists_by_cluster(
        df=indicators_df,
        feature=feature,
        cluster_column='cluster',
        title=f'Distribution of {feature} in each cluster'
    )

# %% [markdown]
# ## Evaluation of the clustering results

# %%
sse_feature = []
for i in range(X.shape[1]):
    sse_feature.append(compute_se_per_point(X=X[:,i], clusters=clusters, centroids=centroids[:,i]).sum())

# %%
plt.figure(figsize=(15, 5))
sse_feature_sorted, clustering_features_sorted = zip(*sorted(zip(sse_feature, features_to_cluster)))
plt.bar(range(len(sse_feature_sorted)), sse_feature_sorted)
plt.xticks(range(len(sse_feature_sorted)), clustering_features_sorted)
plt.xticks(rotation=90)
plt.ylabel('SSE')
plt.xlabel('Feature')
plt.title('SSE per feature')

# %%
plot_clusters_size(clusters)

# %%
# print top 5 points with highest SSE
se_per_point = compute_se_per_point(X=X, clusters=clusters, centroids=centroids)
indices_of_top_contributors = np.argsort(se_per_point)[-5:]
incidents_df.iloc[indices_of_top_contributors]

# %%
plot_scores_per_point(score_per_point=se_per_point, clusters=clusters, score_name='SE')

# %%
silhouette_per_point = silhouette_samples(X=X, labels=clusters)

# %%
plot_scores_per_point(score_per_point=silhouette_per_point, clusters=clusters, score_name='Silhouette score')

# %%
# NOTE: c'è una libreria che lo fa solo per la silhouette (forse meglio usare la nostra funzione generica)
# silhouette_vis = SilhouetteVisualizer(kmeans, title='Silhouette plot', colors=sns.color_palette().as_hex())
# silhouette_vis.fit(X)
# silhouette_per_point = silhouette_vis.silhouette_samples_
# silhouette_vis.show()

# %%
clusters_silh = np.full(clusters.shape[0], -1)
for i, s in enumerate(silhouette_per_point):
    if s >= 0:
        clusters_silh[i] = clusters[i]

palette=([(0,0,0)]+[sns.color_palette()[i] for i in range(k)])
hue_order=[i for i in range(k)]+[-1]
scatter_pca_features_by_cluster(
    X_pca=X_pca,
    n_components=4,
    clusters=clusters_silh,
    palette=palette,
    hue_order=hue_order,
    title='Clusters in PCA space (points with silhouette < 0 marked with -1)'
)

# %%
visualizer = InterclusterDistance(kmeans)
visualizer.fit(X)
visualizer.show()

# %%
# compute cohesion for each cluster
se_per_cluster = np.zeros(k)
sizes = np.ones(centroids.shape[0])
for i in range(k):
    se_per_cluster[i] = np.sum(se_per_point[np.where(clusters == i)[0]])/sizes[i] # TODO: weigthed (or not?)
# compute separation for each cluster
bss_per_cluster = compute_bss_per_cluster(X, clusters, centroids, weighted=True) # TODO: weigthed (or not?)
# compute average silhouette score for each cluster
silhouette_per_cluster = np.zeros(k)
for i in range(k):
    silhouette_per_cluster[i] = silhouette_per_point[np.where(clusters == i)[0]].mean() # TODO: already weighted

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
axs[0].bar(range(k), se_per_cluster, color=sns.color_palette())
axs[0].set_ylim(30000, 0)
axs[0].set_title('Cohesion') # TODO: non è proprio cohesion
axs[0].set_ylabel('SSE')
axs[1].bar(range(k), bss_per_cluster, color=sns.color_palette())
axs[1].set_title('Separation')
axs[1].set_ylabel('BSS')
axs[2].bar(range(k), silhouette_per_cluster, color=sns.color_palette())
axs[2].set_title('Silhouette')
axs[2].set_ylabel('Silhouette score')

for i in range(3):
    axs[i].set_xlabel('Cluster')
    axs[i].set_xticks(range(k))
    axs[i].set_xticklabels(range(k))

plt.suptitle('Cohesion and separation measures for each cluster', fontweight='bold')

# %%
centroids_dm = pd.DataFrame(squareform(pdist(centroids)), columns=range(k), index=range(k))
centroids_dm

# %%
G = nx.from_numpy_array(centroids_dm.values)
clusterings = centroids_dm.columns.values
G = nx.relabel_nodes(G, dict(zip(range(len(clusterings)), clusterings)))
edge_labels = {(i, j): "{:.2f}".format(centroids_dm[i][j]) for i, j in G.edges()}
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=sns.color_palette().as_hex()[:len(clusterings)])
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# %%
scatter_pca_features_by_score(
    X_pca=X_pca,
    clusters=clusters,
    x_component=1,
    y_component=2,
    score_per_point=silhouette_per_point,
    score_name='Silhouette score'
)

# %%
dm, idm = plot_distance_matrices(X=X, n_samples=500, clusters=clusters, random_state=RANDOM_STATE)

# %%
labels1 = [0,0,0,1,1,1,2,2,2,3,3,3]
labels2 = [1,1,1,0,0,2,2,2,2,3,3,4]
labels3 = [1,1,1,1,0,2,2,3,2,3,3,4]
sankey_plot([labels1, labels2, labels3], labels_titles=['Kmeans', 'DBSCAN', 'Heirarchical'], title='Clusterings comparison')

# %%
compute_permutation_invariant_external_metrics(incidents_df, 'cluster', categorical_features)

# %%
clusterings=[
    [0,0,0,1,1,1,2,2,2,3,3,3],
    [1,1,1,0,0,2,2,2,2,3,3,4],
    [1,1,1,0,0,2,2,2,2,3,3,3]
]
labels = [
    'KMeans',
    'DBSCAN',
    'Heirarchical'
]
adj_rand_scores = compute_score_between_clusterings(
    clusterings=clusterings,
    labels=labels,
    score_fun=adjusted_rand_score,
    score_name='Adjusted Rand Score',
    figsize=(5,4)
)

# %%
label1 = [0,0,0,1,1,1,2,2,2]
label2 = [1,1,1,0,0,2,2,2,2]

label1, label2 = align_labels(label1, label2) # TODO: questa cosa è okay?
label2

# %%
confusion_matrix(label1, label2)

# %%
cm = np.array( # TODO: l'entropia calcolata sul libro è sbagliata?
    [
        [3,5,40,506,96,27],
        [4,7,280,29,39,2],
        [1,1,1,7,4,671],
        [10,162,3,119,73,2],
        [331,22,5,70,13,23],
        [5,358,12,212,48,13]
    ]
)

# %%
purities = np.max(cm, axis=1) / np.sum(cm, axis=1)
print('Purity per cluster:')
print(purities)

# %%
purity = np.sum((purities * np.sum(cm, axis=1)) / np.sum(cm))
print(f'Overall purity: {purity}')

# %%
probs = cm / np.sum(cm, axis=1)
log_probs = np.log2(probs, out=np.zeros_like(probs), where=(probs!=0)) # 0 if prob=0
entropies = -np.sum(np.multiply(probs, log_probs), axis=1)
print('Entropy per cluster:')
print(entropies)

# %%
entropy = np.sum((entropies * np.sum(cm, axis=1)) / np.sum(cm))
print(f'Overall entropy: {entropy}')

# %%
compute_external_metrics(df=incidents_df, cluster_column='cluster', external_features=categorical_features) # TODO: nan purity and entropy? only on classes with same size?
