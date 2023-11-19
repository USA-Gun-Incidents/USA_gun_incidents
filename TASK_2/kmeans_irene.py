# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.offline as pyo
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
scatter_pca_features_by_score(
    X_pca=X_pca,
    clusters=clusters,
    x_component=1,
    y_component=2,
    score_per_point=silhouette_per_point,
    score_name='Silhouette score'
)

# %%
labels1 = [0,0,0,1,1,1,2,2,2,3,3,3]
labels2 = [1,1,1,0,0,2,2,2,2,3,3,4]
sankey_plot(labels1, labels2, title='KMeans clustering vs Other clustering')

# %%
# NOTE
# external:
# - entropy
# - purity

# internal:
# - sse, only for centroid based
# - promixity matrix (not good for density based clusteing)
# - cohesion (somma delle distanze interne a ciascun cluster, media per unico score)
# - separation (somma delle distanza tra un punto in un cluster e tutti gli altri punti in altri cluster, media per unico score)

# proximity weighted e non interna
# sep and coh weighted
# approccio statistico: estrai lo stesso numero di punti costruendo le feature pescando dalle distribuzioni di ciascuna
# applica k-means HOPKINS: pyclustertend
# cophenetic
# fare knee anche con altri algoritmi
# fai proximity (come invertire distanza), correlation as a single score


# %%
cat_metrics_df = pd.DataFrame()

adj_rand_scores = []
homogeneity_scores = []
completeness_scores = []
mutual_info_scores = []

for column in categorical_features: # all permutation invariant
    adj_rand_scores.append(adjusted_rand_score(incidents_df[column], incidents_df['cluster']))
    mutual_info_scores.append(normalized_mutual_info_score(incidents_df[column], incidents_df['cluster'], average_method='arithmetic'))
    homogeneity_scores.append(homogeneity_score(incidents_df[column], incidents_df['cluster']))
    completeness_scores.append(completeness_score(incidents_df[column], incidents_df['cluster']))

cat_metrics_df['feature'] = categorical_features
cat_metrics_df['adjusted rand score'] = adj_rand_scores
cat_metrics_df['normalized mutual information'] = mutual_info_scores
cat_metrics_df['homogeneity'] = homogeneity_scores
cat_metrics_df['completeness'] = completeness_scores


cat_metrics_df.set_index(['feature'], inplace=True)
cat_metrics_df

# %%
from sklearn.metrics import confusion_matrix

def align_labels(label1, label2):
    cm = confusion_matrix(label1, label2)
    cm_argmax = cm.argmax(axis=0)
    label2 = np.array([cm_argmax[i] for i in label2])
    return label1, label2

# %%
label1 = [0,0,0,1,1,1,2,2,2]
label2 = [1,1,1,0,0,2,2,2,2]

label1, label2 = align_labels(label1, label2)
label2

# %%
confusion_matrix(label1, label2)

# %%
cat_clf_metrics_df = pd.DataFrame()

accuracy = []
f1 = []
precision = []
recall = []
roc_auc = []
# TODO: entropy and purity
k_categorical_features = []

for column in categorical_features: # TODO: bisognerebbe mergiare cluster vicini e trovare matching?
    if incidents_df['party'].unique().shape[0] != k:
        continue
    k_categorical_features.append(column)
    _, cluster_labels = align_labels(incidents_df[column], incidents_df['cluster'])
    accuracy.append(accuracy_score(incidents_df[column], cluster_labels))
    f1.append(f1_score(incidents_df[column], cluster_labels, average='weighted'))
    precision.append(precision_score(incidents_df[column], cluster_labels, average='weighted', zero_division=0))
    recall.append(recall_score(incidents_df[column], cluster_labels, average='weighted', zero_division=0))
    roc_auc.append(roc_auc_score(incidents_df[column], cluster_labels, multi_class='ovr', average='weighted'))

cat_clf_metrics_df['feature'] = k_categorical_features
cat_clf_metrics_df['accuracy'] = accuracy
cat_clf_metrics_df['f1'] = f1
cat_clf_metrics_df['precision'] = precision
cat_clf_metrics_df['recall'] = recall
cat_clf_metrics_df.set_index(['feature'], inplace=True)
cat_clf_metrics_df



# %%
# TODO: subsample con stratificazione in base a numero di punti per cluster
subsampled_incidents_df = incidents_df.groupby('cluster', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=0))
subsampled_incidents_df.reset_index(inplace=True)
subsampled_incidents_df.sort_values(by=['cluster'], inplace=True)

dm = squareform(pdist(X[subsampled_incidents_df.index]))

n_subsampled_points = subsampled_incidents_df.shape[0]
im = np.zeros((n_subsampled_points, n_subsampled_points))
sub_labels = subsampled_incidents_df['cluster'].values
for i in range(n_subsampled_points):
    for j in range(n_subsampled_points):
        if sub_labels[i] == sub_labels[j]:
            im[i, j] = 1

corr_matrix = np.corrcoef(dm, im)

plt.matshow(corr_matrix)

# %%
plt.matshow(im)

# %%
plt.matshow(dm)

# %%
sns.heatmap(dm)

# %%
# from scipy.stats import pearsonr
# corrm = np.zeros_like(dm)
# for i in range(n_subsampled_points):
#     for j in range(n_subsampled_points):
#         corrm[i][j] = pearsonr(dm[i], im[:,j])[0]

# %%
# plt.matshow(corrm)
