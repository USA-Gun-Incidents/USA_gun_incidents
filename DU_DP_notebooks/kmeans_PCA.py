# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as mplt
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
import json
np.warnings = warnings # altrimenti numpy da problemi con pyclustering, TODO: è un problema solo mio?
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, adjusted_rand_score
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
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %%
incidents_df = pd.read_csv(
    '../data/incidents_indicators.csv',
    index_col=0
)

f = open('../data/indicators_names.json')
features_to_cluster = json.loads(f.read())

categorical_features = [
    'year', 'month', 'day_of_week', 'party', #'state', 'address_type', 'county', 'city'
    'firearm', 'air_gun', 'shots', 'aggression', 'suicide',
    'injuries', 'death', 'road', 'illegal_holding', 'house',
    'school', 'children', 'drugs', 'officers', 'organized', 'social_reasons',
    'defensive', 'workplace', 'abduction', 'unintentional'
    # 'incident_characteristics1', 'incident_characteristics2'
    ]
# other interesting features:
# poverty_perc, date
incidents_df = incidents_df.dropna(subset=features_to_cluster)

# %%
features_to_cluster_no_coord = features_to_cluster[2:]
features_to_cluster_no_coord

# %%
incidents_df.sample(2, random_state=1)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=incidents_df[features_to_cluster],ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler_obj = MinMaxScaler()
normalized_indicators = pd.DataFrame(data=scaler_obj.fit_transform(incidents_df[features_to_cluster].values), columns=features_to_cluster)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=normalized_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
pca = PCA()
X_pca = pca.fit_transform(normalized_indicators)
pca_df = pd.DataFrame(index=incidents_df.index)

# %%
nrows=4
ncols=6
row=0
fig, axs = mplt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), sharex=True, sharey=True)
for i, col in enumerate(normalized_indicators.columns):
    if i != 0 and i % ncols == 0:
        row += 1
    axs[row][i % ncols].scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40, c=normalized_indicators[col], cmap='viridis')
    axs[row][i % ncols].set_title(col)
    axs[row][i % ncols].set_xlabel("1st eigenvector")
    axs[row][i % ncols].set_ylabel("2nd eigenvector")

# %%
x = X_pca[:, 0]
y = X_pca[:, 2]
z = X_pca[:, 1]
fig = px.scatter_3d(x=x, y=y, z=z, labels={'x': '1st eigenvector', 'y': '3rd eigenvector', 'z': '2nd eigenvector'})
fig.show()

# %%
exp_var_pca = pca.explained_variance_ratio_
diff_var = []

for i, var in enumerate(exp_var_pca[:-1]):
    diff_var.append( var-exp_var_pca[i+1])


xtick = []
gap = 0
for i, var in enumerate(diff_var):
    xtick.append(i+gap)
    if i != 0 and diff_var[i-1] <= var:
        gap += 0.5
        if gap == 0.5:
            plt.axvline(x = i+gap+0.25, color = 'green', linestyle = '-.', alpha=0.5, label='possible cut')
        else:
             plt.axvline(x = i+gap+0.25, color = 'green', linestyle = '-.', alpha=0.5)
    

#xtick = [0,1,2,3,4,5.5,6.5,7.5,8.5,9.5,10.5,12,13,14,15,16,17,18,19,20]
#diff_var = list(zip(xtick, diff_var))
xtick.append(23)

plt.bar(xtick, exp_var_pca, align='center')
plt.plot(xtick[1:], diff_var, label='difference from prevoius variance', color='orange')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component')
plt.title('Explained variance by principal component')
plt.xticks(xtick, range(exp_var_pca.shape[0]))
plt.legend();

# %%
def get_reconstruction_error(x_pca, x_orig, pca, n_comp):
    dummy = np.matmul(x_pca[:,:n_comp], pca.components_[:n_comp,:]) + pca.mean_
    return pd.DataFrame(index=x_orig.index, data=np.sum((dummy - x_orig.values)**2, axis=1))

# %%
pca_col = ['1st_comp',
 '2nd_comp',
 '3rd_comp',
 '4th_comp',
 '5th_comp',
 '6th_comp',
 '7th_comp',
 '8th_comp',
 '9th_comp',
 '10th_comp',
 '11th_comp',
 '12th_comp',
 '13th_comp',
 '14th_comp',
 '15th_comp',
 '16th_comp',
 '17th_comp',
 '18th_comp',
 '19th_comp']

# %%
pca_indicators = pd.DataFrame(index=normalized_indicators.index, data=X_pca, columns=pca_col)

# %%
pca_indicators['PCA_rec_error_2C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 2)
pca_indicators['PCA_rec_error_6C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 6)
pca_indicators['PCA_rec_error_8C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 8)

# %%
pca_indicators.sample(3)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=pca_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
pca_normalized_indicators = pd.DataFrame(data=scaler_obj.fit_transform(pca_indicators.values), columns=pca_indicators.columns)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=pca_normalized_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %%
hist_box_plot(
    pca_normalized_indicators,
    'PCA_rec_error_6C',
    title='PCA_rec_error_6C',
    bins=int(np.log(pca_normalized_indicators.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
clustered_components = ['1st_comp',
    '2nd_comp',
    '3rd_comp',
    '4th_comp',
    '5th_comp',
    '6th_comp',
    '7th_comp',
    '8th_comp',
    '9th_comp']
X = pca_normalized_indicators[clustered_components].values

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
RANDOM_STATE = 1
kmeans_params = {'init': INIT_METHOD, 'n_init': N_INIT, 'max_iter': MAX_ITER, 'random_state': RANDOM_STATE}
max_k = 50
possible_k = []

# %% [markdown]
# Ci permette anche di valutare la sensibilità all'inizializzazione dei centroidi. Più di 30 difficile da interpretare. Sia partendo da 1 che da 2. Come interpola.

# %%
ks = plot_score_varying_k(X=X, kmeans_params=kmeans_params, metric='SSE', start_k=2, max_k=[10, 25, 50])
possible_k += ks

# %%
ks = plot_score_varying_k(X=X, kmeans_params=kmeans_params, metric='silhouette', start_k=2, max_k=[50])
possible_k += ks

# %%
k = plot_score_varying_k(X=X, kmeans_params=kmeans_params, metric='calinski_harabasz', start_k=2, max_k=[50])
possible_k += ks

# %%
possible_k = [3,6,10,3,2]

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
best_k= list(set(possible_k))
best_k = sorted(best_k)
best_k

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
    results[str(k)+'_means'] = result

    bisect_kmeans = BisectingKMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit(X) # TODO: salvare i risultati anche di questo?
    kmeans_params['n_init'] = 1
    kmeans_params['init'] = bisect_kmeans.cluster_centers_
    result = fit_kmeans(X=X, params=kmeans_params)
    results[str(k)+'_means_bis_init'] = result

# %%
results_df = pd.DataFrame(results).T
results_df.drop(columns=['model'])

# %%
ax = results_df.plot()
ax.set_yscale('log')
ax.set_xticks(range(0,len(results_df.index)))
ax.set_xticklabels(results_df.index, rotation=45, ha='right');


# %%
k = 3
kmeans = results[f'{k}_means']['model']
clusters = results[f'{k}_means']['model'].labels_
centroids = results[f'{k}_means']['model'].cluster_centers_
incidents_df['cluster'] = clusters + 1
normalized_indicators['cluster'] = clusters + 1

# %% [markdown]
# ## Characterization of the clusters

# %% [markdown]
# ### Analysis of the centroids

# %%
def plot_parallel_coordinates_components(points, features, figsize=(8, 4), title=None): # TODO: va fatto sulle feature trasformate giusto? per algoritmi non centroid based non ha senso?
    plt.figure(figsize=figsize)
    for i in range(0, len(points)):
        plt.plot(points[i], marker='o', label='Cluster %s' % i)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(range(0, len(features)), features, rotation=90)
    plt.legend(fontsize=10)
    plt.title(title)
    plt.show()

# %%
def plot_parallel_coordinates(dataframe, cluster, features, figsize=(8, 4), title=None): # TODO: va fatto sulle feature trasformate giusto? per algoritmi non centroid based non ha senso?
    plt.figure(figsize=figsize)

    for i, row in dataframe.sample(10, random_state=1).iterrows():
        plt.plot(row[features], marker='o', label=row[cluster], alpha=0.3)
        
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(range(0, len(features)), features, rotation=90)
    plt.legend(fontsize=10)
    plt.title(title)
    plt.show()

# %%
plot_parallel_coordinates_components(points=centroids, features=clustered_components, title=f'Centroids of {k}-means clusters')

# %%
plot_parallel_coordinates(normalized_indicators, 'cluster', features_to_cluster, title=f'Centroids of {k}-means clusters') #da fare meglio se ha senso...

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
    pyo.plot(fig, filename=f'../html/centroids_PCA_spider.html', auto_open=False)

# %%
plot_spider(points=centroids, features=clustered_components, title=f'Centroids of {k}-means clusters')

# %% [markdown]
# ## Distribution of variables within the clusters (and in the whole dataset)

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
    df=incidents_df,
    features=features_to_cluster,
    cluster_column='cluster',
    centroids=centroids,
    figsize=(15, 10),
    ncols=3
)

# %%
plot_violin_by_cluster(
    df=incidents_df,
    features=features_to_cluster,
    cluster_column='cluster',
    figsize=(15, 20),
    title='Violin plots of features by cluster'
)

# %%
for feature in features_to_cluster:
    plot_hists_by_cluster(
        df=incidents_df,
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
fig, axs = plt.subplots(1)
plot_clusters_size(clusters=clusters, ax=axs, title='Clusters size')
fig.show()

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
write_clusters_to_csv(clusters, './kmeans_clusters.csv')
write_clusters_to_csv(clusters, './DBSCAN_clusters.csv')
write_clusters_to_csv(clusters, './heirarchical_clusters.csv')

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


