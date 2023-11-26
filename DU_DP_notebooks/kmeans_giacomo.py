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
incidents_df = incidents_df.dropna()

# %%
latlong_projs = utm.from_latlon(incidents_df['latitude'].to_numpy(), incidents_df['longitude'].to_numpy())
scaler= MinMaxScaler()
latlong = scaler.fit_transform(np.stack([latlong_projs[0], latlong_projs[1]]).reshape(-1, 2))
incidents_df['latitude_proj'] = latlong[:,0]
incidents_df['longitude_proj'] = latlong[:,1]

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
X = incidents_df[features_to_cluster].values

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
kmeans_params = {'init': INIT_METHOD, 'n_init': N_INIT, 'max_iter': MAX_ITER}
max_k = 30
best_k = []

# %% [markdown]
# Ci permette anche di valutare la sensibilità all'inizializzazione dei centroidi. Più di 30 difficile da interpretare. Sia partendo da 1 che da 2. Come interpola.

# %%
ks = plot_score_varying_k(X=X, kmeans_params=kmeans_params, metric='SSE', start_k=1, max_k=[10, 25, 50])
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
initial_centers = kmeans_plusplus_initializer(data=X, amount_centers=1).initialize()
xmeans_MDL_instance = xmeans(data=X, initial_centers=initial_centers, kmax=max_k, splitting_type=splitting_type.BAYESIAN_INFORMATION_CRITERION)
xmeans_MDL_instance.process()
n_xmeans_BIC_clusters = len(xmeans_MDL_instance.get_clusters())
print('Number of clusters found by xmeans using BIC score: ', n_xmeans_BIC_clusters)
if n_xmeans_BIC_clusters < max_k and n_xmeans_BIC_clusters not in best_k:
    best_k.append(n_xmeans_BIC_clusters)

# %%
xmeans_MDL_instance = xmeans(data=X, initial_centers=initial_centers, kmax=max_k, splitting_type=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH)
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
    results['davies_bouldin_score'] = davies_bouldin_score(X=X, labels=kmeans.labels_)
    results['calinski_harabasz_score'] = calinski_harabasz_score(X=X, labels=kmeans.labels_)
    #results['silhouette_score'] = silhouette_score(X=X, labels=kmeans.labels_) 
    results['BSS'] = bss(X=X, clusters=kmeans.labels_, centroids=kmeans.cluster_centers_)
    results['n_iter'] = kmeans.n_iter_
    return results

# %%
best_k=[4]

# %%
results = {}
kmeans_params = {}
kmeans_params['max_iter'] = MAX_ITER
for k in best_k:
    kmeans_params['n_init'] = N_INIT
    kmeans_params['n_clusters'] = k
    kmeans_params['init'] = INIT_METHOD
    result = fit_kmeans(X=X, params=kmeans_params)
    results[str(k)+'means'] = result

    bisect_kmeans = BisectingKMeans(n_clusters=k, n_init=5).fit(X) # TODO: salvare i risultati anche di questo?
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
#centroids_inverse = scaler.inverse_transform(centroids)
indicators_df['cluster'] = clusters

# %%
# plt.figure(figsize=(8, 4))
# for i in range(0, len(centroids_inverse)):
#     plt.plot(centroids_inverse[i], marker='o', label='Cluster %s' % i)
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.xticks(range(0, len(features_to_cluster)), features_to_cluster, rotation=90)
# plt.legend(fontsize=10)
# plt.title('Centroids (original features)')
# plt.show()

# %%
plt.figure(figsize=(8, 4))
for i in range(0, len(centroids)):
    plt.plot(centroids[i], marker='o', label='Cluster %s' % i)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster)), features_to_cluster, rotation=90)
plt.legend(fontsize=10)
plt.title('Centroids (scaled features)')
plt.show()

# TODO: 

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
    sse_feature.append(sse_per_point(X=X[:,i], clusters=clusters, centroids=kmeans.cluster_centers_[:,i]).sum())

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
counts = np.bincount(clusters)
plt.bar(range(len(counts)), counts)
plt.xticks(range(len(counts)))
plt.ylabel('Number of points')
plt.xlabel('Cluster')
plt.title('Number of points per cluster');

# %%
# print top 5 points with highest SSE
sse_points = sse_per_point(X=X, clusters=clusters, centroids=centroids)
indices_of_top_contributors = np.argsort(sse_points)[-5:]
incidents_df.iloc[indices_of_top_contributors]

# %%
plot_scattermap_plotly(indicators_df, 'cluster', zoom=2, title='Incidents clustered by Kmeans')

# %%
incidents_df = incidents_df.loc[indicators_df.index]
incidents_df['cluster'] = clusters

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
silhouette_vis = SilhouetteVisualizer(kmeans, title='Silhouette plot', colors=sns.color_palette().as_hex())
silhouette_vis.fit(X)
silhouette_scores = silhouette_vis.silhouette_samples_
silhouette_vis.show()

# %%
clusters_silh = np.full(clusters.shape[0], -1)
for i, s in enumerate(silhouette_scores):
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
plot_boxes_by_cluster(
    df=indicators_df,
    features=features_to_cluster,
    cluster_column='cluster',
    figsize=(15, 35)
)

# %%
for feature in features_to_cluster:
    plot_hists_by_cluster(
        df=indicators_df,
        feature=feature,
        cluster_column='cluster'
    )

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


