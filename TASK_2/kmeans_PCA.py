# -*- coding: utf-8 -*-
# %% [markdown]
# # KMeans clustering of Principal Components

# %% [markdown]
# We import the libraries

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

# %% [markdown]
# We load the data:

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

# %% [markdown]
# We select the features to cluster:

# %%
features_to_cluster_no_coord = features_to_cluster[2:]
features_to_cluster_no_coord

# %%
incidents_df[features_to_cluster_no_coord].describe()

# %% [markdown]
# We display the distribution of the selected features:

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=incidents_df[features_to_cluster_no_coord],ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# In order to obtain meaningful results, we must ensure that no feature presents too high magnitude that could overshadow the contributions of others. To implement this we normalize all features between 0 and 1.

# %%
from sklearn.preprocessing import MinMaxScaler
scaler_obj = MinMaxScaler()
normalized_indicators = pd.DataFrame(data=scaler_obj.fit_transform(incidents_df[features_to_cluster_no_coord].values), columns=features_to_cluster_no_coord)

# %% [markdown]
# We plot the features distribution after the normalization:

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=normalized_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# ## Computing the PCA decomposition

# %% [markdown]
# We calculate the principal component decomposition of the indicators chosen for clustering:

# %%
pca = PCA()
X_pca = pca.fit_transform(normalized_indicators)
pca_df = pd.DataFrame(index=incidents_df.index)

# %% [markdown]
# We visualize the distribution of the features in the space defined by the first two principal components:

# %%
nrows=4
ncols=5
row=0
fig, axs = mplt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), sharex=True, sharey=True)
for i, col in enumerate(normalized_indicators.columns):
    if i != 0 and i % ncols == 0:
        row += 1
    axs[row][i % ncols].scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40, c=normalized_indicators[col], cmap='viridis')
    axs[row][i % ncols].set_title(col)
    axs[row][i % ncols].set_xlabel("1st eigenvector")
    axs[row][i % ncols].set_ylabel("2nd eigenvector")

# %% [markdown]
# We observe that:
# - the first PC is correlated with 'n_injured_prop' and 'n_arrested_prop'
# - the second PC is correlated with 'n_killed_prop'

# %% [markdown]
# We visualize the distribution of the features in the space defined by the third and fourth principal components:

# %%
nrows=4
ncols=5
row=0
fig, axs = mplt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), sharex=True, sharey=True)
for i, col in enumerate(normalized_indicators.columns):
    if i != 0 and i % ncols == 0:
        row += 1
    axs[row][i % ncols].scatter(X_pca[:, 2], X_pca[:, 3], edgecolor='k', s=40, c=normalized_indicators[col], cmap='viridis')
    axs[row][i % ncols].set_title(col)
    axs[row][i % ncols].set_xlabel("3rd eigenvector")
    axs[row][i % ncols].set_ylabel("4th eigenvector")

# %% [markdown]
# We observe that:
# - the third PC is correlated with 'n_unharmed_prop' and slightly with 'n_arrested_prop', 'n_killed_prop' and 'n_injured_prop'
# - the fourth PC is correlated with 'n_teen_prop' and slightly with 'avg_age'

# %% [markdown]
# We display incidents in vector space of the first 3 PC:

# %%
x = X_pca[:, 0]
y = X_pca[:, 2]
z = X_pca[:, 1]
fig = px.scatter_3d(x=x, y=y, z=z, labels={'x': '1st eigenvector', 'y': '3rd eigenvector', 'z': '2nd eigenvector'})
fig.show()

# %% [markdown]
# To narrow the number of PC down to the most relevant ones we plot the explained variance of each component, and relate it to the previous one:

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
    
print(xtick)
xtick.append(xtick[-1]+1.5)

plt.bar(xtick, exp_var_pca, align='center')
plt.plot(xtick[1:], diff_var, label='difference from prevoius variance', color='orange')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component')
plt.title('Explained variance by principal component')
plt.xticks(xtick, range(exp_var_pca.shape[0]))
plt.legend();

# %% [markdown]
# The most significant PC are the third, fourth and fifth

# %%
def get_reconstruction_error(x_pca, x_orig, pca, n_comp):
    dummy = np.matmul(x_pca[:,:n_comp], pca.components_[:n_comp,:]) + pca.mean_
    return pd.DataFrame(index=x_orig.index, data=np.sum((dummy - x_orig.values)**2, axis=1))

# %%
pca_col = [
'1st_comp',
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
 '17th_comp'
]
pca_indicators = pd.DataFrame(index=normalized_indicators.index, data=X_pca, columns=pca_col)

# %% [markdown]
# We calculate the dataset reconstruction error for the first 8, 10 and 12 components:

# %%
pca_indicators['PCA_rec_error_8C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 8)
pca_indicators['PCA_rec_error_10C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 10)
pca_indicators['PCA_rec_error_12C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 12)

# %% [markdown]
# We display the distribution of the principal components and of the reconstruction error:

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=pca_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# We normalize the principal components to apply the clustering algorithm on them:

# %%
pca_normalized_indicators = pd.DataFrame(data=scaler_obj.fit_transform(pca_indicators.values), columns=pca_indicators.columns)

# %% [markdown]
# We display the normalized distributions:

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=pca_normalized_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# We will use the first 10 components, striking a balance between minimizing the number of components and mitigating the reconstruction error.
# Indeed, as shown below, the reconstruction error generated by these 10 components is quite uniform and close to zero:

# %%
hist_box_plot(
    pca_normalized_indicators,
    'PCA_rec_error_10C',
    title='PCA_rec_error_10C',
    bins=int(np.log(pca_normalized_indicators.shape[0])), # Sturger's rule
    figsize=(10, 5)
)

# %%
clustered_components = pca_col[:-7]
X = pca_normalized_indicators[clustered_components].values

# %% [markdown]
# Below we define the parameters of the k-means algorithm:
# - 300 iterations should be enough to converge (it is the default parameter of the scikit-learn implementation)
# - the algorithm is run 10 times with different initializations and the best result in terms of SSE is chosen (10 runs is the default parameter of the scikit-learn implementation)
# - initial centroids are sampled based on an empirical probability distribution of the points’ contribution to the overall inertia (this method is called k-means++ and again it is the default parameter of the scikit-learn implementation)
# - the maximum number of K to later be evaluated is 30 (higher values lead to results that are difficult to interpret)
# - we fixed the random seed to make the results reproducible

# %%
MAX_ITER = 300
N_INIT = 10
INIT_METHOD = 'k-means++'
MAX_K = 30
RANDOM_STATE = 7

# %% [markdown]
# ## Identification of the best value of k

# %% [markdown]
# The following function uses the implementation of the elbow method from the library [Yellowbrick](https://www.scikit-yb.org/en/latest/index.html), to identify the best value of k. This method consists in computing a metric to evaluate the quality of the clustering for each value of k, and then plotting the metric as a function of k. The best value of k is the one that corresponds to the point of inflection of the curve (the point where the metric starts to decrease more slowly).

# %%
def apply_k_elbow_method(X, kmeans_params, metric, start_k, max_k, plot_elbow=True):
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

    for i in range(len(max_k)):
        kmeans_params['n_clusters'] = i
        kmeans = KMeans(**kmeans_params)

        elbow_vis = KElbowVisualizer(kmeans, k=(start_k, max_k[i]), metric=metric, timings=False, ax=axs[i], locate_elbow=plot_elbow)
        elbow_vis.fit(X)
        axs[i].set_title(f'{metric_descr} elbow for K-Means clustering (K in [{str(start_k)}, {str(max_k[i] - 1)}])')
        axs[i].set_ylabel(metric_descr)
        axs[i].set_xlabel('K')
        if plot_elbow:
            axs[i].legend([
                f'{metric_descr}',
                f'elbow at K = {str(elbow_vis.elbow_value_)}, {metric_descr} = {elbow_vis.elbow_score_:0.2f}'
            ])
        else:
            axs[i].legend([
                f'{metric_descr}'
            ])

        if len(max_k)==1:
            axs[1].remove()
    
    plt.show()

# %% [markdown]
# Now we apply the elbow method evaluating the SSE and computing the elbow for each value of k between 1 and 10, 1 and 20 and 1 and 30. We consider also k=1 (absence of clusters) and we repeat the analysis with k in different ranges because the inflection point of the curve could vary depending on the number of points the curve is made of.

# %%
kmeans_params = {'init': INIT_METHOD, 'n_init': N_INIT, 'max_iter': MAX_ITER, 'random_state': RANDOM_STATE}
apply_k_elbow_method(X=X, kmeans_params=kmeans_params, metric='SSE', start_k=1, max_k=[10, 20, 30])

# %% [markdown]
# The elbow method identifies k=4 and k=6 as best possible values of k. For both values of k the elbow is evident.

# %%
best_k = [4, 6]

# %% [markdown]
# We repeat the analysis using the silhouette score.

# %%
apply_k_elbow_method(X=X, kmeans_params=kmeans_params, metric='silhouette', start_k=2, max_k=[15], plot_elbow=False)

# %% [markdown]
# The curve generated above is not monotonic. Indeed there are several local maxima (for k equals to 4, 6, 8, 12 and 14).
# %%
best_k += [4, 8, 12, 14]

# %% [markdown]
# We repeat again the analysing evaluating the Calinski-Harabasz score (the ratio of the sum of between-cluster dispersion and of within-cluster dispersion; the higher it is the better the clustering).

# %%
apply_k_elbow_method(X=X, kmeans_params=kmeans_params, metric='calinski_harabasz', start_k=2, max_k=[15], plot_elbow=False)

# %% [markdown]
# There is a single global maximum at k=4.

# %%
best_k += [4]
best_k = sorted(list(set(best_k)))
best_k

# %% [markdown]
# To identify the best value of k we also apply the X-means algorithm from the library [pyclustering](https://github.com/annoviko/pyclustering/), which is a variant of the k-means algorithm that should automatically find the best value of k. The algorithm starts with k=2 and then it iteratively splits the clusters until a score does not improve anymore. The implementation we will use supports both the BIC score (Bayesian Information Criterion) and the Minimum Noiseless Descriptionlength score (MDL):

# %%
initial_centers = kmeans_plusplus_initializer(data=X, amount_centers=1, random_state=RANDOM_STATE).initialize()
xmeans_MDL_instance = xmeans(
    data=X,
    initial_centers=initial_centers,
    kmax=MAX_K,
    splitting_type=splitting_type.BAYESIAN_INFORMATION_CRITERION,
    random_state=RANDOM_STATE
)
xmeans_MDL_instance.process()
n_xmeans_BIC_clusters = len(xmeans_MDL_instance.get_clusters())
print(f'Number of clusters found by xmeans using BIC score and setting the maximum number of clusters to {MAX_K}: {n_xmeans_BIC_clusters}')

# %%
xmeans_MDL_instance = xmeans(
    data=X,
    initial_centers=initial_centers,
    kmax=MAX_K,
    splitting_type=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH,
    random_state=RANDOM_STATE
)
xmeans_MDL_instance.process()
n_xmeans_MDL_clusters = len(xmeans_MDL_instance.get_clusters())
print(f'Number of clusters found by xmeans using MDL score and setting the maximum number of clusters to {MAX_K}: {n_xmeans_MDL_clusters}')

# %% [markdown]
# X-means terminates with k equal to the maximum number of clusters allowed (30 in our case). This means that the score always improved when splitting the clusters.

# %% [markdown]
# To choose the best value of k among the ones identified by the elbow method, we will compute other metrics to evaluate the quality of the clustering. The following function fits the k-means algorithm with a given set of parameters and computes the following metrics:
# - SSE
# - BSS (i.e. between-cluster sum of squares; the higher the better)
# - Davies-Bouldin score (i.e. the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances; the lower the better)
# - Calinski-Harabasz score
# - Silhouette score

# %%
def fit_kmeans(X, params):
    print(f"Fitting KMeans with k = {params['n_clusters']}")
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

# %% [markdown]
# To study the effect of the centroids initialization on the results, we will apply the algorithm using the k-means++ initialization repeated 10 times (as previously done)

# %%
results = {}
kmeans_params = {}
kmeans_params['random_state'] = RANDOM_STATE
kmeans_params['max_iter'] = MAX_ITER
best_k = sorted(best_k)
for k in best_k:
    kmeans_params['n_init'] = N_INIT
    kmeans_params['n_clusters'] = k
    kmeans_params['init'] = INIT_METHOD
    result = fit_kmeans(X=X, params=kmeans_params)
    results[str(k)+'_means'] = result

# %%
results_df = pd.DataFrame(results).T
results_df.drop(columns=['model'])

# %% [markdown]
# We observe that:
# - SSE and BSS are best for k=14, but these metric are expected to improve while increasing the number of clusters
# - Davies-Bouldin score is best for k=4
# - Calinski-Harabasz score is best for k=4 by a large margin
# - Silhouette score is best for k=4

# %%
best_k = [4, 14]
best_k

# %% [markdown]
# We visualize the size of the clusters with the best values of k:

# %%
fig, axs = plt.subplots(nrows=1, ncols=len(best_k), figsize=(25,5))
for i in range(len(best_k)):
    k = best_k[i]
    plot_clusters_size(clusters=results[f'{k}_means']['model'].labels_, ax=axs[i], title=f'{best_k[i]}-Means clusters size', color_palette=sns.color_palette('tab10'))

# %% [markdown]
# We visualize the silhouette score for each point with the best values of k:

# %%
fig, axs = plt.subplots(nrows=1, ncols=len(best_k), figsize=(30,10), sharey=True)
for i in range(len(best_k)):
    clusters = results[f'{best_k[i]}_means']['model'].labels_
    silhouette_per_point = silhouette_samples(X=X, labels=clusters)
    results[f'{best_k[i]}_means']['silhouette_per_point'] = silhouette_per_point
    plot_scores_per_point(
        score_per_point=silhouette_per_point,
        clusters=clusters,
        score_name='Silhouette score', ax=axs[i],
        title=f'Silhouette score for {best_k[i]}-Means clustering',
        color_palette=sns.color_palette('tab20'),
        minx=-0.18
    )

# %% [markdown]
# For both values of k the resulting clusters are well-balanced. The difference between the number of points in the largest and in the smallest cluster is similar in the two solutions.
#
# For k=4, we observe fewer data points with negative silhouettes. While, for k=14, more points have a negative silhouette score, especially those in the smaller-sized clusters.
#
# We will use k=4 because a smaller number of clusters makes their understanding simpler and more effective, and according to Occam's razor, the simplest solution is the best one.

# %% [markdown]
# ## Characterization of the clusters

# %% [markdown]
# We initialize the centroids with the final centroids computed by BisectingKMeans.

# %%
chosen_k = 4
bisect_kmeans = BisectingKMeans(n_clusters=chosen_k, n_init=5, init=INIT_METHOD, random_state=RANDOM_STATE).fit(X)
kmeans_params = {}
kmeans_params['random_state'] = RANDOM_STATE
kmeans_params['max_iter'] = MAX_ITER
kmeans_params['n_clusters'] = chosen_k
kmeans_params['n_init'] = 1
kmeans_params['init'] = bisect_kmeans.cluster_centers_
final_result = fit_kmeans(X=X, params=kmeans_params)

kmeans = final_result['model']
clusters = kmeans.labels_
incidents_df['cluster'] = clusters
centroids = kmeans.cluster_centers_

# %%
pd.DataFrame(final_result, index=['k=4']).drop(columns=['model'])

# %% [markdown]
# We visualize the centroids with a parallel coordinates plot:

# %%
for j in range(0, len(centroids)):
    plt.plot(centroids[j], marker='o', label='Cluster %s' % j, c=sns.color_palette('tab10')[j])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(clustered_components)), clustered_components, rotation=90)
plt.legend(fontsize=10)
plt.title(f'Centroids of {chosen_k}-means clusters');

# %% [markdown]
# We observe that, as expected, the first components are the ones that most characterize the centroids of the clusters. Diversity decreases as moving through the principal components.
#
# We remind that the 3 principal components were correlated to the following features:
# - 'n_arrested_prop'
# - 'n_killed_prop'
# - 'n_injured_prop'
# - 'n_unharmed_prop'
# - 'n_teen_prop'
#
# For this reason, we expect that the clusters primarily differ in these 4 features.

# %% [markdown]
# We visualize the same information using a interactive radar plot:

# %%
def plot_spider(points, features, title=None, palette=sns.color_palette(), save=False):
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
    if save:
        pyo.plot(fig, filename=f'../html/centroids_spider_PCA.html', auto_open=False)

# %%
plot_spider(points=centroids, features=clustered_components, title=f'Centroids of {chosen_k}-means clusters', palette=sns.color_palette('tab10'))

# %% [markdown]
# We define a function to convert back the centroid in the original feature space:

# %%
def inverse_trasform_k_comp(x_pca, pca, n_comp):
    return np.matmul(x_pca[:,:n_comp], pca.components_[:n_comp,:]) + pca.mean_

# %% [markdown]
# We plot again the centroids in the original feature space:

# %%
transformed_centroids = inverse_trasform_k_comp(centroids, pca, 10)

# %%
plot_spider(points=transformed_centroids, features=features_to_cluster_no_coord, title=f'Centroids of {chosen_k}-means clusters', palette=sns.color_palette('tab10'), save=True)

# %%
for j in range(0, len(centroids)):
    plt.plot(transformed_centroids[j], marker='o', label='Cluster %s' % j, c=sns.color_palette('tab10')[j])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster_no_coord)), features_to_cluster_no_coord, rotation=90)
plt.legend(fontsize=10)
plt.title(f'Centroids of {k}-means clusters');

# %% [markdown]
# We observe that the negative values of the attribute n_arrested_prop (that should always be positive) is due to the reconstruction error.
#
# Clusters differ also in the following features:
#
# - 'avg_age'
# - 'suprisal_n_males'
# - 'suprisal_characteristics'
#
# Cluster 0 groups incidents with an high number of unharmed people.
# Cluster 1 groups incidents with an high number of killed people.
# Cluster 2 groups incidents with an high number of injured people (and a slightly higher-than-average value of number of teens).
# Cluster 3 groups incidents with an high number of arrested people and a low number of unharmed people, killed people and injured people.
#

# %% [markdown]
# We also visualize the values of the principal components in the original feature space:
#
#

# %%
plot_spider(points=pca.components_[:10], features=features_to_cluster_no_coord, title=f'Centroids of {chosen_k}-means clusters', palette=sns.color_palette('tab10'))

for j in range(0, len(pca.components_[:10])):
    plt.plot(pca.components_[j], marker='o', label='Cluster %s' % j, c=sns.color_palette('tab10')[j])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster_no_coord)), features_to_cluster_no_coord, rotation=90)
plt.title(f'Centroids of {chosen_k}-means clusters');

# %%
most_identifying_columns = [
    'avg_age',
    'n_killed_prop',
    'n_injured_prop',
    'n_unharmed_prop',
    'n_males_prop',
    'surprisal_n_males',
    'n_arrested_prop',
    'surprisal_day'
]

# %% [markdown]
# ## Distribution of variables within the 4 clusters

# %% [markdown]
# We plot on a map the points in each cluster:

# %%
plot_scattermap_plotly(
    incidents_df,
    'cluster',
    zoom=2.5,
    height=600,
    title=f'Point divided by cluster',
    black_nan=False
).show()

# %% [markdown]
# Incidents are not clustered according to their geographical location. All clusters are evenly distributed, even in areas with fewer incidents like Hawaii or Alaska.

# %% [markdown]
# Now we inspect the distribution of categorical features within the clusters:

# %%
plot_bars_by_cluster(df=incidents_df, feature='year', cluster_column='cluster')

# %% [markdown]
# For clusters 1 and 2, the distribution of the year of the incident is in line with that of the entire dataset.
# Cluster 3 groups fewer incidents happened in 2014, while cluster 0 groups more incidents happened in 2014.

# %%
incidents_df.loc[incidents_df['year']==2014.0].describe()[features_to_cluster_no_coord]

# %%
incidents_df.loc[incidents_df['year']>2014.0].describe()[features_to_cluster_no_coord]

# %% [markdown]
# | year | n_participants mean | n_arrested mean/ratio | n_unharmed mean/ratio |
# | :- | -: | -: | -: |
# | =2014 | 1.965 | 0.066 / 0.033 | 0.387 / 0.196 |
# | >2014 | 1.774 | 0.384 / 0.216 | 0.102 / 0.001 |

# %% [markdown]
# In 2014, the average number of arrests is 0.06 and the proportion between the average number of arrestees and participants is 0.03, whereas in subsequent years the two values are higher (0.38 and 0.21 respectively).

# %%
plot_bars_by_cluster(df=incidents_df, feature='party', cluster_column='cluster')

# %% [markdown]
# In cluster 3 - characterized by the highest value of n_arrestes_prop - the proportion of incidents happened in Republican states is higher than those happened in Democratic states. This is probably due to variations in the law enforcement policies.

# %%
plot_bars_by_cluster(df=incidents_df, feature='firearm', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['firearm']==False)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df


# %% [markdown]
# Almost all incidents that did not involve firearms belong the cluster with the highest number of unharmed.

# %%
plot_bars_by_cluster(df=incidents_df, feature='shots', cluster_column='cluster')

# %% [markdown]
# The vast majority of incidents within clusters 1 and 2 are shooting incidents and are the incidents that mostly caused deaths and injuries. Cluster 3 groups less shooting incidents.

# %%
plot_bars_by_cluster(df=incidents_df, feature='aggression', cluster_column='cluster')

# %% [markdown]
# As expected, cluster 3 mainly comprises non-aggressive incidents, while cluster 2, almost entirely comprises aggressive incidents. The distribution of aggressions in cluster 0 is similar to the distribution in the whole dataset. Cluster 1, identified with the highest number of killed people, exhibits the largest difference between the number of aggressive and non-aggression incidents.

# %%
incidents_df.groupby('aggression').describe()[['n_killed_prop', 'n_injured_prop', 'n_arrested_prop']]

# %% [markdown]
# The average number of deaths increases to 0.415 for aggressive incidents and drops to 0.065 for non aggressive incidents.

# %%
plot_bars_by_cluster(df=incidents_df, feature='suicide', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['suicide']==True)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df

# %% [markdown]
# Suicides are almost entirely contained within cluster 1.

# %%
plot_bars_by_cluster(df=incidents_df, feature='injuries', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['injuries']==False)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df

# %% [markdown]
# Cluster 2 mainly groups incidents involving injuries. Cluster 1 presents a lower number of incidents involving injuries. Half of the incidents not involving injuries are in cluster 3, the other half are evenly distributed between clusters 0 and 1.

# %%
plot_bars_by_cluster(df=incidents_df, feature='death', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['death']==False)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df

# %% [markdown]
# Cluster 1 mainly groups mortal incidents. Some mortal incidents are also in cluster 2.

# %%
plot_bars_by_cluster(df=incidents_df, feature='illegal_holding', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['illegal_holding']==True)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df

# %% [markdown]
# In clusters 1 and 2, few incidents are tagged as 'illegal_holding'. Incidents exhibiting this tag are mainly grouped in clusters 0 and 3 (most of them did not result in injuries or deaths).

# %%
plot_bars_by_cluster(df=incidents_df, feature='house', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['house']==True)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df

# %%
plot_bars_by_cluster(df=incidents_df, feature='school', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['school']==True)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df

# %% [markdown]
# We observe that incidents that occurred in schools have, on average, resulted in many arrest. The majority of them involve few participants and a peaceful resolution of the event.


# %%
plot_bars_by_cluster(df=incidents_df, feature='drugs', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['drugs']==True)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df

# %% [markdown]
# Cluster 3 groups most of the incidents involving drugs. This means that these incidents often have peaceful resolutions and frequently involve arrests.

# %%
plot_bars_by_cluster(df=incidents_df, feature='officers', cluster_column='cluster')

# %% [markdown]
# Incidents involving officers exhibit higher numbers of arrests.

# %%
plot_bars_by_cluster(df=incidents_df, feature='workplace', cluster_column='cluster')

# %% [markdown]
# Incidents happened at workplace are mainly grouped in clusters 0 and 3. Generally, these incidents conclude peacefully and often result in arrests.

# %%
plot_bars_by_cluster(df=incidents_df, feature='defensive', cluster_column='cluster')

# %%
dummy_df = incidents_df[
    (incidents_df['defensive']==True)]['cluster'].value_counts().to_frame()
dummy_df['percentage'] = dummy_df['cluster']/sum(dummy_df['cluster'])
dummy_df

# %% [markdown]
# The majority of incidents identified as 'defensive' are included in cluster 0 (the cluster with higher numbers of unharmed).

# %% [markdown]
# We visualize the identified clusters projected onto the most influential dimensions in the clustering, i.e.:
# - 'n_teen_prop'
# - 'surprisal_age_groups'
# - 'n_killed_prop'
# - 'n_injured_prop'
# - 'n_unharmed_prop'
# - 'n_males_prop'
# - 'surprisal_n_males'
# - 'surprisal_characteristics'
# - 'n_arrested_prop'
# - 'surprisal_day'

# %%
scatter_by_cluster(
    df=incidents_df,
    features=most_identifying_columns,
    cluster_column='cluster',
    centroids=transformed_centroids[:, [most_identifying_columns.index(feature) for feature in most_identifying_columns]],
    figsize=(15, 20),
    ncols=3,
    color_palette=sns.color_palette('tab10')
)
plt.tight_layout()

# %% [markdown]
# We visualize the clusters in the space of the first 6 components of PCA. As expected the first principal components better separate the 4 clusters.

# %%
palette = [sns.color_palette('tab10')[i] for i in range(chosen_k)]
scatter_pca_features_by_cluster(
    X_pca=X_pca,
    n_components=6,
    clusters=incidents_df['cluster'],
    palette=palette,
    hue_order=None,
    title='Clusters in PCA space'
)

# %% [markdown]
# We plot the distributions of the features used for clustering.

# %%
for feature in features_to_cluster:
    plot_hists_by_cluster(
        df=incidents_df,
        feature=feature,
        cluster_column='cluster',
        title=f'Distribution of {feature} in each cluster',
        color_palette=sns.color_palette('tab10'),
        figsize=(30, 5)
    )

# %% [markdown]
# These plots confirm the observations made so far. We observe that in cluster 2, the value of 'suprisal_charactestic' is lower than in the other clusters.

# %% [markdown]
# ## Evaluation of the clustering results
# ### Internal indices

# %% [markdown]
# We compute the sum of squared error for each separate feature:

# %%
sse_feature = []
for i in range(X.shape[1]):
    sse_feature.append(compute_se_per_point(X=X[:,i], clusters=clusters, centroids=transformed_centroids[:,i]).sum())

# %%
plt.figure(figsize=(15, 5))
sse_feature_sorted, clustering_features_sorted = zip(*sorted(zip(sse_feature, clustered_components)))
plt.bar(range(len(sse_feature_sorted)), sse_feature_sorted)
plt.xticks(range(len(sse_feature_sorted)), clustering_features_sorted)
plt.xticks(rotation=90)
plt.ylabel('SSE')
plt.xlabel('Feature')
plt.title('SSE per feature')

# %% [markdown]
# We compute and plot the silhouette score for each point:

# %%
fig, axs = plt.subplots(1, figsize=(8,5))
silhouette_per_point = silhouette_samples(X=X, labels=clusters)
plot_scores_per_point(
    score_per_point=silhouette_per_point,
    clusters=incidents_df['cluster'],
    score_name='Silhouette score', ax=axs,
    title=f'Silhouette score for {k}-Means clustering',
    color_palette=sns.color_palette('tab10'),
    minx=-0.02
)

# %% [markdown]
# As already observed, few points have a negative silhouette score. Cluster 0 has lower silhouette scores than the other clusters.

# %%
# print top 5 points with highest SSE
se_per_point = compute_se_per_point(X=X, clusters=clusters, centroids=centroids)
indices_of_top_contributors = np.argsort(se_per_point)[-5:]
incidents_df.iloc[indices_of_top_contributors]

# %% [markdown]
# All these points have an high number of participants.

# %% [markdown]
# The number of participants contributes a lot to the SSE.
#
# To evaluate the separation we also display an embedding of the cluster centers in 2 dimensions, using the implementation of [Yellowbrick](https://www.scikit-yb.org/en/latest/index.html):

# %%
visualizer = InterclusterDistance(kmeans)
visualizer.fit(X)
visualizer.show();

# %% [markdown]
# Clusters are well separated.
#
# We now compute cohesion (SSE) and separation (BSS) for each cluster and visualize it:

# %%
# compute cohesion for each cluster
se_per_cluster = np.zeros(chosen_k)
sizes = np.ones(centroids.shape[0])
for i in range(chosen_k):
    se_per_cluster[i] = np.sum(se_per_point[np.where(clusters == i)[0]])/sizes[i]
# compute separation for each cluster
bss_per_cluster = compute_bss_per_cluster(X, clusters, centroids, weighted=True)
# compute average silhouette score for each cluster
silhouette_per_cluster = np.zeros(chosen_k)
for i in range(chosen_k):
    silhouette_per_cluster[i] = silhouette_per_point[np.where(clusters == i)[0]].mean()

# visualize the result
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
axs[0].bar(range(chosen_k), se_per_cluster, color=sns.color_palette('tab10'))
axs[0].set_ylim(8000, 0)
axs[0].set_title('Cohesion')
axs[0].set_ylabel('SSE')
axs[1].bar(range(chosen_k), bss_per_cluster, color=sns.color_palette('tab10'))
axs[1].set_title('Separation')
axs[1].set_ylabel('BSS')
axs[2].bar(range(chosen_k), silhouette_per_cluster, color=sns.color_palette('tab10'))
axs[2].set_title('Silhouette')
axs[2].set_ylabel('Silhouette score')

for i in range(3):
    axs[i].set_xlabel('Cluster')
    axs[i].set_xticks(range(chosen_k))
    axs[i].set_xticklabels(range(chosen_k))

plt.suptitle('Cohesion and separation measures for each cluster', fontweight='bold')

# %% [markdown]
# Cluster 1 is the less cohesive, cluster 2 is the best separated.

# %% [markdown]
# We visualize the distance matrix sorted by cluster computed on a stratified subsample of 5000 points:

# %%
dm, idm = plot_distance_matrices(X=X, n_samples=5000, clusters=clusters, random_state=RANDOM_STATE)

# %% [markdown]
# The pearson correlation coefficient between the two matrix is 0.64. Indeed, the matrix has a sharp block diagonal structure, meaning that clusters are well separated.
#
# ### External indices
#
# We measure the extent to which the discovered clustering structure matches some categorical features of the dataset, using the following permutation invariant scores:
# - **Adjusted rand score**: this score computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings. It is 0 for random labeling, 1.0 when the clusterings are identical and is bounded below by -0.5 for especially discordant clusterings.
# - **Normalized mutual information**: is a normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation). Mutual Information is a function that measures the agreement of the two assignments, ignoring permutations.
# - **Homogeneity**: measure the degree to which each cluster contains only members of a single class; it ranges between 0 and 1, with 1 denoting perfectly homogeneous labeling.
# - **Completeness**: measure the degree to ewhich data points that are members of a given class are also elements of the same cluster; it ranges between 0 and 1, with 1 denoting perfectly complete labeling.

# %%
incidents_df['unharmed'] = incidents_df['n_unharmed'] > 0
incidents_df['arrested'] = incidents_df['n_arrested'] > 0
external_score = compute_permutation_invariant_external_metrics(
    incidents_df,
    'cluster',
    ['shots', 'aggression', 'suicide', 'injuries', 'death', 'drugs', 'illegal_holding', 'unharmed', 'arrested']
)
external_score

# %% [markdown]
# The categories that best matche the clustering are 'unharmed' and 'arrested'. The most homogeneous category is 'unharmed'. However, completeness is quite low for all the categories.

# %%
incidents_df['cluster'].to_csv(f'../data/clustering_labels/{chosen_k}-Means_PCA_clusters.csv')
external_score.to_csv(f'../data/clustering_labels/{chosen_k}-Means_PCA_external_scores.csv')
pd.DataFrame(final_result, index=['4means PCA']).T.to_csv(f'../data/clustering_labels/{chosen_k}-Means_PCA_internal_scores.csv')


