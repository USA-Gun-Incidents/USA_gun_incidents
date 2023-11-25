# -*- coding: utf-8 -*-
# %% [markdown]
# # KMeans clustering

# %% [markdown]
# We import the libraries:

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import KMeans, BisectingKMeans
from scipy.spatial.distance import pdist, squareform
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
import json
import os
import sys
sys.path.append(os.path.abspath('..'))
from plot_utils import *
from clustering_utils import *
# %matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %% [markdown]
# We load the data and prepare it for the clustering:

# %%
# load the data
incidents_df = pd.read_csv('../data/incidents_indicators.csv', index_col=0)
# load the names of the features to use for clustering
f = open('../data/indicators_names.json')
features_to_cluster = json.loads(f.read())
# FIXME: da fare in indicators
features_to_cluster = [feature for feature in features_to_cluster if feature not in ['lat_proj', 'lon_proj']] # 'n_killed_prop', 'n_injured_prop', 'n_unharmed_prop'
# FIXME: da spostare più avanti
categorical_features = [
    'year', 'month', 'day_of_week', 'party', #'state', 'address_type', 'county', 'city'
    'firearm', 'air_gun', 'shots', 'aggression', 'suicide',
    'injuries', 'death', 'road', 'illegal_holding', 'house',
    'school', 'children', 'drugs', 'officers', 'organized', 'social_reasons',
    'defensive', 'workplace', 'abduction', 'unintentional'
    # 'incident_characteristics1', 'incident_characteristics2'
    ]
# FIXME: poverty_perc, date
# drop nan
incidents_df = incidents_df.dropna(subset=features_to_cluster)
# initialize a colum for the clustering labels
incidents_df['cluster'] = None
# project on the indicators
indicators_df = incidents_df[features_to_cluster]

# %% [markdown]
# We apply MinMaxScaler and StandardScaler to the data, visualizing the results of the transformation:

# %%
# apply MinMaxScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(indicators_df.values)
# apply StandardScaler
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(indicators_df.values)
# plot the distributions of the indicators after the transformations
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 6))
axs[0].boxplot(indicators_df.values)
axs[0].set_xticklabels(features_to_cluster, rotation=90);
axs[0].set_title('Original data');
axs[1].boxplot(X_minmax)
axs[1].set_xticklabels(features_to_cluster, rotation=90);
axs[1].set_title('Min-Max scaling');
axs[2].boxplot(X_std)
axs[2].set_xticklabels(features_to_cluster, rotation=90);
axs[2].set_title('Standard scaling');

# %% [markdown]
# Variables have different scales and variances.
#
# Since K-means tends to produce globular clusters, leaving variances unequal is like giving different weights to features based on their variance. To avoid this, we will use StandardScaler.

# %%
X = X_std

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
RANDOM_STATE = 42

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
        axs[i].set_title(f'{metric_descr} elbow for K-Means clustering (K in [{str(start_k)}, {str(max_k[i])}])')
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
# The elbow method identifies k=4, k=7 and k=10 as best possible values of k. At k=10 the point of inflection is more evident.

# %%
best_k = [4, 7, 10]

# %% [markdown]
# We repeat the analysis using the silhouette score.

# %%
apply_k_elbow_method(X=X, kmeans_params=kmeans_params, metric='silhouette', start_k=2, max_k=[12], plot_elbow=False)

# %% [markdown]
# The silhouette score has a global maximum for k=2; k=8 and k=10 are local maximum with very close silhouette scores.

# %%
best_k += [2, 8]

# %% [markdown]
# We repeat again the analysing evaluating the Calinski-Harabasz score (the ratio of the sum of between-cluster dispersion and of within-cluster dispersion; the higher it is the better the clustering).

# %%
apply_k_elbow_method(X=X, kmeans_params=kmeans_params, metric='calinski_harabasz', start_k=2, max_k=[12], plot_elbow=False)

# %% [markdown]
# The Calinski-Harabasz score is maximum for k=2. It also has two local maxima for k=7.

# %% [markdown]
# To identify the best value of k we also apply the X-means algorithm from the library [pyclustering](https://github.com/annoviko/pyclustering/), which is a variation of the k-means algorithm that should automatically find the best value of k. The algorithm starts with k=2 and then it iteratively splits the clusters until a score does not improve anymore. The implementation we will use supports both the BIC score (Bayesian Information Criterion) and the Minimum Noiseless Descriptionlength score (MDL):

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
# X-means terminates with k equal to the maximum number of clusters allowed (30 in our case). This means that the score always improved when splitting the clusters. No value of k is optimal according to these criteria.

# %% [markdown]
# To choose the best value of k among the ones identified by the elbow method, we will compute other metrics to evaluate the quality of the clustering. The following function fits the k-means algorithm with a given set of parameters and computes the following metrics:
# - SSE
# - BSS (i.e. between-cluster sum of squares; the higher the better)
# - Davies-Bouldin score (i.e. the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances; the lower the better)
# - Calinski-Harabasz score
# - Silhouette score

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

# %% [markdown]
# To study the effect of the centroids initialization on the results, we will apply the algorithm using the k-means++ initialization repeated 10 times (as previously done), but we will also initialize the centroids with the final centroids computed by BisectingKMeans. 

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
    results[str(k)+'means'] = result

    # FIXME: fare dopo una volta scelto il migliore
    # bisect_kmeans = BisectingKMeans(n_clusters=k, n_init=5, init=INIT_METHOD, random_state=RANDOM_STATE).fit(X)
    # kmeans_params['n_init'] = 1
    # kmeans_params['init'] = bisect_kmeans.cluster_centers_
    # result = fit_kmeans(X=X, params=kmeans_params)
    # results[str(k)+'means_bis_init'] = result

# %%
results_df = pd.DataFrame(results).T
results_df.drop(columns=['model'])

# %% [markdown]
# We observe that:
# - SSE is best for k=10, but this metric is expected to decrease increasing the number of clusters
# - BSS is best for k=10
# - Davies-Bouldin score is best for k=10
# - Calinski-Harabasz score is best for k=2
# - Silhouette score is best for k=2

# %%
best_k = [2, 10]

# %%
fig, axs = plt.subplots(nrows=1, ncols=len(best_k), figsize=(25,5))
for i in range(len(best_k)):
    k = best_k[i]
    plot_clusters_size(clusters=results[f'{k}means']['model'].labels_, ax=axs[i], title=f'{best_k[i]}-Means clusters size', color_palette=sns.color_palette('tab10'))

# %%
fig, axs = plt.subplots(nrows=1, ncols=len(best_k), figsize=(30,10), sharey=True)
for i in range(len(best_k)):
    clusters = results[f'{best_k[i]}means']['model'].labels_
    silhouette_per_point = silhouette_samples(X=X, labels=clusters)
    results[f'{best_k[i]}means']['silhouette_per_point'] = silhouette_per_point
    plot_scores_per_point(
        score_per_point=silhouette_per_point,
        clusters=clusters,
        score_name='Silhouette score', ax=axs[i],
        title=f'Silhouette score for {best_k[i]}-Means clustering',
        color_palette=sns.color_palette('tab10'),
        minx=-0.18
    )

# %% [markdown]
# When using k=2, approximately half of the points in cluster 0 have a negative silhouette score. This means that the average distance to other points in cluster 0 is greater than the average distance to points in cluster 1. When using k=10, there are still points with negative silhouette score, but some clusters have points with higher silhouette scores. Since it is not clear which value of k is better, we will characterize the clusters obtained with k=2 and k=10.

# %% [markdown]
# ## Characterization of the clusters

# %% [markdown]
# ### Analysis of the centroids with k=2

# %%
k = 2
kmeans = results[f'{k}means']['model']
clusters = results[f'{k}means']['model'].labels_
centroids = results[f'{k}means']['model'].cluster_centers_

# %% [markdown]
# We visualize the centroids with a parallel coordinates plot:

# %%
for j in range(0, len(centroids)):
    plt.plot(centroids[j], marker='o', label='Cluster %s' % j, c=sns.color_palette('tab10')[j])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster)), features_to_cluster, rotation=90)
plt.legend(fontsize=10)
plt.title(f'Centroids of {k}-means clusters');

# %% [markdown]
# We observe that some feature do not vary much between the two clusters, while others do. 
#
# Cluster 0 groups incidents involving higher number of participants, including children, teens and females. These incidents seem less severe (the centroid has a lower value of killed people and an higher value of unharmed people).
# Cluster 1, the largest cluster, probably groups incidents with the most common characteristics in the dataset, i.e. those involving few adult males.

# %% [markdown]
# ### Analysis of the centroids with K=10

# %%
k = 10
kmeans = results[f'{k}means']['model']
clusters = results[f'{k}means']['model'].labels_
centroids = results[f'{k}means']['model'].cluster_centers_
silhouette_per_point = results[f'{k}means']['silhouette_per_point']

# %% [markdown]
# We visualize the centroids with a parallel coordinates plot:

# %%
for j in range(0, len(centroids)):
    plt.plot(centroids[j], marker='o', label='Cluster %s' % j, c=sns.color_palette('tab10')[j])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster)), features_to_cluster, rotation=90)
plt.legend(fontsize=10)
plt.title(f'Centroids of {k}-means clusters');

# %% [markdown]
# We observe that:
# - cluster 0 has the highest values of location_imp and surprisal_address_type
# - cluster 1 has the lowest value of n_males_prop
# - cluster 2 has the highest value of n_arrestes_prop
# - cluster 3 has the highest value of n_teen_prop
# - cluster 4 has the highest value of n_killed_prop (and the lowest of n_injured_prop)
# - cluster 5 has the highest value of n_unharmed_prop
# - cluster 6 has the highest value of age_range
# - cluster 7 has the highest value of n_injured_prop and the lowest of surprisal_characteristics
# - cluster 8 has the highest value of n_child_prop, surprisal_age_groups and surprisal_characteristics; and the lowest of avg_age
# - cluster 9 has the highest value of n_participants (and one of the highest surprisal_n_males)
#
# Surprisal day does not vay much across the clusters.
#
# We decide to analyze in more detail these 10 clusters to identify more specific patterns.

# %% [markdown]
# We visualize the same information using a interactive radar plot:

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
plot_spider(points=centroids, features=features_to_cluster, title=f'Centroids of {k}-means clusters', palette=sns.color_palette('tab10'))

# %% [markdown]
# ## Distribution of variables within the 10 clusters

# %%
incidents_df['cluster'] = clusters

# %% [markdown]
# We plot on a map the points in each cluster:

# %%
for i in range(k):
    plot_scattermap_plotly(
        incidents_df[incidents_df['cluster']==i],
        'cluster',
        zoom=2.5,
        height=400,
        title=f'Cluster {i}',
        color_sequence=sns.color_palette('tab10').as_hex()[i:],
        black_nan=False,
        showlegend=False
    )

# %% [markdown]
# Incidents are not clustered according to their geographical location.
# Points in cluster 0 - the one whose centroid has high location importance - are distributed similarly to points in other clusters. This is probably due to the fact that location importance is correlated with the population density, and we expect that in areas with higher densisty also the number of incidents is higher as well as the etereogenity of their characteristics.

# %% [markdown]
# Now we inspect the distribution of categorical features within the clusters:

# %%
plot_bars_by_cluster(df=incidents_df, feature='year', cluster_column='cluster')

# %% [markdown]
# Cluster 2 - the one with the highest value of n_arrestes_prop - contains fewer incidents happened in 2014, that year less people where arrested
# Cluster 5 - the one with highest value of n_unharmed_prop - contains more incidents happend in 2014, that year incidents where less severe (and probably the smaller number of arrested is a consequence of this)

# %%
plot_bars_by_cluster(df=incidents_df, feature='day_of_week', cluster_column='cluster')


# %% [markdown]
# Cluster 2 - the one with the highest value of n_arrestes_prop - has a different distribution compared to the one in the whole dataset, with a pick on Wednesday and Thursday instead of in the weekend, as in the whole dataset. This deviation may be attributed to police shift pattern

# %%
plot_bars_by_cluster(df=incidents_df, feature='party', cluster_column='cluster')

# %% [markdown]
# In cluster 2 - the one with the highest value of n_arrestes_prop - the number of incidents happened in Republican states is higher than those happened in Democratic states. This is probably due to variations in the law enforcement policies.
# In cluster 7 - the one with the highest value of n_injured_prop and the lowest of surprisal_characteristics - the proportion of incidents happend in Democratic states is higher. This behaviour is not easily explainable.

# %%
plot_bars_by_cluster(df=incidents_df, feature='firearm', cluster_column='cluster')

# %% [markdown]
# Cluster 5 - the one with the highest value of n_unharmed_prop - has the highest number of incidents that did not involved firearms.

# %%
incidents_df[incidents_df['firearm']==False][['incident_characteristics1', 'incident_characteristics2', 'cluster']]

# %%
plot_bars_by_cluster(df=incidents_df, feature='shots', cluster_column='cluster')

# %% [markdown]
# In cluster 4 and cluster 7 - those with highest value of n_killed_prop and n_injured_prop - the majority of incidents were shooting incidents (as expected).
#
# In cluster 2 - the one with the highest value of n_arrestes_prop - the number of shooting or non shooting incidents are approximately the same. Maybe guns were found during commission of other crimes, we check this:

# %%
incidents_df[
    (incidents_df['incident_characteristics1']=='Possession (gun(s) found during commission of other crimes)') |
    (incidents_df['incident_characteristics1']=='Possession (gun(s) found during commission of other crimes)')
]['cluster'].value_counts().reset_index()

# %% [markdown]
# As expected most of the incidents in cluster 5 exhibit that characteristic.

# %%
plot_bars_by_cluster(df=incidents_df, feature='aggression', cluster_column='cluster')

# %% [markdown]
# As expected in cluster 2 the number of non aggresive incidents is higher, while in cluster 7 - the one with highest value of n_injured_prop - is it lower.
# In cluster 4 - the one with highest value of n_killed_prop - the majority of incidents is non aggressive. This could make sense if those incidents are suicides, let's check:

# %%
incidents_df[
    incidents_df['suicide']==True
]['cluster'].value_counts().reset_index()

# %% [markdown]
# The majority of incidents were actually suicides.

# %%
plot_bars_by_cluster(df=incidents_df, feature='suicide', cluster_column='cluster')

# %% [markdown]
# Again, most of suicidal incidents are in cluster 4. But this cluster also contains incidents that are not suicidal.

# %%
plot_bars_by_cluster(df=incidents_df, feature='injuries', cluster_column='cluster')

# %% [markdown]
# Most of incidents in cluster 2 - the one with the highest value of n_arrestes_prop - have no injuries. This is in line with the previous observations.
# Also cluster 4 - the one with highest value of n_killed_prop and the lowest of n_injured_prop - has a high number of incidents with no injuries (if people died are not considered injured).
# Incidens in cluster 7 - the one with the highest value of n_injured_prop - have, as expected, a high number of injuries.
# Incidents in cluster 3, 8 and 9 - the one with the highest value of n_teen_prop, n_child_prop and n_participants respectively - have an higher proportion of injuries.

# %%
plot_bars_by_cluster(df=incidents_df, feature='death', cluster_column='cluster')

# %% [markdown]
# Cluster 4 and cluster 7 - the one with highest value of n_killed_prop and n_injured_prop respectively - have the highest and lowest number of mortal incidents.

# %%
plot_bars_by_cluster(df=incidents_df, feature='illegal_holding', cluster_column='cluster')

# %% [markdown]
# Cluster 4 and cluster 7 - the one with highest value of n_killed_prop and n_injured_prop respectively - have fewer incidents in which participants were illegally armed.

# %%
plot_bars_by_cluster(df=incidents_df, feature='house', cluster_column='cluster')

# %% [markdown]
# All the observations made so far are in line with the characteristics of these distributions.

# %%
plot_bars_by_cluster(df=incidents_df, feature='children', cluster_column='cluster')

# %% [markdown]
# Most of the incidents involving children are in cluster 8 - the one having the highest value of n_child_prop. Some of them are in cluster 6 - the one with the highest value of age_range.

# %%
plot_bars_by_cluster(df=incidents_df, feature='drugs', cluster_column='cluster')

# %% [markdown]
# Cluster 2 - the one with the highest value of n_arrestes_prop - has an higher proportion of incidents involving drugs.

# %%
plot_bars_by_cluster(df=incidents_df, feature='officers', cluster_column='cluster')

# %% [markdown]
# The distributions of incidents involving officers in the clusters are similar to the one in the whole dataset.

# %%
plot_bars_by_cluster(df=incidents_df, feature='defensive', cluster_column='cluster')

# %% [markdown]
# Most of the incidents with defensive use of firearms are in cluster 5 - the one with the highest value of n_unharmed_prop. Guns were probably used as a means of threat.

# %%
plot_bars_by_cluster(df=incidents_df, feature='unintentional', cluster_column='cluster')

# %% [markdown]
# Cluster 3 and 8, the ones with the highest value of n_teen_prop and n_child_prop respectively, have the highest number of accidental incidents.

# %%
# TODO: organized, social reasons, workplace, abduction
# no much to say

# %% [markdown]
# We visualize the clusters in the each pair of dimensions:

# %%
# fare stesso per povertà

# %%
scatter_by_cluster(
    df=incidents_df,
    features=features_to_cluster,
    cluster_column='cluster',
    centroids=centroids,
    figsize=(15, 70),
    ncols=3,
    color_palette=sns.color_palette('tab10')
)
plt.tight_layout()

# %% [markdown]
# We can recognize different clusters when points are scattered in the plane defined by surprisal_n_males and surprisal_age_groups.

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
palette = [sns.color_palette('tab10')[i] for i in range(k)]
scatter_pca_features_by_cluster(
    X_pca=X_pca,
    n_components=4,
    clusters=clusters,
    palette=palette,
    hue_order=None,
    title='Clusters in PCA space'
)

# %% [markdown]
# In the features spaces obtained by pairing the first 4 principal components, the clusters are not well separated.

# %%
plot_boxes_by_cluster(
    df=incidents_df,
    features=features_to_cluster,
    cluster_column='cluster',
    figsize=(15, 35),
    title='Box plots of features by cluster'
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
        title=f'Distribution of {feature} in each cluster',
        color_palette=sns.color_palette('tab10'),
        figsize=(30, 5)
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
# print top 5 points with highest SSE
se_per_point = compute_se_per_point(X=X, clusters=clusters, centroids=centroids)
indices_of_top_contributors = np.argsort(se_per_point)[-5:]
incidents_df.iloc[indices_of_top_contributors]

# %% [markdown]
# All these points have an high number of participants.

# %%
fig, axs = plt.subplots(1)
plot_scores_per_point(score_per_point=se_per_point, clusters=clusters, score_name='SE', ax=axs, color_palette=sns.color_palette('tab10'), minx=100)

# %% [markdown]
# The number of participants contributes a lot to the SSE.

# %%
clusters_silh = np.full(clusters.shape[0], -1)
for i, s in enumerate(silhouette_per_point):
    if s >= 0:
        clusters_silh[i] = clusters[i]

palette=([sns.color_palette('tab10')[i] for i in range(k)]+[(0,0,0)])
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

# %% [markdown]
# Clusters 2, 4, 5, 7 and 0 are highly overlapped, while clusters 8 is well separated from the others.

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
axs[0].bar(range(k), se_per_cluster, color=sns.color_palette('tab10'))
axs[0].set_ylim(200000, 0)
axs[0].set_title('Cohesion') # TODO: non è proprio cohesion
axs[0].set_ylabel('SSE')
axs[1].bar(range(k), bss_per_cluster, color=sns.color_palette('tab10'))
axs[1].set_title('Separation')
axs[1].set_ylabel('BSS')
axs[2].bar(range(k), silhouette_per_cluster, color=sns.color_palette('tab10'))
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
    score_name='Silhouette score',
    cmaps=['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'YlOrBr', 'PuRd', 'Greys', 'Wistia', 'GnBu'],
    figsize=(40, 6)
)

# %%
dm, idm = plot_distance_matrices(X=X, n_samples=5000, clusters=clusters, random_state=RANDOM_STATE)

# %%
write_clusters_to_csv(clusters, f'./{k}means_clusters.csv')

# %%
compute_permutation_invariant_external_metrics(incidents_df, 'cluster', categorical_features)

# %%
compute_external_metrics(incidents_df, 'cluster', categorical_features)


