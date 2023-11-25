# -*- coding: utf-8 -*-
# %% [markdown]
# # KMeans clustering of Principal Components

# %% [markdown]
# we import the libraries

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
# and we load all the data usefoul for the analysis

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
# we select the features we want to use for clustering

# %%
features_to_cluster_no_coord = features_to_cluster[2:]
features_to_cluster_no_coord

# %% [markdown]
# we display the distribution of the selected features 

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=incidents_df[features_to_cluster_no_coord],ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# in order to obtain meaningful results, we must ensure that there is no feature that presents too high magnitude that outweighs the others, to implement this we normalise all ranges between 0 and 1

# %%
from sklearn.preprocessing import MinMaxScaler
scaler_obj = MinMaxScaler()
normalized_indicators = pd.DataFrame(data=scaler_obj.fit_transform(incidents_df[features_to_cluster_no_coord].values), columns=features_to_cluster_no_coord)

# %% [markdown]
# the features distribution after the normalization, the shape is untouched but now there are all included in the same range of values

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=normalized_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# ## Compiting the PCA decomposition

# %% [markdown]
# below we use the object PCA and the funtion fit_transform implemented in the [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) library to calculate the principal component decomposition of the indicators chosen for clustering

# %%
pca = PCA()
X_pca = pca.fit_transform(normalized_indicators)
pca_df = pd.DataFrame(index=incidents_df.index)

# %% [markdown]
# We now expose the records in the new vector space composed of the 2 most relevant eigenvectors, in relations, in relation to the value of an original feature for every original feature

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
# we observe correlation between the first and second eigenvector and some original features, in particular:
#
# - the first eighenvector is slightly correlated with 'n_injured_prop' and 'n_arrested_prop'
# - the second is strictly correlated with 'n_killed_prop'

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
# Above we have the same plot as before, but taking in to consideration the third and the fourth eigenvector, and here too we observe correlation between the two eigenvector and some original features, in particular:
#
# - the third eighenvector is strongly correlated with 'n_unharmed_prop' and slightly with 'n_arrested_prop', 'n_killed_prop' and 'n_injured_prop'
# - the fourth is strictly correlated with 'n_teen_prop' and slightly with 'avg_age'

# %% [markdown]
# we display every incident in the vector space formed by the first 3 eigenvectors

# %%
x = X_pca[:, 0]
y = X_pca[:, 2]
z = X_pca[:, 1]
fig = px.scatter_3d(x=x, y=y, z=z, labels={'x': '1st eigenvector', 'y': '3rd eigenvector', 'z': '2nd eigenvector'})
fig.show()

# %% [markdown]
# Now we want to understand the relevance of each component identified by the PCA, to narrow the clustering down to the most relevant ones, which alone can best approximate the entire dataset, to do this we plot the explained variance of each component, and relate it to the previous one

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
#xtick = [0,1,2,3,4,5.5,6.5,7.5,8.5,9.5,10.5,12,13,14,15,16,17,18,19,20]
#diff_var = list(zip(xtick, diff_var))
xtick.append(xtick[-1]+1.5)

plt.bar(xtick, exp_var_pca, align='center')
plt.plot(xtick[1:], diff_var, label='difference from prevoius variance', color='orange')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component')
plt.title('Explained variance by principal component')
plt.xticks(xtick, range(exp_var_pca.shape[0]))
plt.legend();

# %% [markdown]
# Of all the possible divisions identified, the most significant are the first three

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
 '17th_comp']

# %%
pca_indicators = pd.DataFrame(index=normalized_indicators.index, data=X_pca, columns=pca_col)

# %% [markdown]
# we calculate the dataset reconstruction error for the first 2, 5 and 8 components. 
#
# defined as the error between the original dataset and the one generated by using only k components.

# %%
pca_indicators['PCA_rec_error_2C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 2)
pca_indicators['PCA_rec_error_5C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 5)
pca_indicators['PCA_rec_error_8C'] = get_reconstruction_error(X_pca, normalized_indicators, pca, 8)

# %%
pca_indicators.sample(3)

# %% [markdown]
# we display the distribution of the principal components but also the recostruction error calculate before

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=pca_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# we normalize the previous distribution, because we will use this components for the clustering, and we want uniformity between the new features identified

# %%
pca_normalized_indicators = pd.DataFrame(data=scaler_obj.fit_transform(pca_indicators.values), columns=pca_indicators.columns)

# %% [markdown]
# we display the normalized distributions

# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=pca_normalized_indicators,ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# Ultimately, we choose to use the first 8 components for clustering, because it seems the best compromise between a small number of components but still being able to approximate the original features well.
#
# Below we can see the reconstruction error generated by these 8 components, and we can see that it is quite uniform and close to zero

# %%
hist_box_plot(
    pca_normalized_indicators,
    'PCA_rec_error_8C',
    title='PCA_rec_error_8C',
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
    '8th_comp']
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
# The elbow method identifies k=4, k=5 as best possible values of k.
#
# At both points found we can see a large decrease in the rate of decrease, and being very close to each other we can say that the features chosen are also uniform in this aspect.

# %%
best_k = [4, 5]

# %% [markdown]
# We repeat the analysis using the silhouette score.

# %%
apply_k_elbow_method(X=X, kmeans_params=kmeans_params, metric='silhouette', start_k=2, max_k=[15], plot_elbow=False)

# %% [markdown]
# The curve generated above is not as monotonic as the previous one, in fact there are 3 local minima at k equals to 4, 8 and 14, and probabli the more increase k the more the siluette score is, but having too many clusters is not positive

# %%
best_k += [4, 8, 14]

# %% [markdown]
# We repeat again the analysing evaluating the Calinski-Harabasz score (the ratio of the sum of between-cluster dispersion and of within-cluster dispersion; the higher it is the better the clustering).

# %%
apply_k_elbow_method(X=X, kmeans_params=kmeans_params, metric='calinski_harabasz', start_k=2, max_k=[15], plot_elbow=False)

# %% [markdown]
# In the graph above, the best value of k is clear, as there is a single clear global maximum with k=4

# %%
best_k += [4]
best_k = sorted(list(set(best_k)))
best_k

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
# X-means terminates with k equal to the maximum number of clusters allowed (30 in our case). This means that the score always improved when splitting the clusters. It is not possible to clearly identify a best K-value using this criteria, as the best k imposes too many clusters for these to remain relevant.

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
# - SSE is best for k=4, but this metric is expected to decrease increasing the number of clusters
# - BSS is best for k=13, but this metric is inversely proportional to the previous one
# - Davies-Bouldin score is best for k=5
# - Calinski-Harabasz score is best for k=4
# - Silhouette score is best for k=14

# %%
best_k = [4, 14]
best_k

# %%
fig, axs = plt.subplots(nrows=1, ncols=len(best_k), figsize=(25,5))
for i in range(len(best_k)):
    k = best_k[i]
    plot_clusters_size(clusters=results[f'{k}_means']['model'].labels_, ax=axs[i], title=f'{best_k[i]}-Means clusters size', color_palette=sns.color_palette('tab10'))

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
# For both the two best k values identified (4 and 14), the resulting clusters are well-balanced and evenly divide the records. The difference between the quantity of records in the largest and smallest clusters is similar between the two solutions.
#
# For k=4, we can observe fewer data points with negative silhouettes. However, for k=14, they are generally more frequent, especially within the smaller-sized clusters. In addition, the average silhouette is similar for both solutions.
#
# In conclusion, both solutions have their pros and cons and theoretically could be acceptable, but we prefer the solution with k=4 because, being equal in terms of correctness and efficacy, having a smaller number of clusters makes their understanding simpler and more effective, and according to Occam's razor, the simplest solution, when equally effective, is the best one.
#
# For this reason, future analyses will be conducted based on k=4.

# %% [markdown]
# ## Characterization of the clusters

# %% [markdown]
# Chosen k, now we initialize the centroids with the final centroids computed by BisectingKMeans and then use the same procedure used before to compute the best kmeans clusterization we can.

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
centroids = kmeans.cluster_centers_

# %% [markdown]
# We visualize the centroids with a parallel coordinates plot:

# %%
for j in range(0, len(centroids)):
    plt.plot(centroids[j], marker='o', label='Cluster %s' % j, c=sns.color_palette('tab10')[j])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(clustered_components)), clustered_components, rotation=90)
plt.legend(fontsize=10)
plt.title(f'Centroids of {k}-means clusters');


# %% [markdown]
# We observe that, as expected, the first components are the ones that most characterize the centroids of the identified clusters, with diversity decreasing as we move through the principal components used.
#
# We can now appreciate the initial choice of not using all the PCA components but only a subset of the most relevant ones, as the subsequent ones would have little influence on the result.
#
# Let's remember that the first 3 principal components derived from PCA were correlated to the original features:
# - 'n_arrested_prop'
# - 'n_killed_prop'
# - 'n_injured_prop'
# - 'n_unharmed_prop'
# So, we expect that the identified clusters primarily differ in these 4 features.

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
    pyo.plot(fig, filename=f'../html/centroids_spider_PCA.html', auto_open=False)

# %%
plot_spider(points=centroids, features=clustered_components, title=f'Centroids of {chosen_k}-means clusters', palette=sns.color_palette('tab10'))


# %%
def inverse_trasform_k_comp(x_pca, pca, n_comp):
    dummy = np.matmul(x_pca[:,:n_comp], pca.components_[:n_comp,:]) + pca.mean_
    return dummy


# %%
transformed_centroids = inverse_trasform_k_comp(centroids, pca, 8)

# %%
plot_spider(points=transformed_centroids, features=features_to_cluster_no_coord, title=f'Centroids of {chosen_k}-means clusters', palette=sns.color_palette('tab10'))

# %%
for j in range(0, len(centroids)):
    plt.plot(transformed_centroids[j], marker='o', label='Cluster %s' % j, c=sns.color_palette('tab10')[j])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(features_to_cluster_no_coord)), features_to_cluster_no_coord, rotation=90)
plt.legend(fontsize=10)
plt.title(f'Centroids of {k}-means clusters');

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
# Incidents are not clustered according to their geographical location, all clusters are evenly distributed, even in areas with fewer incidents like Hawaii or Alaska.

# %% [markdown]
# Now we inspect the distribution of categorical features within the clusters:

# %%
plot_bars_by_cluster(df=incidents_df, feature='year', cluster_column='cluster')

# %% [markdown]
# Cluster 2 - the one with the highest value of n_arrestes_prop - contains fewer incidents happened in 2014, that year less people where arrested
# Cluster 5 - the one with highest value of n_unharmed_prop - contains more incidents happend in 2014, that year incidents where less severe (and probably the smaller number of arrested is a consequence of this)

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
plot_bars_by_cluster(df=incidents_df, feature='shots', cluster_column='cluster')

# %% [markdown]
# In cluster 4 and cluster 7 - those with highest value of n_killed_prop and n_injured_prop - the majority of incidents were shooting incidents (as expected).
#
# In cluster 2 - the one with the highest value of n_arrestes_prop - the number of shooting or non shooting incidents are approximately the same. Maybe guns were found during commission of other crimes, we check this:

# %%
incidents_df[
    (incidents_df['incident_characteristics1']=='Possession (gun(s) found during commission of other crimes)') |
    (incidents_df['incident_characteristics1']=='Possession (gun(s) found during commission of other crimes)')
]['cluster'].value_counts().to_frame()

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
]['cluster'].value_counts().to_frame()

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
plot_bars_by_cluster(df=incidents_df, feature='road', cluster_column='cluster')

# %%
plot_bars_by_cluster(df=incidents_df, feature='illegal_holding', cluster_column='cluster')

# %% [markdown]
# Cluster 4 and cluster 7 - the one with highest value of n_killed_prop and n_injured_prop respectively - have fewer incidents in which participants were illegally armed.

# %%
plot_bars_by_cluster(df=incidents_df, feature='house', cluster_column='cluster')

# %% [markdown]
# All the observations made so far are in line with the characteristics of these distributions.

# %%
plot_bars_by_cluster(df=incidents_df, feature='school', cluster_column='cluster')

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
plot_bars_by_cluster(df=incidents_df, feature='workplace', cluster_column='cluster')

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
    features=features_to_cluster_no_coord,
    cluster_column='cluster',
    centroids=transformed_centroids,
    figsize=(15, 70),
    ncols=3,
    color_palette=sns.color_palette('tab10')
)
plt.tight_layout()

# %% [markdown]
# We can recognize different clusters when points are scattered in the plane defined by surprisal_n_males and surprisal_age_groups.

# %% [markdown]
# In the features spaces obtained by pairing the first 4 principal components, the clusters are not well separated.

# %%
for feature in features_to_cluster_no_coord:
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

# %%
# print top 5 points with highest SSE
se_per_point = compute_se_per_point(X=X, clusters=clusters, centroids=centroids)
indices_of_top_contributors = np.argsort(se_per_point)[-5:]
incidents_df.iloc[indices_of_top_contributors]

# %% [markdown]
# All these points have an high number of participants.

# %%
fig, axs = plt.subplots(1, 1, figsize=(8,4))
plot_scores_per_point(score_per_point=se_per_point, clusters=clusters, score_name='SE', ax=axs, color_palette=sns.color_palette('tab10'), minx=1)

# %% [markdown]
# The number of participants contributes a lot to the SSE.

# %%
visualizer = InterclusterDistance(kmeans)
visualizer.fit(X)
visualizer.show()

# %% [markdown]
# Clusters 2, 4, 5, 7 and 0 are highly overlapped, while clusters 8 is well separated from the others.

# %%
# compute cohesion for each cluster
se_per_cluster = np.zeros(chosen_k)
sizes = np.ones(centroids.shape[0])
for i in range(chosen_k):
    se_per_cluster[i] = np.sum(se_per_point[np.where(clusters == i)[0]])/sizes[i] # TODO: weigthed (or not?)
# compute separation for each cluster
bss_per_cluster = compute_bss_per_cluster(X, clusters, centroids, weighted=True) # TODO: weigthed (or not?)
# compute average silhouette score for each cluster
silhouette_per_cluster = np.zeros(chosen_k)
for i in range(chosen_k):
    silhouette_per_cluster[i] = silhouette_per_point[np.where(clusters == i)[0]].mean() # TODO: already weighted

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
axs[0].bar(range(chosen_k), se_per_cluster, color=sns.color_palette('tab10'))
axs[0].set_ylim(200000, 0)
axs[0].set_title('Cohesion') # TODO: non è proprio cohesion
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

# %%
dm, idm = plot_distance_matrices(X=X, n_samples=5000, clusters=clusters, random_state=RANDOM_STATE)

# %%
write_clusters_to_csv(clusters, f'./{k}means_clusters.csv')

# %%
compute_permutation_invariant_external_metrics(incidents_df, 'cluster', categorical_features)

# %%
compute_external_metrics(incidents_df, 'cluster', categorical_features)


