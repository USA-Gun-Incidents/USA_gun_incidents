# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
import plotly.express as px
import warnings
np.warnings = warnings # altrimenti numpy da problemi con pyclustering, TODO: è un problema solo mio?
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans, BisectingKMeans
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import utm
import os
import sys
sys.path.append(os.path.abspath('..'))
from plot_utils import *
# %matplotlib inline

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=0,
    parse_dates=['date'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

# %%
def compute_ratio_indicator(df, ext_df, gby, num, den, suffix, agg_fun):
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    df[num+'_'+den+suffix+'_ratio'] = np.divide(df[num], df[den+suffix], out=np.zeros_like(df[num]), where=(df[den+suffix] != 0))
    df.drop(columns=[den+suffix], inplace=True)
    return df

# %%
incidents_df['city'] = incidents_df['city'].fillna('UKN')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_tot_year_city', 'sum') # 1
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_tot_year_congdist', 'sum') # 2
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_injured', 'n_injured', '_tot_year_congdist', 'sum') # 2
incidents_df['n_killed_n_participants_ratio'] = incidents_df['n_killed'] / incidents_df['n_participants'] # 3
incidents_df['n_injured_n_participants_ratio'] = incidents_df['n_injured'] / incidents_df['n_participants'] # 3
incidents_df['n_unharmed_n_participants_ratio'] = incidents_df['n_unharmed'] / incidents_df['n_participants'] # 3
incidents_df['n_arrested_n_participants_ratio'] = incidents_df['n_arrested'] / incidents_df['n_participants']
incidents_df['n_females_n_males_ratio'] = incidents_df['n_females'] / incidents_df['n_males']
incidents_df['n_child_n_participants_ratio'] = incidents_df['n_participants_child'] / incidents_df['n_participants']
incidents_df['n_teen_n_participants_ratio'] = incidents_df['n_participants_teen'] / incidents_df['n_participants'] 
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year'], 'n_unharmed', 'n_unharmed', '_mean_year', 'mean') # 4

# %%
ratio_features = [col for col in incidents_df.columns if 'ratio' in col]
numeric_features = ratio_features + [
    'latitude',
    'longitude',
    'min_age_participants',
    'avg_age_participants',
    'max_age_participants'
]
categorical_features = [
    'state',
    'year',
    'month',
    'day',
    'day_of_week',
    'party'
]

# %%
print(f"Number of features before dropping rows with nan {incidents_df.shape[0]}")
incidents_df = incidents_df.dropna(subset=numeric_features)
print(f"Number of features after dropping rows with nan {incidents_df.shape[0]}")

# %%
incidents_df.replace([np.inf, -np.inf], 0, inplace=True)

# %%
coord = utm.from_latlon(incidents_df['latitude'].to_numpy(), incidents_df['longitude'].to_numpy())
incidents_df['x'] = coord[0]
incidents_df['y'] = coord[1]
numeric_features += ['x', 'y']
numeric_features.remove('latitude')
numeric_features.remove('longitude')

# %%
incidents_df[ratio_features].boxplot()
plt.xticks(rotation=90);

# %%
features_to_log = [
    'n_males_n_males_tot_year_city_ratio',
    'n_killed_n_killed_tot_year_congdist_ratio',
    'n_injured_n_injured_tot_year_congdist_ratio',
    'n_unharmed_n_unharmed_mean_year_ratio'
]
numeric_transformed_features = [feature for feature in numeric_features if feature not in features_to_log] + ['log_'+feature for feature in features_to_log]


# %%
def logtransform(df, features, perc):
    for col in features:
        eps = (df[df[col]!=0][col].min()*perc)/100
        df['log_'+col] = np.log(incidents_df[col] + eps)
    return df


# %%
incidents_df = logtransform(incidents_df, features_to_log, 1) # TODO: provare altre costanti

# %%
incidents_df[['log_'+feature for feature in features_to_log]].boxplot()
plt.xticks(rotation=90);

# %%
scaler_names = ['Standard scaler', 'MinMax scaler', 'Robust scaler']
scaler_obj = [StandardScaler(), MinMaxScaler(), RobustScaler()]
X_scaled = []
X_scaled_transf = []
df_scaled = []
for scaler in scaler_obj:
    X_scaled.append(scaler.fit_transform(incidents_df[numeric_features].values))
    X_scaled_transf.append(scaler.fit_transform(incidents_df[numeric_transformed_features].values))
    df_scaled.append(pd.DataFrame(X_scaled[-1], columns=numeric_features))

# %%
# plot distributions with different scalers
fig, axs = plt.subplots(len(scaler_names)+1, 1, figsize=(15, 30), sharex=True)
incidents_df[numeric_features].boxplot(ax=axs[0])
axs[0].set_title('Unscaled')
for name, df in zip(scaler_names, df_scaled):
    df.boxplot(ax=axs[scaler_names.index(name)+1])
    axs[scaler_names.index(name)+1].set_title(name)
plt.xticks(rotation=90);


# %%
# # plot UTM coordinates with different scalers
# fig, axs = plt.subplots(len(scaler_names)+1, 1, figsize=(15, 30))
# axs[0].scatter(incidents_df['x'], incidents_df['y'])
# for name, df in zip(scaler_names, df_scaled):
#     axs[scaler_names.index(name)+1].scatter(df['x'], df['y'])
#     axs[scaler_names.index(name)+1].set_title(name)

# %%
def get_sse_varying_k(X, kmeans_params, max_k):
    sse_list = []
    for k in range(2, max_k + 1):
        print(f"Clustering with K={k}...")
        kmeans_params['n_clusters'] = k
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(X)
        sse_list.append(kmeans.inertia_)
    return sse_list


# %%
kmeans_params = {'n_init': 10, 'max_iter': 100}
max_k = 10

# %%
# compute SSE for different values of k
sse_lists = {}
for name, X, X_transf in zip(scaler_names, X_scaled, X_scaled_transf):
    print(f"Computing SSE for different values of k with {name}...")
    sse_lists[name] = get_sse_varying_k(X, kmeans_params, max_k)
    print(f"Computing SSE for different values of k with {name} and transformed features...")
    sse_lists[name+' transformed'] = get_sse_varying_k(X_transf, kmeans_params, max_k)

# %%
# plot SSE at different k with different scalers
for label, sse_feature in sse_lists.items():
    plt.plot(range(2, len(sse_feature) + 2), sse_feature, label=label)
plt.ylabel('SSE')
plt.xlabel('K')
plt.tick_params(axis='both', which='major')
plt.legend()
plt.show()

# %%
plt.plot(range(2, len(sse_lists['MinMax scaler']) + 2), sse_lists['MinMax scaler'], label='MinMax scaler')
plt.plot(range(2, len(sse_lists['MinMax scaler transformed']) + 2), sse_lists['MinMax scaler transformed'], label='MinMax scaler transformed')
plt.ylabel('SSE')
plt.xlabel('K')
plt.tick_params(axis='both', which='major')
plt.legend()
plt.show()

# %%
X = X_scaled[scaler_names.index('MinMax scaler')]
scaler = scaler_obj[scaler_names.index('MinMax scaler')]


# %%
def plot_k_elbow(kmeans, metric, start_k, max_k): # TODO: plottare nello stesso plot curva che parte da 1 e da 2 se sono diverse
    if metric == 'distortion':
        metric_descr = 'SSE'
    elif metric == 'calinski_harabasz':
        metric_descr = 'Calinski Harabasz Score'
    elif metric == 'silhouette':
        metric_descr = 'Silhouette Score'
    else:
        raise ValueError('Metric not supported')
    
    _, axs = plt.subplots(nrows=1, ncols=len(max_k), figsize=(30,5))

    for i in range(len(max_k)):
        elbow_vis = KElbowVisualizer(kmeans, k=(start_k, max_k[i]), metric=metric, timings=False, ax=axs[i])
        elbow_vis.fit(X)
        axs[i].set_title(f'{metric_descr} elbow for K-Means clustering (K = [{str(start_k)}, {str(max_k[i])}])')
        axs[i].set_ylabel(metric_descr)
        axs[i].set_xlabel('K')
        axs[i].legend([
            f'{metric_descr} for K',
            f'elbow at K = {str(elbow_vis.elbow_value_)}, {metric_descr} = {elbow_vis.elbow_score_:0.2f}'
        ])
    
    plt.show()
    return elbow_vis.elbow_value_


# %%
kmeans = KMeans(**kmeans_params)
max_k = [10, 20, 30] # + di 30 è difficile da interpretare
best_k = []

# %%
k = plot_k_elbow(kmeans, 'distortion', 1, max_k)
if k != None and k not in best_k:
    best_k.append(k)

# %%
# k = plot_k_elbow(kmeans, 'distortion', 2, max_k)
# if k != None and k not in best_k:
#     best_k.append(k)

# %%
# k = plot_k_elbow(kmeans, 'calinski_harabasz', 1, max_k)
# if k != None and k not in best_k:
#     best_k.append(k)

# %%
# k = plot_k_elbow(kmeans, 'calinski_harabasz', 2, max_k)
# if k != None and k not in best_k:
#     best_k.append(k)

# %%
# k = plot_elbow(kmeans, 'silhouette', 1, max_k)
# if k != None and k not in best_k:
#     best_k.append(k)

# %%
# k = plot_elbow(kmeans, 'silhouette', 2, max_k)
# if k != None and k not in best_k:
#     best_k.append(k)

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
best_k = [4, 7]


# %%
def bss(X, labels, centroids):
    centroid = X.mean(axis=0)
    sizes = np.bincount(labels)
    return np.sum(np.sum(np.square((centroids - centroid)), axis=1)*sizes)

def sse_per_point(X, labels, centroids):
    return np.sum(np.square((X - centroids[labels])), axis=(1 if X.ndim > 1 else 0))

def fit_kmeans(model, params, X):
    kmeans = model(**params)
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
for k in best_k:
    params = {'n_init': 10, 'max_iter': 100, 'n_clusters': k}
    result = fit_kmeans(KMeans, params, X)
    results[str(k)+'means'] = result

    bisect_kmeans = BisectingKMeans(n_clusters=k, n_init=5).fit(X) # TODO: salvare i risultati anche di questo?
    params = {'max_iter': 100, 'init': bisect_kmeans.cluster_centers_, 'n_init': 1, 'n_clusters': k}
    result = fit_kmeans(KMeans, params, X)
    results[str(k)+'means_bis_init'] = result

# %%
results_df = pd.DataFrame(results).T
results_df.drop(columns=['model']) # only for printing

# %%
k = 4
kmeans = results['4means']['model']
labels = results['4means']['model'].labels_
centroids = results['4means']['model'].cluster_centers_
centroids_inverse = scaler.inverse_transform(centroids)

# %%
plt.figure(figsize=(8, 4))
for i in range(0, len(centroids_inverse)):
    plt.plot(centroids_inverse[i], marker='o', label='Cluster %s' % i)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(incidents_df[numeric_features].columns)), incidents_df[numeric_features].columns, rotation=90)
plt.legend(fontsize=10)
plt.title('Centroids (original features)')
plt.show()

# %%
plt.figure(figsize=(8, 4))
for i in range(0, len(centroids)):
    plt.plot(centroids[i], marker='o', label='Cluster %s' % i)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(incidents_df[numeric_features].columns)), incidents_df[numeric_features].columns, rotation=90) # TODO: fare anche questo interattivo?
plt.legend(fontsize=10)
plt.title('Centroids (scaled features)')
plt.show()

# %%
df = pd.DataFrame()
for i, center in enumerate(centroids):
    tmp_df = pd.DataFrame(dict(r=center, theta=numeric_features))
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
sse_feature, numeric_features_tuple = zip(*sorted(zip(sse_feature, numeric_features)))
plt.bar(range(len(sse_feature)), sse_feature)
plt.xticks(range(len(sse_feature)), numeric_features_tuple)
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
incidents_df['cluster'] = labels
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
for attribute in ['day_of_week', 'month', 'party', 'year']:
    plot_distribution_categorical_attribute(incidents_df, attribute) # TODO: usare subplots?

# %%
features_to_scatter = numeric_features[0:5]
ncols = 3
nplots = len(features_to_scatter)*(len(features_to_scatter)-1)/2
nrows = int(nplots / ncols)
if nplots % ncols != 0:
    nrows += 1

colors = [sns.color_palette()[c] for c in incidents_df['cluster']]
#colors = [cmaps["tab10"].colors[c] for c in incidents_df['cluster']]
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
                marker='o', c="white", alpha=1, s=200, edgecolor='k')
            axs[int(id/ncols)][id%ncols].scatter(
                centroids[c][incidents_df[features_to_scatter].columns.get_loc(x)],
                centroids[c][incidents_df[features_to_scatter].columns.get_loc(y)],
                marker='$%d$' % c, alpha=1, s=50, edgecolor='k')
        axs[int(id/ncols)][id%ncols].set_xlabel(x)
        axs[int(id/ncols)][id%ncols].set_ylabel(y)
        id += 1
for ax in axs[nrows-1, id%ncols:]:
    ax.remove()
plt.suptitle(("Clusters in different feature spaces"), fontsize=20)
plt.show()

# %%
silhouette_vis = SilhouetteVisualizer(kmeans, title='Silhouette plot', colors='tab10')
silhouette_vis.fit(X)
silhouette_vis.show()
# TODO: colorare negli scatter i punti con silhouette negativa?

# %%
subsampled_incidents_df = incidents_df.groupby('cluster', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=0))
subsampled_incidents_df.reset_index(inplace=True)
subsampled_incidents_df.sort_values(by=['cluster'], inplace=True)

# %%
from scipy.spatial.distance import pdist, squareform
dm = squareform(pdist(X[subsampled_incidents_df.index]))

# %%
n_subsampled_points = subsampled_incidents_df.shape[0]
im = np.zeros((n_subsampled_points, n_subsampled_points))
clusters = subsampled_incidents_df['cluster'].to_numpy() # per accesso più veloce sotto
for i in range(n_subsampled_points):
    for j in range(n_subsampled_points):
        if clusters[i] == clusters[j]:
            im[i, j] = 1

# %%
corr_matrix = np.corrcoef(dm, im)

# %%
plt.matshow(corr_matrix) 
plt.show()

# %%
ncols = 3
nplots = len(numeric_features)
nrows = int(nplots/ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36,36))
id = 0
for feature in numeric_features:
    incidents_df.boxplot(column=feature, by='cluster', ax=axs[int(id/ncols)][id%ncols])
    id += 1
for ax in axs[nrows-1, id%ncols:]:
    ax.remove()

# %%
attr = 'n_males_n_males_tot_year_city_ratio'
axes = incidents_df[attr].hist(by=incidents_df['cluster'], bins=20, layout=(1,k), figsize=(20, 5))
plt.suptitle(f'Distribution of {attr} in each cluster', fontweight='bold')
for i, ax in enumerate(axes):
    ax.set_title(f'Cluster {i}')
    ax.set_xlabel(attr)
    ax.set_ylabel('Number of incidents')
# TODO: usare subplots?

# %%
# TODO:
# - prendere punti random e calcolare score

# %%
# KMEDOID CON ATTRIBUTI CATEGORICI
# from pyclustering.cluster.kmedoids import kmedoids
# import gower

# X_cat = np.asarray(incidents_df)
# distance_matrix = gower.gower_matrix(X_cat) # calcolo di distance matrix su tutto il dataset non è fattibile, possiamo però definire noi la distanza e passarla in modo che venga calcolato solo quando necessario
# initial_medoids = kmeans_plusplus_initializer(X, k, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize(return_index=True)
# kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
# kmedoids_instance.process()
# clusters = kmedoids_instance.get_clusters()
# medoids = kmedoids_instance.get_medoids()
