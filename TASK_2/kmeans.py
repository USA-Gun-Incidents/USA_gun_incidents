# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import os
import sys
sys.path.append(os.path.abspath('..'))
from plot_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

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
numeric_features = [col for col in incidents_df.columns if 'ratio' in col]
numeric_features += [
    'latitude',
    'longitude',
    'min_age_participants',
    'avg_age_participants',
    'max_age_participants',
    'location_importance',
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
incidents_df = incidents_df.dropna()
print(f"Number of features after dropping rows with nan {incidents_df.shape[0]}")

# %%
incidents_df.replace([np.inf, -np.inf], 0, inplace=True)

# %%
plt.figure(figsize=(15, 12))
corr_matrix = incidents_df[numeric_features].corr() # TODO: provare altri coef di corr?
sns.heatmap(corr_matrix, mask=np.triu(corr_matrix))

# %%
# highlight attributes with high correlation
threshold = 0.5
plt.figure(figsize=(15, 12))
masked_corr_matrix = corr_matrix[((corr_matrix >= threshold)| (corr_matrix <= -threshold)) & (corr_matrix != 1.000)]
sns.heatmap(masked_corr_matrix, mask=np.triu(masked_corr_matrix), vmin=corr_matrix.values.min(), vmax=corr_matrix.values.max())
plt.show()

# %%
std_scaler = StandardScaler() # TODO: fare pipelines?
std_scaler.fit(incidents_df[numeric_features].values)

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(incidents_df[numeric_features].values)

X_std = std_scaler.fit_transform(incidents_df[numeric_features].values)
X_minmax = minmax_scaler.fit_transform(incidents_df[numeric_features].values)

# %%
kmeans_std = KMeans(n_clusters=10, n_init=10, max_iter=100)
kmeans_std.fit(X_std)

kmeans_minmax = KMeans(n_clusters=10, n_init=10, max_iter=100)
kmeans_minmax.fit(X_minmax)

# %%
hist, bins = np.histogram(kmeans_std.labels_, bins=range(0, len(set(kmeans_std.labels_)) + 1))
cluster_size_std = dict(zip(bins, hist))
hist, bins = np.histogram(kmeans_minmax.labels_, bins=range(0, len(set(kmeans_minmax.labels_)) + 1))
cluster_size_minmax = dict(zip(bins, hist))
print("Cluster size std")
print(cluster_size_std)
print("Cluster size minmax")
print(cluster_size_minmax)

# %%
centroids = std_scaler.inverse_transform(kmeans_std.cluster_centers_)
centroids_std = kmeans_std.cluster_centers_
incidents_df['kmeans_std_labels'] = kmeans_std.labels_

# %%
plt.scatter(
    incidents_df['n_males_n_males_tot_year_city_ratio'],
    incidents_df['location_importance'],
    c=kmeans_std.labels_,
    s=20,
    cmap='tab10');
# draw white circle on centroids
plt.scatter(
    centroids[:, incidents_df[numeric_features].columns.get_loc('n_males_n_males_tot_year_city_ratio')],
    centroids[:, incidents_df[numeric_features].columns.get_loc('location_importance')],
    marker='o',
    c="white",
    alpha=1,
    s=200,
    edgecolor='k'
)
# write cluster number on white circles
for i, centroid in enumerate(centroids):
    plt.scatter(
        centroids[i, incidents_df[numeric_features].columns.get_loc('n_males_n_males_tot_year_city_ratio')],
        centroids[i, incidents_df[numeric_features].columns.get_loc('location_importance')],
        s=50,
        marker='$%d$' % i,
        alpha=1,
        edgecolor='k'
    );
# TODO: scatter matrix of a subset colored by cluster?

# %%
plot_scattermap_plotly(incidents_df, 'kmeans_std_labels', zoom=2, title='Kmeans clustering (with standardized data)')

# %%
plt.figure(figsize=(8, 4))
for i in range(0, len(centroids)):
    plt.plot(centroids[i], marker='o', label='Cluster %s' % i)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(incidents_df[numeric_features].columns)), incidents_df[numeric_features].columns, rotation=90)
plt.legend(fontsize=10)
plt.title('Centroids (original features)')
plt.show()

# %%
plt.figure(figsize=(8, 4))
for i in range(0, len(centroids_std)):
    plt.plot(centroids_std[i], marker='o', label='Cluster %s' % i)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xticks(range(0, len(incidents_df[numeric_features].columns)), incidents_df[numeric_features].columns, rotation=90)
plt.legend(fontsize=10)
plt.title('Centroids (standardized features)')
plt.show()

# %%
df = pd.DataFrame()
for i, center in enumerate(centroids_std):
    tmp_df = pd.DataFrame(dict(r=center, theta=numeric_features))
    tmp_df['Centroid'] = f'Centroid {i}'
    df = pd.concat([df,tmp_df], axis=0)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, color='Centroid')
fig.show()

# %%
for i, center in enumerate(centroids_std):
    tmp_df = pd.DataFrame(dict(r=center, theta=numeric_features))
    tmp_df['Centroid'] = f'Centroid {i}'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        fig = px.line_polar(tmp_df, r='r', theta='theta', line_close=True)
        fig.update_layout(
            title_text=f'Centroid {i}'
        )
        fig.update_traces(line_color=px.colors.qualitative.Plotly[i])
    fig.show()

# %%
print('SSE %s' % kmeans_std.inertia_)
# print('Silhouette %s' % silhouette_score(X_std, kmeans_std.labels_))
# print('Separation %s' % metrics.davies_bouldin_score(X_std, kmeans_std.labels_))

# %%
party_xt = pd.crosstab(kmeans_std.labels_, incidents_df['party'])
party_xt.plot(
    kind='bar',
    stacked=False, 
    title='Party per cluster',
    figsize=(15, 7)
    )
plt.xlabel('Cluster')
plt.ylabel('Party')
plt.show()

# %%
# TODO: provare su subsample
# from scipy.spatial.distance import pdist, squareform
# distance_matrix = squareform(pdist(X_std, metric='euclidean'))

# sorted_indices = np.argsort(kmeans_std.labels_)
# sorted_distance_matrix = distance_matrix[sorted_indices][:, sorted_indices]

# plt.figure(figsize=(8, 8))
# plt.imshow(sorted_distance_matrix)
# plt.colorbar()
# plt.title("Similarity Matrix Sorted by K-means Clusters")
# plt.show()

# %%
sse_list = list()
max_k = 40
for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100)
    kmeans.fit(X_std)
    
    sse = kmeans.inertia_
    sse_list.append(sse)

# %%
plt.plot(range(2, len(sse_list) + 2), sse_list)
plt.ylabel('SSE')
plt.xlabel('K')
plt.tick_params(axis='both', which='major')
plt.show()



# %%
kmeans = KMeans(n_init=10, max_iter=100)
max_k = [10, 20, 30]
best_k = []

f, axs = plt.subplots(nrows=1, ncols=len(max_k), figsize=(30,5))

for i in range(len(max_k)):
    vis = KElbowVisualizer(kmeans, k=(2, max_k[i]), metric='distortion', timings=False, ax=axs[i])
    vis.fit(X_std)  
    axs[i].set_title(f'SSE elbow for K-Means clustering (K = [{str(2)}, {str(max_k[i])}])')
    axs[i].set_ylabel('SSE')
    axs[i].set_xlabel('K')
    axs[i].legend([
        'SSE for K',
        f'elbow at K = {str(vis.elbow_value_)}, SSE = {vis.elbow_score_:0.2f}'
    ])
    if vis.elbow_value_ != None and vis.elbow_value_ not in best_k:
        best_k.append(vis.elbow_value_)
plt.show()
