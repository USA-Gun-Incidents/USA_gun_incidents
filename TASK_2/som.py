# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utm
from sklearn.preprocessing import MinMaxScaler
from pyclustering.nnet.som import som_parameters, som, type_conn
from pyclustering.cluster.somsc import somsc
from clustering_utils import *

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv'
)
indicators_df = pd.read_csv(
    '../data/incidents_cleaned_indicators.csv', index_col=0
)
features_to_cluster = [
    #'latitude_proj', 'longitude_proj',
    'location_importance', 'city_entropy', 'address_entropy',
    'avg_age_participants', 'age_range', 'log_avg_age_mean_SD', 'avg_age_entropy',
    'n_participants', 'n_participants_child_prop', 'n_participants_teen_prop', 'n_participants_adult_entropy',
    'n_males_pr', 'log_n_males_n_males_mean_semest_congd_ratio',
    'n_killed_pr', 'n_injured_pr', 'n_arrested_pr', 'n_unharmed_pr',
    'tags_entropy'
]

indicators_df = indicators_df.dropna()
incidents_df = incidents_df.loc[indicators_df.index]

indicators_df = indicators_df.loc[incidents_df['state']=='ILLINOIS']
incidents_df = incidents_df.loc[indicators_df.index]

incidents_df.reset_index(drop=True, inplace=True)
indicators_df.reset_index(drop=True, inplace=True)

latlong_projs = utm.from_latlon(indicators_df['latitude'].to_numpy(), indicators_df['longitude'].to_numpy())
scaler= MinMaxScaler()
latlong = scaler.fit_transform(np.stack([latlong_projs[0], latlong_projs[1]]).reshape(-1, 2))
indicators_df['latitude_proj'] = latlong[:,0]
indicators_df['longitude_proj'] = latlong[:,1]

X = indicators_df[features_to_cluster].values

# %%
som_params = som_parameters()
rows = 3
cols = 3
n_clusters = rows*cols
structure = type_conn.grid_four;  # each neuron has max four neighbors
network = som(rows, cols, structure, som_params)

# %%
# TODO: provare altri parametri in som_paramaters()
# adaptation_threshold (used if autostop=True, default=0.001)
# init_learn_rate (default=0.1)
# init_radius (default = 2 if cols+rows>4 else (1.5 if cols>1 and rows>1 else 1))
# init_type (default='ditributed in line with uniform grid') # TODO: che significa? sembra che per le prime due feature le inizializza in base alla dimensione della mappa??

# %%
network.train(X, autostop=True, epochs=100000)

# %%
indicators_df['cluster'] = -1
for i in range(network._rows*network._cols):
    indicators_df.loc[network.capture_objects[i], 'cluster'] = i

# %%
fig, axs = plt.subplots(1, figsize=(6,4))
award_mtx = np.array(network._award).reshape(network._rows, network._cols)
sns.heatmap(award_mtx, annot=True, ax=axs, fmt='.0f')
axs.set_xticks([])
axs.set_yticks([])
plt.title("Number of point per cluster")

# %%
len(network.capture_objects[3])

# %%
network.show_winner_matrix() # è buggata (i numeri vanno trasposti)

# %%
network.show_distance_matrix() # euclidean distance between neighboring neurons weights

# %%
#network.show_density_matrix() # is a display of the density relationships in the data space using Pareto Density Estimation

# %%
def agg_feature_by_cluster(network, df, feature, agg_fun):
    agg_feature_per_cluster = []
    for i in range(network._rows*network._cols):
        agg_feature = df.iloc[network.capture_objects[i]][feature].agg(agg_fun)
        if agg_fun=='mode':
            agg_feature = agg_feature.values[0]
        agg_feature_per_cluster.append(agg_feature)
    return agg_feature_per_cluster

# %%
ncols = 3
nplots = len(features_to_cluster)
nrows = int(nplots/ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30,30))
for i, feature in enumerate(features_to_cluster):
    avg_feature_per_cluster = agg_feature_by_cluster(network, indicators_df, feature=feature, agg_fun='mean')
    avg_feature_mtx = np.array(avg_feature_per_cluster).reshape(network._rows, network._cols)
    sns.heatmap(avg_feature_mtx, ax=axs[int(i/ncols)][i%ncols], annot=True)
    axs[int(i/ncols)][i%ncols].set_title(feature)
    axs[int(i/ncols)][i%ncols].set_xticks([])
    axs[int(i/ncols)][i%ncols].set_yticks([])
for ax in axs[nrows-1, i%ncols+1:]:
    ax.remove()
f.suptitle("Average value of each feature per cluster", fontweight='bold');

# %%
features_to_mode = ['year', 'party', 'children', 'death', 'month']
nplots = len(features_to_mode)
ncols = 3
nrows = int(nplots / ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,4))
for i, feature in enumerate(features_to_mode):
    mode_per_cluster = agg_feature_by_cluster(network, incidents_df, feature=feature, agg_fun='mode')
    unique_values = incidents_df[feature].unique()
    unique_values_map = {unique_value: i for i, unique_value in enumerate(unique_values)}
    num_mode_per_cluster = [unique_values_map[mode] for mode in mode_per_cluster]
    num_mode_per_cluster_mtx = np.array(num_mode_per_cluster).reshape(network._rows, network._cols)
    mode_per_cluster_mtx = np.array(mode_per_cluster).reshape(network._rows, network._cols)
    sns.heatmap(num_mode_per_cluster_mtx, ax=axs[int(i/ncols)][i%ncols], annot=mode_per_cluster_mtx, cmap='tab10', cbar=False, fmt='')
    axs[int(i/ncols)][i%ncols].set_title(feature)
    axs[int(i/ncols)][i%ncols].set_xticks([])
    axs[int(i/ncols)][i%ncols].set_yticks([])
for ax in axs[nrows-1, i%ncols+1:]:
    ax.remove()
f.suptitle("Most frequent value for each feature in each cluster", fontweight='bold');

# %%
plot_violin_by_cluster(
    indicators_df,
    features_to_cluster,
    cluster_column='cluster',
    ncols=3,
    figsize=(36,36),
    title=None
)

# %%
plot_distance_matrices(X, n_samples=500, clusters=indicators_df['cluster'].to_numpy())

# %%
X_2dim = indicators_df[features_to_cluster[:2]].values
network2dim = som(rows, cols, structure, som_params)
network2dim.train(X_2dim, autostop=True, epochs=100000)

# %%
network2dim.show_network()

# %%
# c'è anche il codice che fa solo clustering ma non consente di specificare la dimensione della mappa, TODO: ricontrollare!
# from pyclustering.cluster.somsc import somsc
# somsc_instance = somsc(X, autostop=True, epochs=100000, amount_clusters=9)
# somsc_instance.process()
# clusters = somsc_instance.get_clusters()

# TODO:
# espolare parametri
# provare con griglie di dimensioni diverse
# usare funzioni di visualizzazione e validazione in utils
# fare diverse run e vedere quanto cambiano gli score permutation invariant (dovrebbe provare il fatto che ha raggiunto convergenza)


