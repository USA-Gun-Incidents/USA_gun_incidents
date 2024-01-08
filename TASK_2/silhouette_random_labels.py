# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
#
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa

# %%
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# %%
# load the data
incidents_df = pd.read_csv('../data/incidents_indicators.csv', index_col=0)
# load the names of the features to use for clustering
features_to_cluster = json.loads(open('../data/indicators_names.json').read())
# for clustering we will use all the extracted indicators except the projected coordinates
features_to_cluster = [feature for feature in features_to_cluster if feature not in ['lat_proj', 'lon_proj']]
# drop nan
incidents_df = incidents_df.dropna(subset=features_to_cluster)
# initialize a colum for the clustering labels
incidents_df['cluster'] = None
# project on the indicators
indicators_df = incidents_df[features_to_cluster]

# %%
minmax_scaler = MinMaxScaler()
X = minmax_scaler.fit_transform(indicators_df.values)

# %%
n_clusters = [4, 7, 9]
silhouette = {}
for n_cluster in n_clusters:
    print(f'Trying {n_cluster} clusters')
    silhouette_values = []
    for i in range(10):
        print(f'Iteration {i}')
        labels = np.random.randint(0, n_cluster, len(indicators_df))
        silhouette_values.append(silhouette_score(X, labels))
    mean_silhouette = np.mean(silhouette_values)
    std_silhouette = np.std(silhouette_values)
    silhouette[n_cluster] = {'mean': mean_silhouette, 'std': std_silhouette}
silhouette_df = pd.DataFrame(silhouette)
silhouette_df

# %%
silhouette_df.to_csv('../data/silhouette.csv')


