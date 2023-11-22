# %% [markdown]
# # Hierarchical Clustering

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# %%
# import dataset
incidents_df = pd.read_csv('../data/incidents_cleaned_indicators.csv', index_col=False)
incidents_df.drop("Unnamed: 0", axis=1, inplace=True)

# %%
incidents_df.info()

# %% [markdown]
# ## Dataset preparation
# We already have normalized data, so we just select the better state for clustering analysis

# %%
incidents_df_full = pd.read_csv('../data/incidents_cleaned.csv')
incidents_df_full.head(2)

# %%
# select a subset of records regarding a certain state
incidents_df['state'] = incidents_df_full['state']

state = "ILLINOIS"
incidents_df = incidents_df[incidents_df['state'] == state]
incidents_df.drop('state', axis=1, inplace=True)

# %%
incidents_df.isna().sum()

# %%
incidents_df.dropna(inplace=True)

incidents_df.shape

# %%
# print all indexes for clustering
incidents_df.columns

# %% [markdown]
# ## Clustering

# %%
# clustering
algorithms = ["single", "complete", "average", "ward"]
models = []

for algorithm in algorithms:
    models.append(AgglomerativeClustering(linkage=algorithm, compute_distances=True).fit(incidents_df))

# %%
# results
print("min: " + str(single_link.labels_))
print("max: " + str(complete_link.labels_))
print("average: " + str(average.labels_))
print("ward: " + str(ward.labels_))

# %%
models[3].n_clusters_

# %%
def plot_dendrogram(model, p, ax):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, truncate_mode="lastp", p=p, leaf_rotation=60, leaf_font_size = 8, show_contracted=True, ax=ax)

# %%
f, axs = plt.subplots(ncols=4, figsize=(32,7))

for i in range(len(models)):
    axs[i].set_title(algorithms[i])
    axs[i].set_xlabel('IncidentID or (Cluster Size)')
    axs[i].set_ylabel('Distance')

    plot_dendrogram(models[i], 30, axs[i])

plt.suptitle(('Hierarchical Clustering Dendograms'), fontsize=18, fontweight='bold')
plt.show()

# %%


# %%
from enum import Enum

class IncidentTag(Enum):
    firearm = 1
    air_gun = 2
    shots = 3
    aggression = 4
    suicide = 5
    injuries = 6
    death = 7
    road = 8
    illegal_holding = 9
    house = 10
    school = 11
    children = 12
    drugs = 13
    officers = 14
    organized = 15
    social_reasons = 16
    defensive = 17
    workplace = 18
    abduction = 19
    unintentional = 20

tags = [v.name for v in IncidentTag]

group_values = incidents_df_full[incidents_df_full['state'] == state].dropna().groupby(tags, group_keys=True, as_index=False).count()
tag_groups = group_values.filter(tags)
group_values = group_values.drop(tags, axis=1)

# %%
tag_groups['n_incidents'] = group_values.max(axis=1)

tag_groups

# %%
len(incidents_df) - tag_groups['n_incidents'].sum() # must be 0

# %%
incidents_per_different_tagmap = tag_groups.sort_values('n_incidents')['n_incidents'].values
n_clusters = len(incidents_per_different_tagmap)

incidents_per_different_tagmap

# %%
cluster_map = pd.DataFrame()
cluster_map['cluster'] = single_link.labels_

single_link_cluster_population = []
for i in range(n_clusters):
    single_link_cluster_population.append((cluster_map.loc[cluster_map['cluster'] == i].count()).values[0])
single_link_cluster_population.sort()

np.array(single_link_cluster_population)

# %%
cluster_map = pd.DataFrame()
cluster_map['cluster'] = complete_link.labels_

complete_link_cluster_population = []
for i in range(n_clusters):
    complete_link_cluster_population.append((cluster_map.loc[cluster_map['cluster'] == i].count()).values[0])
complete_link_cluster_population.sort()

np.array(complete_link_cluster_population)

# %%
cluster_map = pd.DataFrame()
cluster_map['cluster'] = average.labels_

average_link_cluster_population = []
for i in range(n_clusters):
    average_link_cluster_population.append((cluster_map.loc[cluster_map['cluster'] == i].count()).values[0])
average_link_cluster_population.sort()

np.array(average_link_cluster_population)

# %%
cluster_map = pd.DataFrame()
cluster_map['cluster'] = ward.labels_

ward_cluster_population = []
for i in range(n_clusters):
    ward_cluster_population.append((cluster_map.loc[cluster_map['cluster'] == i].count()).values[0])
ward_cluster_population.sort()

np.array(ward_cluster_population)


