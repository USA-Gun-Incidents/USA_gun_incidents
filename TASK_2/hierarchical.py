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
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=False,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

# %% [markdown]
# ## Dataset preparation

# %% [markdown]
# ### Indexes

# %%
def compute_ratio_indicator(df, ext_df, gby, num, den, suffix, agg_fun):
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    df[num + '_' + den + suffix + '_ratio'] = np.divide(df[num], df[den + suffix], out=np.zeros_like(df[num]), where=(df[den + suffix] != 0))
    df.drop(columns=[den + suffix], inplace=True)
    return df

# %%
state = "ILLINOIS"
incidents_df['city'] = incidents_df['city'].fillna('UNK')
incidents_df['county'] = incidents_df['county'].fillna('UNK')

incidents_df = incidents_df[incidents_df['state'] == state].copy()
incidents_df['city'] = incidents_df['city'].fillna('UNK')
incidents_df['county'] = incidents_df['county'].fillna('UNK')

# %%
# 1st indicator
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_tot_year_city', 'sum')

# 2nd indicator
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'congressional_district'], 'n_killed', 'n_killed', '_tot_year_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'congressional_district'], 'n_injured', 'n_injured', '_tot_year_congdist', 'sum')

# 3rd indicator
incidents_df['n_killed_n_participants_ratio'] = np.divide(incidents_df['n_killed'], incidents_df['n_participants'], out=np.zeros_like(incidents_df['n_killed']), where=(incidents_df['n_participants'] != 0))
incidents_df['n_injured_n_participants_ratio'] = np.divide(incidents_df['n_injured'], incidents_df['n_participants'], out=np.zeros_like(incidents_df['n_injured']), where=(incidents_df['n_participants'] != 0))

# 4th indicator
incidents_df['n_unharmed_n_participants_ratio'] = np.divide(incidents_df['n_unharmed'], incidents_df['n_participants'], out=np.zeros_like(incidents_df['n_unharmed']), where=(incidents_df['n_participants'] != 0))
incidents_df['n_arrested_n_participants_ratio'] = np.divide(incidents_df['n_arrested'], incidents_df['n_participants'], out=np.zeros_like(incidents_df['n_arrested']), where=(incidents_df['n_participants'] != 0))

# other indicators
incidents_df = compute_ratio_indicator(incidents_df[incidents_df['state'] == state], incidents_df[incidents_df['state'] == state], ['year', 'city'], 'n_participants', 'n_participants', '_tot_year_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df[incidents_df['state'] == state], incidents_df[incidents_df['state'] == state], ['year', 'congressional_district'], 'n_participants', 'n_participants', '_tot_year_district', 'sum')
incidents_df['n_participants_female_over_male_ratio'] = np.divide(incidents_df['n_males'], incidents_df['n_females'])
incidents_df['n_participants_child_n_participants_ratio'] = np.divide(incidents_df['n_participants_child'], incidents_df['n_participants'])
incidents_df['n_participants_teen_n_participants_ratio'] = np.divide(incidents_df['n_participants_teen'], incidents_df['n_participants'])
incidents_df['n_participants_adult_n_participants_ratio'] = np.divide(incidents_df['n_participants_adult'], incidents_df['n_participants'])

# %%
incidents_df

# %%
incidents_df_clustering = incidents_df.select_dtypes(include='number')
incidents_df_clustering.isna().sum()

# %% [markdown]
# ## Clustering

# %%
clustering_columns = ['n_participants', 'avg_age_participants', 'max_age_participants',
    'n_males', 'n_females', 
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed',
    'n_males_n_males_tot_year_city_ratio',
    'n_killed_n_killed_tot_year_congdist_ratio',
    'n_injured_n_injured_tot_year_congdist_ratio',
    'n_killed_n_participants_ratio', 'n_injured_n_participants_ratio',
    'n_unharmed_n_participants_ratio', 'n_arrested_n_participants_ratio',
    'n_participants_n_participants_tot_year_city_ratio',
    'n_participants_n_participants_tot_year_district_ratio',
    'n_participants_female_over_male_ratio',
    'n_participants_child_n_participants_ratio',
    'n_participants_teen_n_participants_ratio',
    'n_participants_adult_n_participants_ratio']

standardization_columns = ['n_participants', 'avg_age_participants', 'max_age_participants',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed']

dropna_columns = ['latitude', 'longitude', 'n_participants', 'min_age_participants', 'avg_age_participants', 'max_age_participants', 'n_participants_child', 
    'n_participants_teen', 'n_participants_adult', 'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants']

# %%
incidents_df_clustering = incidents_df_clustering.dropna(subset=dropna_columns)
incidents_df = incidents_df.dropna(subset=dropna_columns)
incidents_df_clustering[clustering_columns].isna().sum()

# %%
# standardization
std_scaler = StandardScaler()
std_scaler.fit(incidents_df_clustering[standardization_columns].values)
X = std_scaler.fit_transform(incidents_df_clustering[standardization_columns].values)

# %%
# clustering
n_clusters = 134
single_link = AgglomerativeClustering(linkage="single", compute_distances=True, n_clusters=n_clusters).fit(X)
complete_link = AgglomerativeClustering(linkage="complete", compute_distances=True, n_clusters=n_clusters).fit(X)
average = AgglomerativeClustering(linkage="average", compute_distances=True, n_clusters=n_clusters).fit(X)
ward = AgglomerativeClustering(linkage="ward", compute_distances=True, n_clusters=n_clusters).fit(X)

# %%
# results
print("min: " + str(single_link.labels_))
print("max: " + str(complete_link.labels_))
print("average: " + str(average.labels_))
print("ward: " + str(ward.labels_))

# %%
def plot_dendrogram(model, p):
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
    dendrogram(linkage_matrix, truncate_mode="level", p=p)

# %%
plt.figure(figsize=(15, 12))
plt.title("Hierarchical Clustering Dendrogram - Ward")
plot_dendrogram(ward, 5)
plt.xlabel("Number of points in node (or index of point if no parenthesis)")
plt.show()

# %%
plt.figure(figsize=(15, 12))
plt.title("Hierarchical Clustering Dendrogram - Single Link")
plot_dendrogram(single_link, 8)
plt.xlabel("Number of points in node (or index of point if no parenthesis)")
plt.show()

# %%
plt.figure(figsize=(15, 12))
plt.title("Hierarchical Clustering Dendrogram - Complete Link")
plot_dendrogram(complete_link, 5)
plt.xlabel("Number of points in node (or index of point if no parenthesis)")
plt.show()

# %%
plt.figure(figsize=(15, 12))
plt.title("Hierarchical Clustering Dendrogram - Average Link")
plot_dendrogram(average, 10)
plt.xlabel("Number of points in node (or index of point if no parenthesis)")
plt.show()

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

group_values = incidents_df.groupby(tags, group_keys=True, as_index=False).count()
tag_groups = group_values.filter(tags)
group_values = group_values.drop(tags, axis=1)

# %%
tag_groups['n_incidents'] = group_values.max(axis=1)

tag_groups

# %%
len(incidents_df) - tag_groups['n_incidents'].sum() # must be 0

# %%
incidents_per_different_tagmap = tag_groups.sort_values('n_incidents')['n_incidents'].values

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


