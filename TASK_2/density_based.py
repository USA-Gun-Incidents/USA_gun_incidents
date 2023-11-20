# -*- coding: utf-8 -*-
# %% [markdown]
# # Import library and dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=False,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

indicators_df = pd.read_csv(
    '../data/incidents_cleaned_indicators.csv', 
    index_col=0
)

# %%
indicators_df.head(2)

# %%
incidents_df.head(2)

# %%
indicators_df = pd.concat([indicators_df, incidents_df[['state', 'city', 'county']]], axis=1)
indicators_df.head(2)

# %% [markdown]
# # Chose best state

# %% [markdown]
# ### Visualizzo il numero di dati che ho per ogni stato

# %%
indicators_df.groupby('state').size().sort_values(ascending=False).head(15)

# %%
# convert series a in dataframe
indicators_df.groupby(['state', 'county', 'city']).size().sort_values(ascending=False).head(15)

# %% [markdown]
# ### Visualizzo per stato

# %%
usa_states_df = pd.read_csv(
    'https://www2.census.gov/geo/docs/reference/state.txt',
    sep='|',
    dtype={'STATE': str, 'STATE_NAME': str}
)
usa_name_alphcode = usa_states_df.set_index('STATE_NAME').to_dict()['STUSAB']

# %%
incidents_grouped_by_state = indicators_df.groupby(['state']).size().sort_values(ascending=False).to_frame().reset_index()
incidents_grouped_by_state.rename(columns={0: 'not_nan_entries'}, inplace=True)

incidents_grouped_by_state['px_code'] = incidents_grouped_by_state['state'].str.title().map(usa_name_alphcode)
incidents_grouped_by_state['nan_entries_city'] = indicators_df[indicators_df['city'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_county'] = indicators_df[indicators_df['county'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_lat_long'] = indicators_df[indicators_df['latitude'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_n_participants'] = indicators_df[indicators_df['n_participants'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_age_range'] = indicators_df[indicators_df['age_range'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_male_pr'] = indicators_df[indicators_df['n_males_pr'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_killed_pr'] = indicators_df[indicators_df['n_killed_pr'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_injured_pr'] = indicators_df[indicators_df['n_injured_pr'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_arrested_pr'] = indicators_df[indicators_df['n_arrested_pr'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_unharmed_pr'] = indicators_df[indicators_df['n_unharmed_pr'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_address_entropy'] = indicators_df[indicators_df['address_entropy'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_tags_entropy'] = indicators_df[indicators_df['tags_entropy'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]

# %%
fig = px.choropleth(incidents_grouped_by_state, 
    color='not_nan_entries',
    locations='px_code',
    locationmode="USA-states",
    scope="usa",
    title="Number of entries by state", 
    hover_name='state',
    hover_data={'px_code': False,
                'not_nan_entries': True,
                'nan_entries_city': True,
                'nan_entries_county': True,
                'nan_entries_lat_long': True,
                'nan_entries_n_participants': True,
                'nan_entries_age_range': True,
                'nan_entries_male_pr': True,
                'nan_entries_killed_pr': True,
                'nan_entries_injured_pr': True,
                'nan_entries_arrested_pr': True,
                'nan_entries_unharmed_pr': True,
                'nan_entries_address_entropy': True,
                'nan_entries_tags_entropy': True},
)

fig.show()

# %%
# create ration for number of NaN entries for each attributes by state wrt the total number of entries by state
incidents_grouped_by_state['nan_entries_city_ratio'] = 100*incidents_grouped_by_state['nan_entries_city'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_lat_long_ratio'] = 100*incidents_grouped_by_state['nan_entries_lat_long'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_n_participants_ratio'] = 100*incidents_grouped_by_state['nan_entries_n_participants'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_age_range_ratio'] = 100*incidents_grouped_by_state['nan_entries_age_range'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_male_pr_ratio'] = 100*incidents_grouped_by_state['nan_entries_male_pr'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_killed_pr_ratio'] = 100*incidents_grouped_by_state['nan_entries_killed_pr'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_injured_pr_ratio'] = 100*incidents_grouped_by_state['nan_entries_injured_pr'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_arrested_pr_ratio'] = 100*incidents_grouped_by_state['nan_entries_arrested_pr'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_unharmed_pr_ratio'] = 100*incidents_grouped_by_state['nan_entries_unharmed_pr'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_address_entropy_ratio'] = 100*incidents_grouped_by_state['nan_entries_address_entropy'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_tags_entropy_ratio'] = 100*incidents_grouped_by_state['nan_entries_tags_entropy'
    ] / incidents_grouped_by_state['not_nan_entries']

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

attribute_list = ['nan_entries_city_ratio', 'nan_entries_lat_long_ratio',
                'nan_entries_n_participants_ratio', 'nan_entries_male_pr_ratio', 'nan_entries_killed_pr_ratio',
                'nan_entries_injured_pr_ratio', 'nan_entries_arrested_pr_ratio', 'nan_entries_unharmed_pr_ratio',
                'nan_entries_address_entropy_ratio', 'nan_entries_tags_entropy_ratio']
label_list = ['City', 'Latitude and Longitude', 'Participants', 'Males Proportion',
            'Killed Proportion', 'Injured Proportion', 'Arrested Proportion', 'Unharmed Proportion',
            'Address Entropy', 'Tags Entropy']

# Create subplots
rows = 2
cols = 5
fig = make_subplots(
    rows=rows, cols=cols,
    specs=[[{'type': 'choropleth'} for c in range(cols)] for r in range(rows)],
    subplot_titles=label_list,
    vertical_spacing=0.3,
    horizontal_spacing=0.01,
)

for i, (attribute, label) in enumerate(zip(attribute_list, label_list), start=1):
    frame = px.choropleth(
        incidents_grouped_by_state,
        color=attribute,
        locations='px_code',
        locationmode="USA-states",
        hover_name='state',
        hover_data={
            'px_code': False,
            'not_nan_entries': True,
        },
    )

    choropleth_trace = frame['data'][0]
    fig.add_trace(choropleth_trace, 
        row=(i-1)//cols+1, 
        col=(i-1) % cols+1
    )
    fig.update_layout(
        title_text="Ratio of NaN entries by state for different attributes",
        showlegend=False,
    )
    fig.update_geos( 
        scope = 'usa',
        visible=False)

fig.show()

# %% [markdown]
# Visualizzo i missing value per ogni attributo per stato, le percentuali sono rispetto al numero totali si samples per stato, sopar i rates sono calcolati rispetto solo alle entrate con tutti valori non nulli, quindi sono diversi

# %%
indicators_df.columns


# %%
def plot_missing_values_for_state(df, attribute):
    fig, ax = plt.subplots(figsize=(20, 2))
    ax.bar(df.groupby('state')['state'].count().index, df.groupby('state')['state'].count().values, 
        label='#Total', edgecolor='black', linewidth=0.8, alpha=0.5)
    ax.bar(df[df[attribute].isna()].groupby('state')['state'].count().index, df[df[attribute].isna()
        ].groupby('state')['state'].count().values, label=f'#Missing {attribute}', edgecolor='black', linewidth=0.8)
    ax.set_xlabel('State')
    ax.set_yscale('log')
    ax.set_ylabel('Number of incidents')
    ax.legend()
    ax.set_title(f'Percentage of missing values for {attribute} values by state')
    ax.xaxis.set_tick_params(rotation=90)
    for state in df['state'].unique():
        plt.text(
            x=state, 
            y=df[df[attribute].isna()].groupby('state')['state'].count()[state], 
            s=str(round(100*df[df[attribute].isna()].groupby('state')['state'].count()[state] / 
            df.groupby('state')['state'].count()[state]))+'%', 
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=8)
    plt.show()

for attribute in ['city', 'latitude', 'location_importance', 'avg_age_participants',
       'n_participants', 'age_range', 'n_participants_child_prop',
       'n_participants_teen_prop', 'n_males_pr', 'n_killed_pr', 'n_injured_pr',
       'n_arrested_pr', 'n_unharmed_pr',
       'log_n_males_n_males_mean_semest_congd_ratio', 'log_avg_age_mean_SD',
       'avg_age_entropy', 'city_entropy', 'address_entropy',
       'n_participants_adult_entropy', 'tags_entropy']:
    plot_missing_values_for_state(df=indicators_df, attribute=attribute)

# %% [markdown]
# ### Entries per state without NaN

# %%
indicators_df[indicators_df['state']=='CALIFORNIA'].dropna().shape[0]

# %%
indicators_df[indicators_df['state']=='ILLINOIS'].dropna().shape[0]

# %%
indicators_df[indicators_df['state']=='TEXAS'].dropna().shape[0]

# %%
indicators_df[indicators_df['state']=='FLORIDA'].dropna().shape[0]

# %% [markdown]
# ## Illinois

# %%
def discrete_attribute_distribuition_plot(df, attribute, state):    
    plt.figure(figsize=(20, 2))
    plt.bar(df.groupby(attribute)[attribute].count().index,
        df.groupby(attribute)[attribute].count().values, 
        label='Whole dataset', edgecolor='black', linewidth=0.8, alpha=0.5)
    plt.bar(df[df['state']==state].groupby(attribute)[attribute].count().index, 
        df[df['state']==state].groupby(attribute)[attribute].count().values, 
        label=state, edgecolor='black', linewidth=0.8, alpha=0.8)
    plt.xlabel(f'Number of {attribute}')
    plt.ylabel('Number of incidents')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Number of {attribute} per incident')
    plt.show()

def continuous_attribute_distribuition_plot(df, attribute, state):
    plt.figure(figsize=(20, 2))
    plt.hist(df[attribute], bins=100, label='Whole dataset', edgecolor='black', linewidth=0.8, alpha=0.5)
    plt.hist(df[df['state']==state][attribute], bins=100, label=state, edgecolor='black', linewidth=0.8, alpha=0.8)
    plt.xlabel(f'{attribute}')
    plt.ylabel('Number of incidents')
    plt.legend()
    plt.yscale('log')
    plt.title(f'{attribute} distribuition')
    plt.show()

# %%
for attribute in ['n_participants', 'min_age_participants', 'avg_age_participants', 'max_age_participants',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed']:
    discrete_attribute_distribuition_plot(df=incidents_df, attribute=attribute, state='ILLINOIS')

# %%
for attribute in ['location_importance', 'avg_age_participants',
    'n_participants_child_prop', 'n_participants_teen_prop', 'n_males_pr', 'n_killed_pr', 'n_injured_pr',
    'n_arrested_pr', 'n_unharmed_pr', 'log_n_males_n_males_mean_semest_congd_ratio', 'log_avg_age_mean_SD',
    'avg_age_entropy', 'city_entropy', 'address_entropy', 'n_participants_adult_entropy', 'tags_entropy']:
    continuous_attribute_distribuition_plot(df=indicators_df, attribute=attribute, state='ILLINOIS')

# %%
fig, ax = plt.subplots(9, 2, figsize=(20, 32))
for i, attribute in enumerate(['location_importance', 'avg_age_participants', 'n_participants', 'age_range', 
    'n_participants_child_prop', 'n_participants_teen_prop', 'n_males_pr', 'n_killed_pr', 'n_injured_pr',
    'n_arrested_pr', 'n_unharmed_pr', 'log_n_males_n_males_mean_semest_congd_ratio', 'log_avg_age_mean_SD',
    'avg_age_entropy', 'city_entropy', 'address_entropy', 'n_participants_adult_entropy', 'tags_entropy']):
    ax[i//2, i%2].boxplot(indicators_df[indicators_df['state']=='ILLINOIS'][attribute].dropna(), vert=False)
    ax[i//2, i%2].set_title(attribute)
plt.show()

# %% [markdown]
# # Prepare dataset and indices for choosen state

# %%
columns = ['location_importance', 'avg_age_participants', 'age_range', 
    'n_participants_child_prop', 'n_participants_teen_prop', 'n_males_pr', 
    'n_killed_pr', 'n_arrested_pr']

# %%
illinois_df = indicators_df[indicators_df['state']=='ILLINOIS'][columns].dropna()
illinois_df.info()
illinois_df.head(2)

# %% [markdown]
# # Density clustering

# %% [markdown]
# DBSCAN: density-based cluster, define a cluster as a dense region of objects.
#
# Partitional clustering, number of clester automatically detected from algorithm.
# Points in low-density region are classified as noise.
#
# Pros: can handle irregular clusters and with arbitrary shape and size, works well when noise or oulier are present.
# an find many cluster that K-means could not find.
#
# Contro: not able to classified correctly whan the clusters have widley varing density, and have trouble with high dimensional data because density is difficult to define.
#
#

# %% [markdown]
# ## Indices correlation

# %%
corr_matrix_illinois = illinois_df[columns].corr()

import seaborn as sns

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_illinois, annot=True, cmap=plt.cm.Reds, mask=np.triu(corr_matrix_illinois))
plt.show()

# %%
illinois_df[columns].describe()

# %%
# show Nan values in illinois_df[columns]
illinois_df[columns].isna().sum()

# %% [markdown]
# ## Utilities

# %%
from sklearn.preprocessing import StandardScaler

def standardization(df, columns):
    std_scaler = StandardScaler()
    std_scaler.fit(df[columns].values)
    return std_scaler.transform(df[columns].values)

# %%
def plot_dbscan(X_std, db): 
    labels = db.labels_ 
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True # create an array of booleans where True = core point
    # core point = point that has at least min_samples in its eps-neighborhood (punto interno al cluster)

    plt.figure(figsize=(20, 8))

    colors = [plt.cm.rainbow_r(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k # array of booleans where True = point in cluster k

        xy = X_std[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=6,
            label=f'Cluster {k}'
        )

        # plot noise points
        xy = X_std[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "x",
            markerfacecolor=tuple(col),
            markeredgecolor=col,
            markersize=8,
            label=f'Cluster {k}'
        )

    plt.grid()
    plt.legend()
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

# %%
from sklearn.cluster import DBSCAN
from sklearn import metrics 

def dbscan(X, eps=0.1, min_samples=10, plot_clusters=False):
    # Compute DBSCAN      
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    if plot_clusters:
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        if len(np.unique(labels)) > 1:
            print("Silhouette Coefficient: %0.3f"
                % metrics.silhouette_score(X, labels))
        plot_dbscan(X, db)
    
    return {'eps': eps, 'min_samples': min_samples, 
        '#clusters': len(set(labels)) - (1 if -1 in labels else 0),
        '#noise': list(labels).count(-1),  '%noise': list(labels).count(-1)/X.shape[0]*100,
        'silhouette_coef': metrics.silhouette_score(X, labels), 
        '#cluster0': list(labels).count(0), '#cluster1': list(labels).count(1), 
        '#cluster2': list(labels).count(2), '#cluster3': list(labels).count(3), 
        '#cluster4': list(labels).count(4), '#cluster5': list(labels).count(5),
        '#cluster6': list(labels).count(6), '#cluster7': list(labels).count(7)}

# %% [markdown]
# The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

# %% [markdown]
# ### Find best EPS

# %%
from scipy.spatial.distance import pdist, squareform

def find_best_eps(X, k_list=[3, 5, 9, 15]):
    dist = pdist(X, 'euclidean') # pair wise distance
    dist = squareform(dist) # distance matrix given the vector dist
    
    # Calculate sorted list of distances for points for each k in k_list
    # and plot the graph of distance from k-th nearest neighbour
    fig, ax = plt.subplots(int(np.ceil(len(k_list)/3)), 3, figsize=(20, 8))

    for i, k in enumerate(k_list):
        kth_distances = list()
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])

        # Plot the graph of distance from k-th nearest neighbour
        ax[int(i/3), int(i%3)].plot(range(0, len(kth_distances)), sorted(kth_distances))
        ax[int(i/3), int(i%3)].set_ylabel('%sth near neighbor distance' %k)
        ax[int(i/3), int(i%3)].set_xlabel('Point Sorted according to distance of %sth near neighbor' %k)
        #ax[int(i/3), int(i%3)].set_yticks(np.linspace(0, 5, 12))
        ax[int(i/3), int(i%3)].set_ylim(0, 3)
        ax[int(i/3), int(i%3)].tick_params(axis='both', which='major', labelsize=8)
        ax[int(i/3), int(i%3)].grid(linestyle='--', linewidth=0.5, alpha=0.6)

    plt.show()

# %% [markdown]
# ## Clustering: Illinois

# %% [markdown]
# ### Std data

# %%
X_std_illinois = standardization(illinois_df, columns) #TODO: sono già standardizzati

# %%
#kneed_algorithm(X_std_illinois, neighbors=5)

# %%
find_best_eps(X_std_illinois, k_list=[3, 5, 9, 15, 20, 30]) # altro metodo per kneed point

# %%
eps = [0.5, 1, 1.5, 2]
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
min_samples = [5, 10, 15, 20]

dbscan_illinois = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in eps:
    for k in min_samples:
        db = dbscan(X_std_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois = pd.concat([dbscan_illinois, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois

# %%
dbscan_illinois_second = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in [1.7, 2.3, 2.5]:
    for k in [5, 10, 15, 20]:
        db = dbscan(X_std_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois_second = pd.concat([dbscan_illinois_second, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois_second

# %%
dbscan_illinois_third = pd.DataFrame(columns=['eps', 'min_samples', '#clusters', '#noise', '%noise', 'silhouette_coef',
    '#cluster0', '#cluster1', '#cluster2', '#cluster3', '#cluster4', '#cluster5', '#cluster6', '#cluster7'])

for e in [2.1, 2.2, 2.3]:
    for k in [5, 7, 10]:
        db = dbscan(X_std_illinois, eps=e, min_samples=k, plot_clusters=False)
        dbscan_illinois_third = pd.concat([dbscan_illinois_third, pd.DataFrame(db, index=[0])], ignore_index=True)

# %%
dbscan_illinois_third

# %% [markdown]
# ### Visualize results

# %% [markdown]
# scegliamo config 
# eps = 2.3, k = 10
#
# ultima

# %%
db = DBSCAN(eps=2.3, min_samples=10).fit(X_std_illinois)
plot_dbscan(X_std_illinois, db)

# %%
fig, ax = plt.subplots(7, 4, figsize=(20, 30))
index = 0
for i in range(8):
    for j in range(i+1, 8):
        ax[int(index/4), index%4].scatter(illinois_df.values[:, i], illinois_df.values[:, j], c=db.labels_, cmap='plasma_r', s=6)
        ax[int(index/4), index%4].set_xlabel(illinois_df.columns[i], fontsize=8)
        ax[int(index/4), index%4].set_ylabel(illinois_df.columns[j], fontsize=8)
        ax[int(index/4), index%4].tick_params(axis='both', which='major', labelsize=6)
        ax[int(index/4), index%4].grid(linestyle='--', linewidth=0.5, alpha=0.6)
        index = index + 1
#plt.suptitle('DBSCAN Clustering', fontsize=16)
plt.show()

# %%
# bar plot of number of incidents per cluster
cluster_counts = pd.Series(db.labels_).value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(cluster_counts.index, cluster_counts.values, edgecolor='black', linewidth=0.8, alpha=0.5)
plt.xlabel('Cluster')
plt.xticks(cluster_counts.index)
plt.ylabel('Number of incidents')
plt.yscale('log')
for i, v in enumerate(cluster_counts.values):
    plt.text(x=i-1, y=v, s=str(v), horizontalalignment='center', verticalalignment='bottom', fontsize=8)
plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
plt.title('Number of incidents per cluster')
plt.show()


# %%
fig, ax = plt.subplots(4, 2, figsize=(20, 15), sharex=False, sharey=False)
index = 0
for i in range(8):
    for cluster in np.unique(db.labels_):
        ax[int(index/2), index%2].hist(illinois_df.values[db.labels_==cluster, i], 
            bins=int(1+3.3*np.log(X_std_illinois[db.labels_==cluster, i].shape[0])), 
            label=f'Cluster {cluster}', edgecolor='black', linewidth=0.8, alpha=0.7)
    ax[int(index/2), index%2].set_xlabel(illinois_df.columns[i], fontsize=8)
    ax[int(index/2), index%2].set_yscale('log')
    ax[int(index/2), index%2].tick_params(axis='both', which='major', labelsize=6)
    ax[int(index/2), index%2].legend()
    ax[int(index/2), index%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)
    index = index + 1
        

# %%
illinois_df = pd.concat([illinois_df, pd.DataFrame(db.labels_, columns=['cluster'])], axis=1)
illinois_df = pd.concat([illinois_df, incidents_df[incidents_df['state']=='ILLINOIS'][['latitude', 
    'longitude', 'county', 'city' ]]], axis=1).dropna(subset=[
        'location_importance', 'avg_age_participants', 'age_range',
        'n_participants_child_prop', 'n_participants_teen_prop', 'n_males_pr',
        'n_killed_pr', 'n_arrested_pr'])
illinois_df.head(2)

# %% [markdown]
# ### cose da spostare sopra

# %%
import geopandas as gpd

illinois_df['county'] = illinois_df['county'].str.replace(' County', '')
illinois_df['county'] = illinois_df['county'].str.replace('Saint Clair', 'St. Clair')
illinois_df['county'] = illinois_df['county'].str.replace('DeWitt', 'De Witt')

illinois_map = gpd.read_file('../cb_2018_us_county_500k')
illinois_merged = illinois_map.merge(illinois_df, left_on='NAME', right_on='county')
illinois_merged = illinois_merged[illinois_merged['STATEFP']=='17']

# %%
illinois_merged.plot(column='NAME', cmap='plasma', figsize=(20, 6), 
    legend=True, legend_kwds={'loc':'center left', 'bbox_to_anchor':(1, 0.5), 'ncol':4}, 
    edgecolor='black', linewidth=0.5)
plt.title('Illinois counties')
plt.xticks([])
plt.yticks([])
plt.show()

# %%
for column in ['location_importance', 'avg_age_participants', 'age_range', 'n_participants_child_prop',
    'n_participants_teen_prop', 'n_males_pr', 'n_killed_pr', 'n_arrested_pr']:
    vmin, vmax = illinois_merged[column].agg(['min', 'max'])
    illinois_merged.plot(column=column, cmap='plasma', figsize=(10, 6), vmin=vmin, vmax=vmax,
        legend=True, legend_kwds={'label': column, 'shrink': 1}, edgecolor='black', linewidth=0.5)
    plt.title(f'Illinois counties')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# %%
for column in ['location_importance', 'avg_age_participants', 'age_range', 'n_participants_child_prop',
    'n_participants_teen_prop', 'n_males_pr', 'n_killed_pr', 'n_arrested_pr']:
    vmin, vmax = illinois_merged[column].agg(['min', 'max'])
    illinois_merged.plot(column=column, cmap='plasma', figsize=(5, 6), vmin=vmin, vmax=vmax,
        legend=True, legend_kwds={'shrink': 1}, edgecolor='black', linewidth=0.5)
    plt.title(f'Illinois counties - {column}')
    plt.xticks([])
    plt.yticks([])
    plt.show()
