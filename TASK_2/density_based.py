# -*- coding: utf-8 -*-
# %% [markdown]
# # Import library and dataset

# %%
import pandas as pd
import numpy as np

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=False,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

# %%
incidents_df.head(2)

# %%
incidents_df.columns

# %% [markdown]
# # Chose best state

# %% [markdown]
# ### Visualizzo il numero di dati che ho per ogni stato

# %%
incidents_df[incidents_df['county'].isna() | incidents_df['city'].isna()].groupby('state').size().sort_values(ascending=False)

# %%
incidents_df.groupby('state').size().sort_values(ascending=False)

# %%
incidents_df.groupby(['state', 'county', 'city']).size().sort_values(ascending=False)

# %%
# convert series a in dataframe
incidents_grouped = incidents_df.groupby(['state', 'county', 'city']).size().sort_values(ascending=False).to_frame().reset_index()
incidents_grouped.rename(columns={0: 'entries'}, inplace=True)
incidents_grouped

# %%
import matplotlib.pyplot as plt

# %%
# associate a color to each state and map each county to the color of its state
states = incidents_grouped['state'].unique()
plt_colors = plt.cm.tab20(np.linspace(0, 1, len(states)))
colors_dict_state = dict(zip(states, plt_colors))

def get_color(row):
    return colors_dict_state[row['state']]
colors_dict_county = dict(zip(incidents_grouped['county'], incidents_grouped.apply(get_color, axis=1)))


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
incidents_grouped_by_state = incidents_df.groupby(['state']).size().sort_values(ascending=False).to_frame().reset_index()
incidents_grouped_by_state.rename(columns={0: 'not_nan_entries'}, inplace=True)

incidents_grouped_by_state['px_code'] = incidents_grouped_by_state['state'].str.title().map(usa_name_alphcode)
incidents_grouped_by_state['nan_entries_city'] = incidents_df[incidents_df['city'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_county'] = incidents_df[incidents_df['county'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_lat_long'] = incidents_df[incidents_df['latitude'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_n_participants'] = incidents_df[incidents_df['n_participants'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]
incidents_grouped_by_state['nan_entries_date'] = incidents_df[incidents_df['date'].isna()].groupby('state'
    ).size().sort_values(ascending=False).to_frame().reset_index()[0]

# %%
import plotly.express as px

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
                'nan_entries_date': True},
)
fig.update_geos(fitbounds="locations", visible=False)
fig.show()

# %%
incidents_grouped_by_state['nan_entries_city_ratio'] = incidents_grouped_by_state['nan_entries_city'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_county_ratio'] = incidents_grouped_by_state['nan_entries_county'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_lat_long_ratio'] = incidents_grouped_by_state['nan_entries_lat_long'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_n_participants_ratio'] = incidents_grouped_by_state['nan_entries_n_participants'
    ] / incidents_grouped_by_state['not_nan_entries']
incidents_grouped_by_state['nan_entries_date_ratio'] = incidents_grouped_by_state['nan_entries_date'
    ] / incidents_grouped_by_state['not_nan_entries']

# %%
attribute_list = ['nan_entries_city_ratio', 'nan_entries_county_ratio', 'nan_entries_lat_long_ratio', 
    'nan_entries_n_participants_ratio', 'nan_entries_date_ratio']
label_list = ['City', 'County', 'Latitude and Longitude', 'Number of Participants', 'Date']

for attribute, label in zip(attribute_list, label_list):
    frame = px.choropleth(
        incidents_grouped_by_state,
        color=attribute,
        locations='px_code',
        locationmode="USA-states",
        scope="usa",
        title=f"Number of NaN entries by state for {label}",
        hover_name='state',
        hover_data={
            'px_code': False,
            'not_nan_entries': True,
            'nan_entries_city': True,
            'nan_entries_county': True,
            'nan_entries_lat_long': True,
            'nan_entries_n_participants': True,
            'nan_entries_date': True,
        },
    )

    frame.update_layout(
        #coloraxis=dict(colorbar=dict(orientation='h', y=-0.15)),
        coloraxis_colorbar=dict(title=f'Ratio NaN entries for {label}'))
    frame.update_geos(fitbounds="locations", visible=False)
    frame.show()

# %% [markdown]
# ### Entries per state without NaN

# %%
columns = ['latitude', 'longitude', 'n_participants', 'min_age_participants', 'avg_age_participants', 'max_age_participants', 
    'n_participants_child', 'n_participants_teen', 'n_participants_adult', 'n_males', 'n_females', 'n_killed', 'n_injured', 
    'n_arrested', 'n_unharmed', ]

# %%
incidents_df[incidents_df['state']=='CALIFORNIA'].dropna(subset=columns).shape[0]

# %%
incidents_df[incidents_df['state']=='ILLINOIS'].dropna(subset=columns).shape[0]

# %%
incidents_df[incidents_df['state']=='TEXAS'].dropna(subset=columns).shape[0]

# %%
incidents_df[incidents_df['state']=='FLORIDA'].dropna(subset=columns).shape[0]

# %% [markdown]
# # Prepare dataset and indices for choosen state

# %%
columns = ['latitude', 'longitude', 'n_participants', 'min_age_participants', 'avg_age_participants', 'max_age_participants', 'n_participants_child', 
    'n_participants_teen', 'n_participants_adult', 'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants']

# %%
def compute_ratio_indicator(df, ext_df, gby, num, den, suffix, agg_fun):
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    df[num+'_'+den+suffix+'_ratio'] = np.divide(df[num], df[den+suffix], out=np.zeros_like(df[num]), where=(df[den+suffix] != 0))
    df.drop(columns=[den+suffix], inplace=True)
    return df

# %%
def create_indicator(df, state):
    df = compute_ratio_indicator(incidents_df[incidents_df['state'] == state], incidents_df[incidents_df['state'] == state], 
        ['year', 'city'], 'n_participants', 'n_participants', '_tot_year_city', 'sum')
    df = compute_ratio_indicator(incidents_df[incidents_df['state'] == state], incidents_df[incidents_df['state'] == state], 
        ['year', 'congressional_district'], 'n_participants', 'n_participants', '_tot_year_district', 'sum')

    df['n_killed_n_participants_ratio'] = np.divide(df['n_killed'], df['n_participants'],
        out=np.zeros_like(df['n_killed']), where=(df['n_participants'] != 0))
    df['n_injured_n_participants_ratio'] = np.divide(df['n_injured'], df['n_participants'],
        out=np.zeros_like(df['n_injured']), where=(df['n_participants'] != 0))
    df['n_arrested_n_participants_ratio'] = np.divide(df['n_arrested'], df['n_participants'],
        out=np.zeros_like(df['n_arrested']), where=(df['n_participants'] != 0))
    df['n_unharmed_n_participants_ratio'] = np.divide(df['n_unharmed'], df['n_participants'],
        out=np.zeros_like(df['n_unharmed']), where=(df['n_participants'] != 0))
    
    df['n_participants_child_n_participants_ratio'] = np.divide(df['n_participants_child'], df['n_participants'])
    df['n_participants_teen_n_participants_ratio'] = np.divide(df['n_participants_teen'], df['n_participants'])
    df['n_participants_adult_n_participants_ratio'] = np.divide(df['n_participants_adult'], df['n_participants'])

    df['n_participants_female_over_male_ratio'] = np.divide(df['n_males'], df['n_females'])

    return df

def create_dataset(state):
    df = incidents_df[incidents_df['state'] == state].copy()
    df['city'] = df['city'].fillna('UNK')
    df['county'] = df['county'].fillna('UNK')

    df = create_indicator(df, state)

    df = df.dropna(subset=columns)

    return df

# %%
illinois_df = create_dataset('ILLINOIS')
california_df = create_dataset('CALIFORNIA')

# %%
illinois_df.head(2)

# %% [markdown]
# # Density clustering

# %% [markdown]
# ## Indices correlation

# %%
columns = ['n_participants', 'avg_age_participants',
    'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed',
    'n_participants_n_participants_tot_year_district_ratio',
    'n_killed_n_participants_ratio', 'n_injured_n_participants_ratio',
    'n_arrested_n_participants_ratio', 'n_unharmed_n_participants_ratio',
    'n_participants_child_n_participants_ratio',
    'n_participants_teen_n_participants_ratio',
    'n_participants_adult_n_participants_ratio',
    'n_participants_female_over_male_ratio']

# %%
corr_matrix_illinois = illinois_df[columns].corr()

import seaborn as sns

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_illinois, annot=True, cmap=plt.cm.Reds, mask=np.triu(corr_matrix_illinois))
plt.show()

# %%
corr_matrix_california = california_df[columns].corr()

import seaborn as sns

plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_california, annot=True, cmap=plt.cm.Reds, mask=np.triu(corr_matrix_california))
plt.show()

# %%
illinois_df[columns].describe()

# %%
# show Nan values in illinois_df[columns]
illinois_df[columns].isna().sum()


# %%
# select features for clustering
columns = ['n_participants', 'avg_age_participants', 'max_age_participants',
    #'n_males', 'n_females', 
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed',
    #'n_killed_n_participants_ratio', 'n_injured_n_participants_ratio',
    #'n_arrested_n_participants_ratio', 'n_unharmed_n_participants_ratio',
    #'n_participants_child_n_participants_ratio',
    #'n_participants_teen_n_participants_ratio',
    #'n_participants_adult_n_participants_ratio'
    ]

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
    #core point = point that has at least min_samples in its eps-neighborhood (punto interno al cluster)

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k # array of booleans where True = point in cluster k

        xy = X_std[class_member_mask & core_samples_mask] # punti sul bordo dei cluster
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=10,
        )

        # plot noise points
        xy = X_std[class_member_mask & ~core_samples_mask] # punti che non sono nel cluster
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "x",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=3,
        )
    
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

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, labels))

    if plot_clusters:
        plot_dbscan(X, db)


# %% [markdown]
# The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

# %%
eps = [0.05, 0.02]#, 0.1]
min_samples = [5, 10]#, 20]#,30]

# %% [markdown]
# ## Clustering: Illinois

# %%
X_std_illinois = standardization(illinois_df, columns)
for e in eps:
    for m in min_samples:
        print(f'eps={e}, min_samples={m}')
        dbscan(X_std_illinois, eps=e, min_samples=m)
        print('\n')

# %%
dbscan(X_std_illinois, eps=0.05, min_samples=5, plot_clusters=True)

# %%
dbscan(X_std_illinois, eps=0.3, min_samples=20, plot_clusters=True)

# %% [markdown]
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.

# %%
X_illinois = illinois_df[columns].values
dbscan(X_illinois, eps=0.3, min_samples=5, plot_clusters=True)

# %% [markdown]
# ## Clustering: California

# %%
X_std_california = standardization(california_df, columns)
for e in eps:
    for m in min_samples:
        print(f'eps={e}, min_samples={m}')
        dbscan(X_std_california, eps=e, min_samples=m)
        print('\n')

# %%
dbscan(X_std_california, eps=0.03, min_samples=5, plot_clusters=True)
