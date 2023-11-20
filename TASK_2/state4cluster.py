# -*- coding: utf-8 -*-
# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
#  
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
#  
# # Data and indicators understanding by state for clustering
#

# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
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
columns = ['location_importance', 'avg_age_participants', 'n_participants', 'age_range', 
    'n_participants_child_prop', 'n_participants_teen_prop', 'n_males_pr', 'n_killed_pr', 'n_injured_pr',
    'n_arrested_pr', 'n_unharmed_pr', 'log_n_males_n_males_mean_semest_congd_ratio', 'log_avg_age_mean_SD',
    'avg_age_entropy', 'city_entropy', 'address_entropy', 'n_participants_adult_entropy', 'tags_entropy']

# %%
illinois_df = indicators_df[indicators_df['state']=='ILLINOIS'][columns].dropna()
illinois_df[['latitude', 'longitude', 'county', 'city']] = incidents_df.loc[illinois_df.index, [
    'latitude', 'longitude', 'county', 'city']]

illinois_df.info()
illinois_df.head(2)

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
for column in columns:
    vmin, vmax = illinois_merged[column].agg(['min', 'max'])
    illinois_merged.plot(column=column, cmap='plasma', figsize=(10, 6), vmin=vmin, vmax=vmax,
        legend=True, legend_kwds={'label': column, 'shrink': 1}, edgecolor='black', linewidth=0.5)
    plt.title(f'Illinois counties')
    plt.xticks([])
    plt.yticks([])
    plt.show()
