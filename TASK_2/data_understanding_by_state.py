# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
#  
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
#  
# # Data and indicators understanding by state for clustering
#

# %% [markdown]
# The goal of this section of the data analysis is to examine the distribution of data for each state in the dataset. The main purpose is to select a specific state and identify its most relevant characteristics in order to subsequently apply clustering algorithms.
#
# This phase of the analysis also aims to determine if there is a state that can serve as a well-representative sample of the entire dataset. The objective is to assess whether the information derived from clustering algorithms applied to a single state can be generalized and extended to the entire dataset.

# %% [markdown]
# We import the libraries:

# %%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# %% [markdown]
# We load the dataset and a the list of indicators conteined in dataset:

# %%
incidents_df = pd.read_csv(
    '../data/incidents_indicators.csv',
    index_col=0,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

# %%
incidents_df.head(2)

# %%
f = open('../data/indicators_names.json')
ind_names_list = json.loads(f.read())

# %% [markdown]
# ### Number of not NaN Entries by State

# %% [markdown]
# We group data by state, and subsequently by state, county, and city, in order to understand how much data we have for each geographical region.

# %%
incidents_df.groupby('state').size().sort_values(ascending=False).head(15)

# %%
incidents_df.groupby(['state', 'county', 'city']).size().sort_values(ascending=False).head(15)

# %% [markdown]
# We first notice that the state with the most entries is Illinois, with 17554 entries. The majority of entries for this state are in Chicago, Cook County, which is also the city with the most entries in the dataset. Following that, we have California and Florida states, which have 16291 and 15005 entries, respectively.

# %% [markdown]
# In order to visualize the data on a map, we include a code for each state from an [official list of states](https://www2.census.gov/geo/docs/reference/state.txt') that is compatible with the Plotly library used for plotting.

# %%
usa_states_df = pd.read_csv(
    'https://www2.census.gov/geo/docs/reference/state.txt',
    sep='|',
    dtype={'STATE': str, 'STATE_NAME': str}
)
usa_name_alphcode = usa_states_df.set_index('STATE_NAME').to_dict()['STUSAB']

# %% [markdown]
# We create a dataset representing data from the dataset, grouped by state, and containing the count of non-NaN entries for each indices in each state.

# %%
incidents_grouped_by_state = incidents_df.groupby(['state']).size().sort_values(ascending=False).to_frame().reset_index()
incidents_grouped_by_state.rename(columns={0: 'not_nan_entries'}, inplace=True)

incidents_grouped_by_state['px_code'] = incidents_grouped_by_state['state'].str.title().map(usa_name_alphcode)
for attribute in ['city', 'county'] + ind_names_list + ['severity', 'n_participants', 'surprisal_characteristics']:
    incidents_grouped_by_state[attribute+'_count'] = incidents_df.groupby('state')[
        attribute].count().sort_values(ascending=False).to_frame().reset_index()[attribute]

# %% [markdown]
# Establishing plot utilities:

# %%
hover_data = {
    'px_code': False,
    'not_nan_entries': True,
    'county_count': True,
    'city_count': True, 
    'lat_proj_count': True,
    'location_imp_count': True,
    'age_range_count': True,
    'surprisal_min_age_count': True,
    'n_child_prop_count': True,
    'surprisal_age_groups_count': True,
    'severity_count': True,
    'n_unharmed_prop_count': True,
    'n_males_prop_count': True,
    'surprisal_n_males_count': True,
    'n_arrested_prop_count': True,
    'n_participants_count': True,
    'surprisal_characteristics_count': True,
}

labels={
    'not_nan_entries':'Number of entries',
    'city_count': 'Number of cities',
    'county_count': 'Number of counties',
    'lat_proj_count': 'Latidude and Longitude', #'lat_proj_count, lon_proj_count'
    'location_imp_count': 'Location Importance and Address Type Suprisal', #location_imp_count, surprisal_address_type_count',
    'age_range_count': 'Age attributes', # 'age_range_count, avg_age_count', 
    'surprisal_min_age_count': 'Min Age Suprisal',
    'n_child_prop_count': 'Child and Teen Proportion',
    'surprisal_age_groups_count': 'Age Groups Surprisal', # 'surprisal_age_groups_count, surprisal_n_child_count
    'surprisal_n_child_count': 'Age Groups Suprisal',
    'severity_count': 'Severity Indices',
    'n_unharmed_prop_count': 'Unharmed Proportion',
    'n_males_prop_count': 'Males Proportion',
    'surprisal_n_males_count': 'Males Surprisal',
    'n_arrested_prop_count': 'Arrested Proportion',
    'n_participants_count': 'Number of Participants',
    'surprisal_characteristics_count': 'Characteristics Surprisal',
}


# %%
counts_columns = ['city_count', 'county_count'] + [
    indices + '_count' for indices in ind_names_list] + [
    'severity_count', 'n_participants_count', 'surprisal_characteristics_count']

# %% [markdown]
# We plot on a USA map the number of non-NaN entries by state for indices and numerical features.

# %%
fig = px.choropleth(incidents_grouped_by_state, 
    color='not_nan_entries',
    locations='px_code',
    locationmode="USA-states",
    scope="usa",
    title="Number of non-NaN Entries by State", 
    hover_name='state',
    hover_data=hover_data,
    labels=labels
)

fig.show()

# %% [markdown]
# Below, we create ratios for each index and numerical attribute, expressing the ratio of null entries for each attribute in the same state to the total number of entries with all non-null values for that state. The ratios have been scaled between 0 and 100 for visualization purposes and are represented in choropleth maps.

# %%
# create ration for number of NaN entries for each attributes by state wrt the total number of entries by state
ratio_columns = []

for attribute in ['city', 'county'] + ind_names_list + ['severity', 'n_participants']:
    incidents_grouped_by_state['nan_entries_'+attribute] = incidents_df[incidents_df[attribute].isna()
        ].groupby('state').size().sort_values(ascending=False).to_frame().reset_index()[0]
    
    incidents_grouped_by_state['nan_entries_'+attribute+'_ratio'] = 100*incidents_grouped_by_state['nan_entries_'+attribute
        ] / incidents_grouped_by_state['not_nan_entries']
    
    ratio_columns += ['nan_entries_'+attribute+'_ratio']

# %%
incidents_grouped_by_state[['state']+ratio_columns].head(4)

# %%
from plot_utils import plot_not_nan_entries_by_state

labels_list = ['city', 'county'] + ind_names_list + ['severity', 'n_participants']

plot_not_nan_entries_by_state(df=incidents_grouped_by_state, 
    labels=labels_list[:8],
    attribute_list=ratio_columns[:8], n_rows=2, n_columns=4)
plot_not_nan_entries_by_state(df=incidents_grouped_by_state, labels=labels_list[9:17],
    attribute_list=ratio_columns[8:16], n_rows=2, n_columns=4)
plot_not_nan_entries_by_state(df=incidents_grouped_by_state, labels=labels_list[17:],
    attribute_list=ratio_columns[16:], n_rows=2, n_columns=4)

# %% [markdown]
# In the choropleth maps above, values of ratios for null entries are visualized for each attribute, categorized by state.
#
# The ratio scale is depicted using colors ranging from blue to yellow: cooler colors represent lower ratios, indicating states with fewer null entries in relation to the total, while warmer colors represent higher ratios. A cooler color suggests a state with a lower number of null entries for that feature, which is preferable.
#
# Note that the choropleth maps are presented in three separate figures, each with a different color scale, in order to best represent the ranges of ratios for the displayed attributes.
#
# We can observe that, across all states, features related to age have the highest number of missing entries. In all features, Texas exhibits lower ratio values compared to other states, followed closely by Illinois and California.

# %% [markdown]
# Below, we have presented the mean values of the ratios among states to highlight which features have fewer missing entries. Consistent with the observations above, geographic data has the fewest missing values as a percentage, followed by indices representing incident characteristics and the number of participants.

# %%
incidents_grouped_by_state[ratio_columns].describe().mean().sort_values(ascending=True)

# %% [markdown]
# Below, we have presented bar plots to visualize the percentages of missing values for each attribute by state (in orange) relative to the total number of entries (in blue) for that state.
#
# The percentages displayed above each bar represent the proportion of missing values for the respective features.
#
# These percentages are calculated based on the total number of entries for the state. In contrast, in the choropleth maps and the data frame above, the ratios are calculated based on the number of entries with all non-null values, resulting in higher values.
#
# We have chosen to calculate and represent both ratios to provide a more comprehensive view of the dataset, aiming to identify the most suitable features for each state to use in clustering algorithms.
#
# Note that the y-axis of the bar plots is in a logarithmic scale.

# %%
from plot_utils import plot_missing_values_for_state

for attribute in labels_list:
    plot_missing_values_for_state(df=incidents_df, attribute=attribute)

# %% [markdown]
# The observations that can be made are similar to those from before; there is a noticeable higher number of missing values for features related to age. Illinois and California appear to be among the states with lower percentages of missing values.

# %% [markdown]
# Finally, we printed the number of entries containing all non-null values for the columns listed in the following list.

# %%
columns = [
    'lat_proj', 'lon_proj', 'location_imp', 'surprisal_address_type', 'age_range', 'avg_age',
    'surprisal_min_age', 'n_child_prop', 'n_teen_prop', 'surprisal_age_groups', 'n_killed_prop',
    'n_injured_prop', 'surprisal_n_injured', 'n_unharmed_prop', 'n_males_prop',
    'surprisal_n_males', 'surprisal_characteristics', 'n_arrested_prop', 'n_participants',
    'surprisal_day', 'severity'
]

# %%
incidents_df[incidents_df['state']=='CALIFORNIA'][columns].dropna().shape[0]

# %%
incidents_df[incidents_df['state']=='ILLINOIS'][columns].dropna().shape[0]

# %%
incidents_df[incidents_df['state']=='TEXAS'][columns].dropna().shape[0]

# %%
incidents_df[incidents_df['state']=='FLORIDA'][columns].dropna().shape[0]

# %% [markdown]
# Given that Illinois has the highest number of non-null features, we have decided to continue the analysis by focusing on the data related to this state.

# %% [markdown]
# ## Illinois Data

# %% [markdown]
# Below, we printed the distribution of data related to the state of Illinois, divided by features (in orange), and compared them with the distribution of data for the entire dataset for the same features (in blue).

# %%
ind_names_discrete_list = [
    'age_range',
    'avg_age',
    'n_participants',
    ]
ind_names_continuous_list = [indice for indice in columns if indice not in ind_names_discrete_list]

# %%
from plot_utils import continuous_attribute_distribuition_plot, discrete_attribute_distribuition_plot

for attribute in ind_names_continuous_list:
    continuous_attribute_distribuition_plot(df=incidents_df, attribute=attribute, state='ILLINOIS')
for attribute in ['age_range','avg_age',]:
    discrete_attribute_distribuition_plot(df=incidents_df, attribute=attribute, state='ILLINOIS')

# %% [markdown]
# With the exception of latitude and longitude projections, which are naturally different for each state, we can observe that the distribution of data for the state of Illinois is very similar to that of the entire dataset for all features.

# %% [markdown]
# Below, the mean, standard deviation, minimum, maximum, and quantiles for the selected features have been printed for both the Illinois-specific data and the entire dataset.
#
# To make the data more easily interpretable, boxplots representing the distributions of some selected features have been printed, comparing data related to the state of Illinois and the entire dataset.

# %%
incidents_df[incidents_df['state']=='ILLINOIS'][columns].describe()

# %%
incidents_df[columns].describe()

# %%
attributes = ['location_imp', 'surprisal_address_type', 'age_range', 'avg_age',
    'surprisal_n_males', 'surprisal_characteristics', 'n_participants', 'severity']
 
fig, ax = plt.subplots(int(np.ceil(len(attributes))), 2, figsize=(20, 15))
for i, attribute in enumerate(attributes):
    ax[i, 0].boxplot(incidents_df[incidents_df['state']=='ILLINOIS'][attribute].dropna(), vert=False)
    ax[i, 0].set_title(attribute+' Illinois')
    ax[i, 1].boxplot(incidents_df[attribute].dropna(), vert=False)
    ax[i, 1].set_title(attribute+' whole dataset')
fig.tight_layout()
plt.show()


# %% [markdown]
# We notice that the severity index shows a lower range of values for the state of Illinois (between 0 and 0.7) compared to the entire dataset (between 0 and 1.4), highlighting a significant number of outliers.
#
# The maximum number of participants is 20, while in the dataset, incidents with a higher number of people (up to a maximum of 103) are present, although they are considered outliers. However, the mean is very similar between the data for Illinois and the entire dataset.
#
# The average age in incidents that occurred in Illinois is lower (27 years) compared to the overall dataset average (30 years).
#
# The index representing the surprisal of address types in incidents has a lower mean and a slightly wider standard deviation for Illinois compared to the total dataset (mean=1.24 and standard deviation=1.63 vs mean=1.44 and standard deviation=1.48).
#
# For all other features, Illinois shows significant similarities with the entire population of the dataset.
#
# In conclusion, the analysis highlights some notable differences in Illinois data compared to the entire dataset, particularly for the severity index and the maximum number of participants. However, for most features, Illinois behaves similarly to the rest of the population, indicating that it could be a valid representative for further analysis and clustering algorithms.

# %% [markdown]
# Note that in a preliminary analysis, similar tests were also conducted for data related to the state of California, and the observations are analogous. However, in the notebook, only the results related to the state of Illinois are reported to keep the notebook clean.

# %% [markdown]
# Below we have plotted boxplots for the indicators and numerical features related to the state of Illinois, including those already reported above. This provides an overview of the selectable indicators for clustering algorithms.

# %%
fig, ax = plt.subplots(int(np.ceil(len(columns)/2)), 2, figsize=(20, 20))
for i, attribute in enumerate(columns):
    ax[i//2, i%2].boxplot(incidents_df[incidents_df['state']=='ILLINOIS'][attribute].dropna(), vert=False)
    ax[i//2, i%2].set_title(attribute)
fig.tight_layout()
plt.show()

# %% [markdown]
# Finally, we reported the correlation matrix. We chose to use Kendall because it does not assume that the variables are normally distributed, and it is less affected by outliers compared to other correlation coefficients.

# %%
plt.figure(figsize=(20, 8))
sns.heatmap(incidents_df[incidents_df['state']=='ILLINOIS'][columns].corr('kendall'), 
    annot=True, cmap='coolwarm',
    mask=np.triu(np.ones_like(incidents_df[incidents_df['state'
    ]=='ILLINOIS'][columns].corr('kendall'), dtype=bool)))
plt.title(f'Kendall Correlation Heatmap')
plt.tight_layout()
plt.show()

# %% [markdown]
# From the matrix, we can observe that some features are correlated with each other, both positively and negatively. However, these are features or indicators with similar semantics, such as the number of children and the average age. These results will be taken into consideration for the attribute selection for clustering.

# %% [markdown]
# ### Illinois Data Visualization

# %%
illinois_df = incidents_df[incidents_df['state']=='ILLINOIS'][columns].dropna()
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
for column in ind_names_list:
    vmin, vmax = illinois_merged[column].agg(['min', 'max'])
    illinois_merged.plot(column=column, cmap='plasma', figsize=(10, 6), vmin=vmin, vmax=vmax,
        legend=True, legend_kwds={'label': column, 'shrink': 1}, edgecolor='black', linewidth=0.5)
    plt.title(f'Illinois counties')
    plt.xticks([])
    plt.yticks([])
    plt.show()



# %%
import geopandas
def plot_illinois_map(
    illinoise_df,
    col_to_plot,
    cmap='plasma',
    ):
    # select only illinois data
    illinois_df = illinoise_df[illinoise_df['state']=='ILLINOIS']

    # make county names uniform
    illinois_df['county'] = illinois_df['county'].str.replace(' County', '')
    illinois_df['county'] = illinois_df['county'].str.replace('Saint Clair', 'St. Clair')
    illinois_df['county'] = illinois_df['county'].str.replace('DeWitt', 'De Witt')

    # load map data
    illinois_map = geopandas.read_file('../cb_2018_us_county_500k') # todo
    illinois_merged = illinois_map.merge(illinois_df, left_on='NAME', right_on='county')
    illinois_merged = illinois_merged[illinois_merged['STATEFP']=='17']

    vmin, vmax = illinois_merged[col_to_plot].agg(['min', 'max'])
    illinois_merged.plot(column=col_to_plot, cmap=cmap, figsize=(10, 6), vmin=vmin, vmax=vmax,
        legend=True, legend_kwds={'label': col_to_plot, 'shrink': 1}, edgecolor='black', linewidth=0.5)
        
    plt.title(f'{col_to_plot} in Illinois counties')
    plt.xticks([])
    plt.yticks([])
    plt.show()
