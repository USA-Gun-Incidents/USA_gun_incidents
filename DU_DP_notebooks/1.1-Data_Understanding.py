# %% [markdown]
# # Task 1.1 Data Understanding

# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.append(os.path.abspath('..')) # TODO: c'è un modo per farlo meglio?
from plot_utils import *
from data_preparation_utils import *

# %% [markdown]
# We define constants and settings for the notebook:

# %%
%matplotlib inline

DATA_FOLDER_PATH = '../data/'

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %% [markdown]
# # Incidents Data

# %% [markdown]
# We load the dataset:

# %%
incidents_path = DATA_FOLDER_PATH + 'incidents.csv'
incidents_data = pd.read_csv(incidents_path, low_memory=False)

# %% [markdown]
# We assess the correct loading of the dataset printing the first 2 rows:

# %%
incidents_data.head(n=2)

# %% [markdown]
# This dataset contains information about gun incidents in the USA.
# 
# In the following table we provide the characteristics of each attribute of the dataset. To define the type of the attributes we used the categorization described by Pang-Ning Tan, Michael Steinbach and Vipin Kumar in the book *Introduction to Data Mining*. For each attribute, we also reported the desidered pandas dtype for later analysis.
# 
# | # | Name | Type | Description | Desired dtype |
# | :-: | :--: | :--: | :---------: | :-----------: |
# | 0 | date | Numeric (Interval) | Date of incident occurrence| datetime |
# | 1 | state | Categorical (Nominal) | Dtate where incident took place | object |
# | 2 | city_or_county | Categorical (Nominal) | City or county where incident took place | object |
# | 3 | address | Categorical (Nominal) | Address where incident took place | object |
# | 4 | latitude | Numeric (Interval) | Latitude of the incident | float64 |
# | 5 | longitude | Numeric (Interval) | Longitude of the incident | float64 |
# | 6 | congressional_district | Categorical (Nominal) | Congressional district where the incident took place | int64 |
# | 7 | state_house_district | Categorical (Nominal) | State house district | int64 |
# | 8 | state_senate_district | Categorical (Nominal) | State senate district where the incident took place | int64 |
# | 9 | participant_age1 | Numeric (Ratio) | Exact age of one (randomly chosen) participant in the incident | int64 |
# | 10 | participant_age_group1 | Categorical (Ordinal) | Exact age group of one (randomly chosen) participant in the incident | object |
# | 11 | participant_gender1 | Categorical (Nominal) | Exact gender of one (randomly chosen) participant in the incident | object |
# | 12 |min_age_participants | Numeric (Ratio) | Minimum age of the participants in the incident | int64 |
# | 13 | avg_age_participants | Numeric (Ratio) | Average age of the participants in the incident | float64 |
# | 14 | max_age_participants | Numeric (Ratio) | Maximum age of the participants in the incident | int64 |
# | 15 | n_participants_child | Numeric (Ratio) | Number of child participants 0-11 | int64 |
# | 16 | n_participants_teen | Numeric (Ratio) | Number of teen participants 12-17 | int64 |
# | 17 | n_participants_adult | Numeric (Ratio) | Number of adult participants (18 +) | int64 |
# | 18 | n_males | Numeric (Ratio) | Number of males participants | int64 |
# | 19 | n_females | Numeric (Ratio) | Number of females participants | int64 |
# | 20 | n_killed | Numeric (Ratio) | Number of people killed | int64 |
# | 21 | n_injured | Numeric (Ratio) | Number of people injured | int64 |
# | 22 | n_arrested | Numeric (Ratio) | Number of arrested participants | int64 |
# | 23 | n_unharmed | Numeric (Ratio) | Number of unharmed participants | int64 |
# | 24 | n_participants | Numeric (Ratio) | Number of participants in the incident | int64 |
# | 25 | notes | Categorical (Nominal) | Additional notes about the incident | object |
# | 26 | incident_characteristics1 | Categorical (Nominal) | Incident characteristics | object |
# | 27 | incident_characteristics2 | Categorical (Nominal) | Incident characteristics (not all incidents have two available characteristics) | object |

# %% [markdown]
# After analysing the dataset and the features it contains, it is immediately apparent that there are missing values, syntactic errors, semantic errors and unwanted outliers. We must then proceed to correct each feature present so that it is correct and consistent with the others present. To do this, we divide the features into groups that differ by semantic field and correct them in order of independence, starting with the most independent groups (such as date or geographical location) and then correcting the remaining, more specific ones, which must also be consistent with the corrected ones.

# %% [markdown]
# ## geographic consistency

# %% [markdown]
# The first group addressed concerns fields describing the geographical location of the recorded incident, and includes:
# 
# - state
# - city_or_county
# - address
# - latitude
# - longitude

# %%
data_check_consistency = pd.DataFrame()
data_check_consistency[['state', 'city_or_county', 'address', 'latitude', 'longitude']] = incidents_data[[
    'state', 'city_or_county', 'address', 'latitude', 'longitude']]

# %%
data_check_consistency.drop(['latitude', 'longitude'], axis=1).loc[[1, 2, 14, 19, 1595, 23900, 70906, 114746]]

# %% [markdown]
# An examination of these columns immediately reveals the following problems
# - the 'city_or_county' field represents the two different concepts of county and city where the incident took place, which is not effective because it is difficult to understand when the field refers to the former, the latter or both
# - The 'city_or_county' feature may contain non-unique references to the same abstract object, as is often the case in human language, which is full of synonyms, diminutives of names, etc.
# - The 'address' field contains information that is not precise or uniform, and it is often useless or complicated to understand its content.
# 
# Finally, the presence of errors or outliers is obvious, so that the need to check the consistency of the fields arises spontaneously, a task that is difficult to perform with a priori knowledge.

# %% [markdown]
# By using an external data source, obtained through the Geopy library, we can check the consistency of the data; by running a query on each record with coordinates, it is possible to have a detailed and uniform description of the dataset of the geographical reference where the incident occurred.

# %%
#TODO: sparare grafici a profusione per validare la tesi
# geopy data esample
geopy_sample = {
    "place_id": 327684232, 
    "licence": "Data \u00a9 OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright", 
    "osm_type": "way", 
    "osm_id": 437510561, 
    "lat": "39.832221333801186", 
    "lon": "-86.24921127905256", 
    "class": "highway", 
    "type": "secondary", 
    "place_rank": 26, 
    "importance": 0.10000999999999993, 
    "addresstype": "road", 
    "name": "Pike Plaza Road", 
    "display_name": "Pike Plaza Road, Indianapolis, Marion County, Indiana, 46254, United States", 
    "address": {"road": "Pike Plaza Road", 
                "city": "Indianapolis", 
                "county": "Marion County", 
                "state": "Indiana", 
                "ISO3166-2-lvl4": "US-IN", 
                "postcode": "46254", 
                "country": "United States", 
                "country_code": "us"}, 
    "boundingbox": ["39.8322034", "39.8324807", "-86.2492452", "-86.2487207"]}

# %% [markdown]
# GeoPy keys:
# 
# - place_id: unique numeric place identifier.
# 
# - licence: licence to use the geographic data.
# 
# - osm_type: type of OpenStreetMap (OSM) object the place belongs to ('node' for a point, 'way' for a road or 'relation' for a relation between elements).
# 
# - osm_id: unique identifier assigned to the OSM object.
# 
# - lat + lon: Latitude and longitude of the location.
# 
# - class: classification of the location (e.g. 'place').
# 
# - type: Classification of the location (e.g. 'city').
# 
# - place_rank: Rank or priority of the place in the geographical hierarchy (how important a place is).
# 
# - importance: Numerical value indicating the importance of the place in relation to other places.
# 
# - addresstype: type of address (e.g. 'house', 'street', 'postcode')
# 
# - name: name of place (e.g. name of town or street)
# 
# - display_name: user-readable representation of the location, often formatted as a full address.
# 
# - address: detailed address.
# 
# - boundingbox: list of four coordinates (latitude and longitude) that define a rectangle surrounding the location (this is an approximation of the area covered by the location).

# %% [markdown]
# Our intention is therefore to use the supporting dataset to verify the consistency of the data and to have unique references to states, cities and counties. Finally, we will integrate additional and potentially useful information for the future study, such as: the geographical importance and type of location, identified by: importance and address_type

# %%
# select only relevant columns from incidents_data
geo_data = incidents_data[['date', 'state', 'city_or_county', 'address', 'latitude', 'longitude',
       'congressional_district', 'state_house_district', 'state_senate_district']]
geo_data

# %% [markdown]
# Our comparison takes into account synonyms, diminutives and typing errors as much as possible, assuming that the data is mostly correct and that our check must therefore either confirm its correctness or identify errors that are not too complex to find.

# %%
#TODO caricare il dataset final e fare tutti i controlli del caso

# %% [markdown]
# ## other shit

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
incidents_data.info()

# %% [markdown]
# We notice that:
# - `congressional_district`, `state_house_district`, `state_senate_district`, `participant_age1`, `n_males`, `n_females`, `n_arrested`, `n_unharmed`, `n_participants` are stored as `float64` while should be `int64`
# - `min_age_participants`, `avg_age_participants`, `max_age_participants`, `n_participants_child`, `n_participants_teen`, `n_participants_adult` are stored as `object` while should be `int64`, this probably indicates the presence of out of syntactic errors (not in the domain)
# - the presence of missing values within many attributes; the only attributes without missing values are the following: `date`, `state`, `city_or_county`, `n_killed`, `n_injured`, `n_participants`
# 
# We cast the attributes to the correct type:

# %%
# NUMERIC ATTRIBUTES
# positive integers
incidents_data['participant_age1'] = pd.to_numeric(incidents_data['participant_age1'], downcast='unsigned', errors='coerce')
incidents_data['n_males'] = pd.to_numeric(incidents_data['n_males'], downcast='unsigned', errors='coerce')
incidents_data['n_females'] = pd.to_numeric(incidents_data['n_females'], downcast='unsigned', errors='coerce')
incidents_data['n_killed'] = pd.to_numeric(incidents_data['n_killed'], downcast='unsigned', errors='coerce')
incidents_data['n_injured'] = pd.to_numeric(incidents_data['n_injured'], downcast='unsigned', errors='coerce')
incidents_data['n_arrested'] = pd.to_numeric(incidents_data['n_arrested'], downcast='unsigned', errors='coerce')
incidents_data['n_unharmed'] = pd.to_numeric(incidents_data['n_unharmed'], downcast='unsigned', errors='coerce')
incidents_data['n_participants'] = pd.to_numeric(incidents_data['n_participants'], downcast='unsigned', errors='coerce')
incidents_data['min_age_participants'] = pd.to_numeric(incidents_data['min_age_participants'], downcast='unsigned', errors='coerce')
incidents_data['max_age_participants'] = pd.to_numeric(incidents_data['max_age_participants'], downcast='unsigned', errors='coerce')
incidents_data['n_participants_child'] = pd.to_numeric(incidents_data['n_participants_child'], downcast='unsigned', errors='coerce')
incidents_data['n_participants_teen'] = pd.to_numeric(incidents_data['n_participants_teen'], downcast='unsigned', errors='coerce')
incidents_data['n_participants_adult'] = pd.to_numeric(incidents_data['n_participants_adult'], downcast='unsigned', errors='coerce')
# float
incidents_data['avg_age_participants'] = pd.to_numeric(incidents_data['avg_age_participants'], errors='coerce')

# DATE
incidents_data['date'] = pd.to_datetime(incidents_data['date'], format='%Y-%m-%d')

# CATEGORICAL ATTRIBUTE
# nominal
incidents_data['congressional_district'] = incidents_data['congressional_district'].astype("category")
incidents_data['state_house_district'] = incidents_data['state_house_district'].astype("category")
incidents_data['state_senate_district'] = incidents_data['state_senate_district'].astype("category")
incidents_data['participant_gender1'] = incidents_data['participant_gender1'].astype("category")
# ordinal
incidents_data['participant_age_group1'] = incidents_data['participant_age_group1'].astype(
    pd.api.types.CategoricalDtype(categories = ["Child 0-11", "Teen 12-17", "Adult 18+"], ordered = True))

# %% [markdown]
# We display again information about the dataset to check the correctness of the casting and the number of missing values:

# %%
incidents_data.info()

# %% [markdown]
# And now we visualize missing values:

# %%
fig, ax = plt.subplots(figsize=(12,8)) 
sn.heatmap(incidents_data.isnull(), cbar=False, xticklabels=True, ax=ax)

# %% [markdown]
# TODO: commentare

# %% [markdown]
# We drop duplicates:

# %%
print(f"# of rows before dropping duplicates: {incidents_data.shape[0]}")
incidents_data.drop_duplicates(inplace=True)
print(f"# of rows after dropping duplicates: {incidents_data.shape[0]}")

# %% [markdown]
# We display descriptive statistics:

# %%
incidents_data.describe(include='all')

# %% [markdown]
# We notice that:
# - age min 0, max 311
# - ...

# %% [markdown]
# # Poverty Data

# %% [markdown]
# We load the dataset:

# %%
poverty_path = DATA_FOLDER_PATH + 'poverty_by_state_year.csv'
poverty_data = pd.read_csv(poverty_path)

# %% [markdown]
# We assess the correct loading of the dataset printing the first 2 rows:

# %%
poverty_data.head(n=2)

# %% [markdown]
# This dataset contains information about the poverty percentage for each USA state and year.
# 
# In the following table we provide the characteristics of each attribute of the dataset. To define the type of the attributes we used the categorization described by Pang-Ning Tan, Michael Steinbach and Vipin Kumar in the book *Introduction to Data Mining*. For each attribute, we also reported the desidered pandas dtype for later analysis.
# 
# | # | Name | Type | Description | Desired dtype |
# | :-: | :--: | :--: | :---------: | :------------: |
# | 0 | state | Categorical (Nominal) | Name of the state | object |
# | 1 | year | Numeric (Interval) | Year | int64 |
# | 2 | povertyPercentage | Numeric (Ratio) | Poverty percentage for the corresponding state and year | float64 |
# 

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
poverty_data.info()

# %% [markdown]
# We notice that:
# - the inferred types of the attributes are correct
# - the presence of missing values within the attribute `povertyPercentage`

# %% [markdown]
# We display descriptive statistics:

# %%
poverty_data.describe(include='all')

# %% [markdown]
# We notice that:
# - the data are provided also for the United States as a whole
# - `year` spans from 2004 to 2020

# %% [markdown]
# We check whether the tuple <`state`, `year`> uniquely identify each row:

# %%
poverty_data.groupby(['state', 'year']).size().max()==1

# %% [markdown]
# Since it does not, we display the duplicated <`state`, `year`> tuples:

# %%
poverty_state_year_size = poverty_data.groupby(['state', 'year']).size()
poverty_state_year_size[poverty_state_year_size>1]

# %% [markdown]
# We display the data for Wyoming, the only one with this issue:

# %%
poverty_data[(poverty_data['state']=='Wyoming')]

# %% [markdown]
# We notice that the entry relative to 2010 is missing. Since the other entries are ordered by year, we correct this error setting the year of the row with a povertyPercentage equal to 10.0 to 2010.

# %%
poverty_data.loc[
    (poverty_data['state'] == 'Wyoming') &
    (poverty_data['year'] == 2009) &
    (poverty_data['povertyPercentage'] == 10),
    'year'] = 2010

# %% [markdown]
# We check if each state has the expected number or rows:

# %%
(poverty_data.groupby('state').size()==(poverty_data['year'].max()-poverty_data['year'].min()+1)).all()

# %% [markdown]
# Since the tuple <`state`, `year`> uniquely identifies each row we can conclude that there are no missing rows.
# 
# Now, we count how many rows have missing values:

# %%
poverty_data[poverty_data['povertyPercentage'].isnull()].shape[0]

# %% [markdown]
# Given that there are 52 unique values for the `state` attribute, data for a specific year is probably missing. To check this, we list the years with missing values.

# %%
poverty_data[poverty_data['povertyPercentage'].isnull()]['year'].unique()

# %% [markdown]
# As expected we have no data from 2012. Later we will fix this issue.
# 
# Now we visualize the distribution of poverty percentage for each state.

# %%
poverty_data.boxplot(column='povertyPercentage', by='state', figsize=(20, 10), rot=90, xlabel='state', ylabel='Poverty (%)')
plt.suptitle('Poverty Percentage by State')
plt.title('')
plt.tight_layout()

# %% [markdown]
# This plot shows that Arkansas, Kentucky, Nebraska and North Dakota seems to be affected by fliers. We check this by plotting their poverty percentage over the years.

# %%
poverty_data[
    poverty_data['state'].isin(['Arkansas', 'Kentucky', 'Nebraska', 'North Dakota', 'United States'])
    ].pivot(index='year', columns='state', values='povertyPercentage').plot(kind='line')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Poverty (%)')
plt.title('Poverty (%) over the years')

# %% [markdown]
# The plot above shows that actually those values are not outliers. TODO: migliorare questo commento

# %%
poverty_data.groupby('year')['povertyPercentage'].mean().plot(kind='line', figsize=(15, 5), label='USA average', color='black', style='--')
plt.fill_between(
    poverty_data.groupby('year')['povertyPercentage'].mean().index,
    poverty_data.groupby('year')['povertyPercentage'].mean() - poverty_data.groupby('year')['povertyPercentage'].std(),
    poverty_data.groupby('year')['povertyPercentage'].mean() + poverty_data.groupby('year')['povertyPercentage'].std(),
    alpha=0.2,
    color='gray'
)

# %% [markdown]
# We now plot the average poverty percentage over the years for each state:

# %%
poverty_data.groupby(['state'])['povertyPercentage'].mean().sort_values().plot(kind='bar', figsize=(15, 5))
plt.title(f'Average Poverty (%) in the period {poverty_data.year.min()}-{poverty_data.year.max()}')
plt.xlabel('State')
plt.ylabel('Average Poverty (%)')

# %% [markdown]
# It is evident that New Hampshire's average poverty rate is markedly lower than that of the other states, whereas Mississippi's average poverty rate is notably higher than the rest. 
# 
# To inspect and compare the poverty percentage of each state over the year, we plot an interactive line chart:

# %%
fig = px.line(
    poverty_data.pivot(index='year', columns='state', values='povertyPercentage'),
    title='Poverty percentage in the US over the years')
fig.show()

# %% [markdown]
# We can oberserve that New Hampshire always had the lowest poverty percentage, whereas Mississippi had the highest till 2009, then it was surpassed by New Mexico and Louisiana.

# %% [markdown]
# TODO: spostare in preparation.
# 
# To imputate the missing data from 2012, we calculate the average of the `povertyPercentage` values for the preceding and succeeding year.

# %%
poverty_perc_2012 = poverty_data[poverty_data['year'].isin([2011, 2013])].groupby(['state'])['povertyPercentage'].mean()
poverty_data['povertyPercentage'] = poverty_data.apply(
    lambda x: poverty_perc_2012[x['state']] if x['year']==2012 else x['povertyPercentage'], axis=1
)

# %% [markdown]
# Now we plot again the interactive line chart:

# %%
fig = px.line(
    poverty_data.pivot(index='year', columns='state', values='povertyPercentage'),
    title='Poverty percentage in the US over the years')
fig.show()

# %% [markdown]
# We also visualize how the poverty percentage changed with an animated map:

# %%
poverty_data.sort_values(by=['state', 'year'], inplace=True)
poverty_data['px_code'] = poverty_data['state'].map(usa_code) # retrieve the code associated to each state (the map is defined in the file utils.py)
fig = px.choropleth(
    poverty_data[poverty_data['state']!='United States'],
    locations='px_code',
    locationmode="USA-states",
    color='povertyPercentage',
    color_continuous_scale="rdbu",
    range_color=(
        min(poverty_data[poverty_data['state']!='United States']['povertyPercentage']),
        max(poverty_data[poverty_data['state']!='United States']['povertyPercentage'])),
    scope="usa",
    animation_frame='year',
    title="US Poverty Percentage over the years",
    hover_name='state',
    hover_data={'px_code': False}
)
fig.update_layout(
    title_text='US Poverty Percentage over the years',
    coloraxis_colorbar_title_text = 'Poverty (%)'
)
fig.show()

# %%
# TODO: usare unica color bar e aggiustare dimensioni in modo che si leggano gli stati (questa versione potrebbe servire per il report)

fig, axs = plt.subplots(ncols=3, nrows=6, figsize=(30, 40))
vmin, vmax = poverty_data['povertyPercentage'].agg(['min', 'max'])

row_count = 0
col_count = 0
for year in poverty_data['year'].unique():
    plot_usa_map(
        poverty_data[poverty_data['year']==year],
        col_to_plot='povertyPercentage',
        ax=axs[row_count][col_count],
        state_col='state',
        vmin=vmin,
        vmax=vmax,
        title=str(year),
        cbar_title="Poverty (%)",
        cmap='RdBu',
        borders_path="../cb_2018_us_state_500k"
    )
    col_count += 1
    if col_count == 3:
        col_count = 0
        row_count += 1

fig.delaxes(axs[5][2])
fig.suptitle("Povery percentage over the years", fontsize=25)
fig.tight_layout()

# %% [markdown]
# # Elections Data

# %% [markdown]
# We load the dataset:

# %%
elections_path = DATA_FOLDER_PATH + 'year_state_district_house.csv'
elections_data = pd.read_csv(elections_path)

# %% [markdown]
# We assess the correct loading of the dataset printing the first 2 rows:

# %%
elections_data.head(n=2)

# %% [markdown]
# This dataset contains information about the winner of the congressional elections in the USA, for each year, state and congressional district.
# 
# In the following table we provide the characteristics of each attribute of the dataset. To define the type of the attributes we used the categorization described by Pang-Ning Tan, Michael Steinbach and Vipin Kumar in the book *Introduction to Data Mining*. For each attribute, we also reported the desidered pandas `dtype` for later analysis.
# 
# | # | Name | Type | Description | Desired dtype |
# | :-: | :--: | :--: | :---------: | :------------: |
# | 0 | year | Numeric (Interval) | Year | int64 |
# | 1 | state | Categorical (Nominal) | Name of the state | object |
# | 2 | congressional_district | Categorical (Nominal) | Congressional district | int64 |
# | 3 | party | Categorical (Nominal) | Winning party fort the corresponding congressional_district in the state, in the corresponding year | object |
# | 4 | candidateVotes | Numeric (Ratio) | Number of votes obtained by the winning party in the corresponding election | int64 |
# | 5 | totalVotes | Numeric (Ratio)| Number total votes for the corresponding election | int64 |

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
elections_data.info()

# %% [markdown]
# We notice that:
# - the inferred types are correct
# - there are no missing values (however, we should still assess whether there are any missing rows for specific years, states, or congressional districts)

# %% [markdown]
# We display descriptive statistics:

# %%
elections_data.describe(include='all')

# %% [markdown]
# We notice that:
# - year spans from 2004 to 2020
# - there are 6 unique parties
# - the minimum of candidatevotes and totalvotes are negative numbers, meaning that there are actually missing values

# %% [markdown]
# First we check if the triple <`year`, `state`, `congressional_district`> uniquely identifies each row:

# %%
elections_data.groupby(['year', 'state', 'congressional_district']).size().max() == 1

# %% [markdown]
# Then, we check if `candidatevotes` are always less or equal than `totalvotes`:

# %%
elections_data[elections_data['candidatevotes'] <= elections_data['totalvotes']].size == elections_data.size

# %% [markdown]
# We list the unique values in the column `state`:

# %%
states = elections_data['state'].unique()
states.sort()
print(f'States: {states}')
print(f'Number of states: {states.size}')

# %% [markdown]
# All the states (District og Columbia included) are present.
# 
# We now display the states and the years for which there are missing rows:

# %%
years = [i for i in range(elections_data['year'].min(), elections_data['year'].max(), 2)]
for year in years:
    for state in states:
        if elections_data[(elections_data['state']==state) & (elections_data['year']==year)].size == 0:
            print(f"No data for '{state}' in {year}")

# %% [markdown]
# Except for District of Columbia, there are no missing rows.
# For District of Columbia we have only the following row:

# %%
elections_data[elections_data['state']=='DISTRICT OF COLUMBIA']

# %% [markdown]
# Missing values are probably due to the fact that District of Columbia is a non voting delegate district. Anyway, we gathered the missing values from Wikipedia. We noticed that as for the 2020 elecetions, the number of votes received by the winning party coincides, but the number of totalvotes is different (see [here](https://en.wikipedia.org/wiki/2020_United_States_House_of_Representatives_election_in_the_District_of_Columbia)). To be consistent with the other data, we replace the totalvotes value from 2020 with the one from Wikipedia.
# 
# Now we import those data:

# %%
dc_elections_data = pd.read_csv('../data/wikipedia/district_of_columbia_house.csv')
dc_elections_data.head(n=2)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
dc_elections_data.info()

# %% [markdown]
# The inferred types are correct.
# 
# We now merge the two dataframes:

# %%
elections_data.drop(elections_data[elections_data['state']=='DISTRICT OF COLUMBIA'].index, inplace=True)
elections_data = pd.concat([elections_data, dc_elections_data], ignore_index=True)
elections_data.sort_values(by=['year', 'state', 'congressional_district'], inplace=True, ignore_index=True)

# %% [markdown]
# We now check if congressional districts are numbered correctly (with '0' for states with only one congressional district, or with incremental values starting from '1' otherwise):

# %%
correct_numbering = True
for state in states:
    if state == 'DISTRICT OF COLUMBIA':
        continue
    for year in years:
        districts = elections_data[(elections_data['state']==state) & (elections_data['year']==year)]['congressional_district'].unique()
        districts.sort()
        if districts.size > 1:
            if (districts != [i for i in range(1, districts.size+1)]).any():
                correct_numbering = False
                break
        elif districts[0] != 0:
            correct_numbering = False
            break
correct_numbering

# %% [markdown]
# We now plot the distribution of `totalvotes` for each state in the years of interest, excluding 0 and negative values (this plot makes sense because congressional districts are redrawn so that the population of each district is roughly equal):

# %%
elections_data[
    (elections_data['totalvotes']>0)&(elections_data['year']>2012)
].boxplot(column='totalvotes', by='state', figsize=(20, 10), rot=90, xlabel='State', ylabel='Total votes')
plt.suptitle('Total votes from 2014')
plt.title('')
plt.tight_layout()

# %% [markdown]
# We can observe that for both total and candidate votes Florida, Louisian and Oklahoma have lower outliers, while Maine has an upper outlier. 
# 
# We display the rows relative to Maine:

# %%
elections_data[(elections_data['year']>2013) & (elections_data['state']=='MAINE')]

# %% [markdown]
# We found in [Wikipedia](https://en.wikipedia.org/wiki/2022_United_States_House_of_Representatives_elections_in_Maine) that in Maine, that year, the Democratic party received 165136 votes out of a total of 311278 votes. We correct the error:

# %%
elections_data.loc[
    (elections_data['state']=='MAINE') &
    (elections_data['year']==2022) &
    (elections_data['congressional_district']==2),
    'candidatevotes'] = 165136
elections_data.loc[
    (elections_data['state']=='MAINE') &
    (elections_data['year']==2022) &
    (elections_data['congressional_district']==2),
    'totalvotes'] = 311278

# %%
elections_data[
    (elections_data['year']>2013) &
    (elections_data['state'].isin(['FLORIDA', 'LOUSIANA', 'OKLAHOMA'])) &
    ((elections_data['candidatevotes']<100) | (elections_data['totalvotes']<100))
]

# %% [markdown]
# We found in Wikipedia (e.g. [here](https://en.wikipedia.org/wiki/2014_United_States_House_of_Representatives_elections_in_Florida)), that for all the years and states above, no candidates filed to challenge the incumbent representative for their seat. Therefore, we will copy the `candidatevotes` and `totalvotes` values from the previous year:

# %%
for index, row in elections_data.iterrows():
    if row['candidatevotes'] < 2:
        replacement = elections_data[
            (elections_data['year']==row['year']-2) &
            (elections_data['state']==row['state']) &
            (elections_data['congressional_district']==row['congressional_district'])
        ]
        if replacement.size > 0:
            elections_data.at[index, 'candidatevotes'] = replacement['candidatevotes'].iloc[0]
            elections_data.at[index, 'totalvotes'] = replacement['totalvotes'].iloc[0]

# %% [markdown]
# We now plot the distribution of `totalvotes` (summing over the districts) after cleaning the data:

# %%
elections_data[
    elections_data['year']>2012
].groupby(['year', 'state']).agg('sum').boxplot(column='totalvotes', by='state', figsize=(20, 10), rot=90, xlabel='State', ylabel='Total votes')
plt.suptitle('Total votes from 2014')
plt.title('')
plt.tight_layout()

# %% [markdown]
# It is evident that in some states the number of votes fluctuates significantly from year to year.
# 
# We get the unique names of the parties for the years of interest:

# %%
elections_data[
    (elections_data['year']>2012)
]['party'].unique()

# %% [markdown]
# The Democratic Farmer Labor is the affiliate of the Democratic Party in the U.S. state of Minnesota [[Wikipedia](https://en.wikipedia.org/wiki/Minnesota_Democratic–Farmer–Labor_Party)], hence we replace this party name with 'DEMOCRATIC' to ease later analysis.

# %%
elections_data['party'] = elections_data['party'].apply(
    lambda x: 'DEMOCRAT' if x=='DEMOCRATIC-FARMER-LABOR' else x
)

# %% [markdown]
# We now compute the percentage of votes obtained by the winner party and we plot the distribution of these percentages for the years of interest:

# %%
elections_data['candidateperc'] = (elections_data['candidatevotes']/elections_data['totalvotes'])*100
elections_data[elections_data['year']>2012]['candidateperc'].plot.hist(bins=50, figsize=(10, 5), title='Percentage of winner votes')

# %% [markdown]
# It seems that in some districts the winner party obtained 100% of the votes. We disaply those districts:

# %%
elections_data[(elections_data['candidateperc']==100) & (elections_data['year']>2012)]

# %% [markdown]
# Wikipedia reports the same data, in those cases there was not an opponent party.
# 
# The histogram above also shows that in some disticts the winner party obtained less than 50% of the votes. We display those districts:

# %%
elections_data[(elections_data['candidateperc']<=50) & (elections_data['year']>2012)] # TODO: Connecticut 5, 2014  is wrong, Louisiana 5, 2014 is wrong...

# %%
elections_data[(elections_data['candidateperc']<=30) & (elections_data['year']>2012)]

# %% [markdown]
# Now we compute, for each year and state, the party with the highest percentage of votes, so to have a better understanding of the political orientation of each state:

# %%
winning_party_per_state = elections_data.groupby(['year', 'state', 'party'])['candidateperc'].mean()
winning_party_per_state = winning_party_per_state.groupby(['year', 'state']).idxmax().apply(lambda x: x[2])
winning_party_per_state

# %% [markdown]
# We can now merge the dataframe with the poverty data with the one with the aggregated election data:

# %%
poverty_data['state'] = poverty_data['state'].apply(lambda x: x.upper())
poverty_elections_data = pd.merge(
    poverty_data,
    winning_party_per_state.to_frame(name='winningparty'),
    how='inner',
    left_on=['year', 'state'],
    right_on=['year', 'state']
)
poverty_elections_data

# %%
poverty_elections_data[poverty_elections_data['state']=='VERMONT']

# %% [markdown]
# We now plot on a map the winning party over the years:

# %%
fig = px.choropleth(
    poverty_elections_data[poverty_elections_data['year']>2004],
    locations='px_code',
    locationmode="USA-states",
    color='winningparty',
    scope="usa",
    animation_frame='year',
    title="Results of the elections over the years", 
    hover_name='state',
    hover_data={'px_code': False}
)
fig.update_layout(
    legend_title_text='Party'
)
fig.show()

# %% [markdown]
# We display the correlation matrix between the attributes:

# %%
poverty_elections_data['winningparty'] = poverty_elections_data['winningparty'].astype('category').cat.codes
corr = poverty_elections_data.corr()
corr.style.background_gradient(cmap='coolwarm')

# %% [markdown]
# No correlation is evident.


