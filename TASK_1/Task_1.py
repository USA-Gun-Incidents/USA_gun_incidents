# -*- coding: utf-8 -*-
# %% [markdown]
# # Task 1 Data Understanding and Preparation

# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import math
import os
import sys
import calendar
sys.path.append(os.path.abspath('..')) # TODO: c'è un modo per farlo meglio?
from plot_utils import *
from sklearn.neighbors import KNeighborsClassifier
from geopy import distance as geopy_distance
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from pyproj import Transformer

# %% [markdown]
# We define constants and settings for the notebook:

# %%
# %matplotlib inline

DATA_FOLDER_PATH = '../data/'

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

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
# The plot above shows that actually those values are not errors.

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
poverty_data['px_code'] = poverty_data['state'].map(usa_code) # retrieve the code associated to each state (the map is defined in the file data_preparation_utils.py)
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

# fig, axs = plt.subplots(ncols=3, nrows=6, figsize=(30, 40))
# vmin, vmax = poverty_data['povertyPercentage'].agg(['min', 'max'])

# row_count = 0
# col_count = 0
# for year in poverty_data['year'].unique():
#     plot_usa_map(
#         poverty_data[poverty_data['year']==year],
#         col_to_plot='povertyPercentage',
#         ax=axs[row_count][col_count],
#         state_col='state',
#         vmin=vmin,
#         vmax=vmax,
#         title=str(year),
#         cbar_title="Poverty (%)",
#         cmap='RdBu',
#         borders_path="../cb_2018_us_state_500k"
#     )
#     col_count += 1
#     if col_count == 3:
#         col_count = 0
#         row_count += 1

# fig.delaxes(axs[5][2])
# fig.suptitle("Povery percentage over the years", fontsize=25)
# fig.tight_layout()

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
# | 4 | candidatevotes | Numeric (Ratio) | Number of votes obtained by the winning party in the corresponding election | int64 |
# | 5 | totalvotes | Numeric (Ratio)| Number total votes for the corresponding election | int64 |

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
# All the states (District of Columbia included) are present.
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

# %%
hist_box_plot(elections_data[elections_data['year']>2012], col='candidateperc', title='Percentage of winner votes')

# %% [markdown]
# It seems that in some districts the winner party obtained 100% of the votes. We disaply those districts:

# %%
elections_data[(elections_data['candidateperc']==100) & (elections_data['year']>2012)]

# %% [markdown]
# Wikipedia reports the same data, in those cases there was not an opponent party.
#
# The histogram above also shows that in some disticts the winner party obtained less than 50% of the votes. We display those districts:

# %%
elections_data[(elections_data['candidateperc']<=50) & (elections_data['year']>2012)] # TODO: maybe some are wrong

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
# We display a concise summary of the DataFrame:

# %%
incidents_data.info()

# %% [markdown]
# We notice that:
# - `congressional_district`, `state_house_district`, `state_senate_district`, `participant_age1`, `n_males`, `n_females`, `n_arrested`, `n_unharmed`, `n_participants` are stored as `float64` while should be `int64`
# - `min_age_participants`, `avg_age_participants`, `max_age_participants`, `n_participants_child`, `n_participants_teen`, `n_participants_adult` are stored as `object` while should be `int64`, this probably indicates the presence of out of syntactic errors (not in the domain)
# - the presence of missing values within many attributes; the only attributes without missing values are the following: `date`, `state`, `city_or_county`, `n_killed`, `n_injured`, `n_participants`
#
# We display descriptive statistics of the DataFrame so to better understand how to cast the data:

# %%
incidents_data.describe(include='all')

# %% [markdown]
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
# (the following attributes should be categorical, but for convenience we keep them numeric)
incidents_data['congressional_district'] = pd.to_numeric(incidents_data['congressional_district'], downcast='unsigned', errors='coerce')
incidents_data['state_house_district'] = pd.to_numeric(incidents_data['state_house_district'], downcast='unsigned', errors='coerce')
incidents_data['state_senate_district'] = pd.to_numeric(incidents_data['state_senate_district'], downcast='unsigned', errors='coerce')
# float
incidents_data['avg_age_participants'] = pd.to_numeric(incidents_data['avg_age_participants'], errors='coerce')

# DATE FIX: dopo?
incidents_data['date'] = pd.to_datetime(incidents_data['date'], format='%Y-%m-%d')

# CATEGORICAL ATTRIBUTES
# nominal
incidents_data['participant_gender1'] = incidents_data['participant_gender1'].astype("category")
# ordinal
incidents_data['participant_age_group1'] = incidents_data['participant_age_group1'].astype(
    pd.api.types.CategoricalDtype(categories = ["Child 0-11", "Teen 12-17", "Adult 18+"], ordered = True))

# %% [markdown]
# We display again information about the dataset to check the correctness of the casting and the number of missing values:

# %%
incidents_data.info()

# %% [markdown]
# We observe that the downcasting of many attributes has not succeeded. This is due to the presence of missing or out of range values. TODO: to handle
#
# Now we visualize missing values:

# %%
fig, ax = plt.subplots(figsize=(12,8)) 
sns.heatmap(incidents_data.isnull(), cbar=False, xticklabels=True, ax=ax)

# %% [markdown]
# We observe that:
# - The following attributes are missing together:
#     - `latitude` and `longitude`
#     - `n_participants_child`, `n_participants_teen`, `n_participants_adult`
#     - `n_males`, `n_females`
#     - `n_arrested`, `n_unharmed`
# - There are many missing values for the following attributes:
#     - `participant_age1`
#     - `min_age_participants`, `avg_age_participants`, `max_age_participants` (often missing together but not always)
#     - `notes`
#     - `incident_characteristics2`
# - Often `participant_age1` is missing but `participant_age_group1` is not and the same holds for `state_house_district` w.r.t `state_senate_district`.
# - `latitude` and `longitude` are often available and could be used to recover the missing values of `address`, `congressional_district`, `state_house_district` and `state_senate_district` (using external libraries that offer this service).

# %% [markdown]
# We drop duplicates:

# %%
print(f"# of rows before dropping duplicates: {incidents_data.shape[0]}")
incidents_data.drop_duplicates(inplace=True)
print(f"# of rows after dropping duplicates: {incidents_data.shape[0]}")

# %% [markdown]
# We display descriptive statistics:

# %%
incidents_data.describe(include='all', datetime_is_numeric=True)

# %% [markdown]
# We can already make some considerations about the dataset:
# - incidents happened in 51 different states (we probably have at least one incident for each state)
# - the most frequent value for the attrbute `state` is Illinois and the most frequent value for  `city_or_county` is Chicago (which is in illinois, it is consistent)
# - 148 incidents happened at the address "2375 International Pkwy" (an airport in Dallas, Texsas)
# - the majority of incidents involved males
# - there are 52 unique values for the attribute `incidents_characteristics1` and the most frequent is "Shot - Wounded/Injured" (at the time of data collection, it is likely that the values this attribute could take on were limited to a predefined set)
# - there are 90 unique values for the attribute `incidents_characteristicsch2` and the most frequent is "Officer Involved Incident"; this attribute presents more missing values than `incidents_characteristics1` (at the time of data collection, it is likely that the values this attribute could take on were limited to a predefined set)
# - the most frequent value for the attribute notes is "man shot", but the number of unique values this attribute assumes is very high (at the time of data collection the values this attribute could take on were not defined)
# - there are many inconsistencies and/or erros, for example:
#     - the maximum value for the attribute `date` is 2030-11-28
#     - the range of the attributes `age`, `min_age_participants`, `avg_age_participants`, `max_age_participants`, `n_participants_child`, `n_participants_teen`, `n_participants_adult` is outside the domain of the attributes (e.g. the maximum value for the attribute age is 311)

# %% [markdown]
# To avoid having to recompute the data every time the kernel is interrupted and to make the results reproducible in a short execution time, we decided to save the data to CSV files at the end of each data preparation phase.
#
# Below, we provide two specific functions to perform this task.

# %%
# FIX: spostare
LOAD_DATA_FROM_CHECKPOINT = True # boolean: True if you want to load data, False if you want to compute it
CHECKPOINT_FOLDER_PATH = '../data/checkpoints/'

def checkpoint(df, checkpoint_name):
    df.to_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv')

def load_checkpoint(checkpoint_name, casting={}):
    if casting:
        return pd.read_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv', low_memory=False, index_col=0, parse_dates=['date'], dtype=casting)
    else: #TODO: sistemare il casting quando ci sono tutte le colonne 
        return pd.read_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv', low_memory=False, index_col=0)#, parse_dates=['date'])

# %% [markdown]
# ## Date attribute: exploration and preparation

# %% [markdown]
# We plot the distribution of the dates using different binning strategies:

# %%
def plot_dates(df_column, title=None, color=None):
    n_items = len(df_column.index)
    min = df_column.min()
    max = df_column.max()

    fig, axs = plt.subplots(3, sharex=True, figsize=(14, 6))
    fig.suptitle(title)

    # one bin per month
    n_bin = int((max - min).days / 30)
    axs[0].hist(df_column, bins=n_bin, density=True, color=color)
    axs[0].set_ylabel("One bin per month")
    axs[0].grid(axis='y')

    # number of bins computed using Sturge's rule
    n_bin = int(1 + math.log2(n_items))
    axs[1].hist(df_column, bins=n_bin, density=True, color=color)
    axs[1].set_ylabel("Sturge\'s rule binning")
    axs[1].grid(axis='y')

    axs[2].boxplot(x=mdates.date2num(df_column), labels=[''], vert=False)
    axs[2].set_xlabel('date')

plot_dates(incidents_data['date'], title='Dates distribution')
print('Range data: ', incidents_data['date'].min(), ' - ', incidents_data['date'].max())
num_oor = incidents_data[incidents_data['date'].dt.year>2018].shape[0]
print(f'Number of rows with out of range value for the attribute date: {num_oor} ({num_oor/incidents_data.shape[0]*100:.2f}%)')

# %% [markdown]
# These plots show that the number of incidents with an out of range value for the attribute date is non negligible (9.6%) and, excluding these points, there are no incidents happened after the year 2018.
# Instead of discarding rows with out-of-range dates, we will correct the errors to prevent excessive data loss.
# Since there are no other features that could suggest the timeframe of the incident, we can only proceed using one of the following approaches:
# - check if those records have duplicates with a correct date
# - suppose dates were entered manually using a numeric keypad and that the errors are typos (e.g. 2030 is actually 2020)
# - replace the errors with the mean or median value
#
# Let's check if there are duplicates with a correct date:

# %%
incidents_future = incidents_data[incidents_data['date'].dt.year>2018].drop(columns=['date'])
incidents_past = incidents_data[incidents_data['date'].dt.year<2019].drop(columns=['date'])
incidents_past[incidents_past.isin(incidents_future).any(axis=1)].size!=0

# %% [markdown]
# Since there are no duplicates, we proceed with the second and third approach:

# %%
incidents_data['year'] = incidents_data['date'].dt.year
mean_date = incidents_data[incidents_data['year']<2019]['date'].mean()
median_date = incidents_data[incidents_data['year']<2019]['date'].median()

incidents_data['date_minus10'] = incidents_data['date']
incidents_data['date_minus10'] = incidents_data['date'].apply(lambda x : x - pd.DateOffset(years=10) if x.year>2018 else x)
incidents_data['date_minus11'] = incidents_data['date']
incidents_data['date_minus11'] = incidents_data['date'].apply(lambda x : x - pd.DateOffset(years=11) if x.year>2018 else x)
incidents_data['date_mean'] = incidents_data['date']
incidents_data['date_mean'] = incidents_data['date'].apply(lambda x : mean_date if x.year>2018 else x)
incidents_data['date_mean'] = pd.to_datetime(incidents_data['date_mean'], format='%Y-%m-%d') # discard hours, minutes and seconds
incidents_data['date_median'] = incidents_data['date']
incidents_data['date_median'] = incidents_data['date'].apply(lambda x : median_date if x.year>2018 else x)

# %%
plot_dates(incidents_data['date_minus10'], 'Dates distribution (year - 10 for oor)')
plot_dates(incidents_data['date_minus11'], 'Dates distribution (year - 11 for oor)', color='orange')
plot_dates(incidents_data['date_mean'], 'Dates distribution (oor replaced with mean)', color='green')
plot_dates(incidents_data['date_median'], 'Dates distribution (oor replaced with median)', color='red')

# %% [markdown]
# Unfortunately, these methods lead to unsatisfactory results, as they all remarkably alter the distribution. Therefore, we will keep the errors and take them into account in subsequent analyses. 

# %% [markdown]
# ## Geospatial features: exploration and preparation

# %% [markdown]
# Columns of the dataset are considered in order to verify the correctness and consistency of data related to geographical features:
# - *state*
# - *city_or_county*
# - *address*
# - *latitude*
# - *longitude*

# %% [markdown]
# Plot incidents' location on a map:

# %%
fig = px.scatter_mapbox(
    lat=incidents_data['latitude'],
    lon=incidents_data['longitude'],
    zoom=0, 
    height=500,
    width=800
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %% [markdown]
# We note that, excluding 7923 entries in the dataset where latitude and longitude data are missing and 4 entries outside the borders of the USA, the remaining dataset entries have latitude and longitude values. The *state* field is non-null for all entries.
#
# Therefore, we decided to use external sources to verify the data consistency in the dataset. For the fields where latitude and longitude were present within the United States, we utilized GeoPy data, while for the remaining data and those that did not have a positive match with GeoPy's results, we used data from Wikipedia.
#
# In the following sections of the notebook, we provide a summary of the additional data used and the methodologies for verifying data consistency.

# %% [markdown]
# ### Additional dataset: GeoPy

# %% [markdown]
# In order to check the data consistency of our dataset, we use [GeoPy](https://geopy.readthedocs.io/en/stable/). We have previously saved the necessary data for performing these checks using the latitude and longitude data from our dataset.

# %% [markdown]
# We load GeoPy data from the CSV file where it is stored.

# %%
geopy_path = os.path.join(DATA_FOLDER_PATH, 'geopy/geopy.csv')
geopy_data = pd.read_csv(geopy_path, index_col=['index'], low_memory=False, dtype={})

# %% [markdown]
# geopy_data is a data frame containing one row for each row in our dataset, with matching indices.

# %% [markdown]
# **GeoPy Keys** we saved and used:
#
# - *lat* and *lon*: Latitude and longitude of the location.
#
# - *importance*: Numerical value $\in [0,1]$, indicates the importance of the location relative to other locations.
#
# - *addresstype*: Address type (e.g., "house," "street," "postcode") to classify the incident's place.
#
# - *address*: A dictionary containing detailed address information. \
#     Dictionary keys included in the data frame are: *state*, *county*, *suburb*, *city*, *town*, *village*.
#
# - *display_name*: User-friendly representation of the location, often formatted as a complete address. Used by us to cross-reference with the address in case we are unable to find a match between our data and the GeoPy data set using other information from the address.
#
# Additional columns we added to the dataframe:
#
# - *coord_presence*: Boolean, false if the corresponding row in the original dataset did not have latitude and longitude values, making it impossible to query GeoPy.

# %%
print('Number of rows without surburbs: ', geopy_data.loc[geopy_data['suburb'].isna()].shape[0])
display(geopy_data.loc[geopy_data['suburb'].isna()].head(2))

# %%
print('Number of rows without coordinates: \n', geopy_data['coord_presence'].value_counts())
print('\nNumber of rows without importance: \n', geopy_data['importance'].isnull().value_counts())

# %%
print('Number of rows in which city is null and town is not null: ', 
    geopy_data[(geopy_data['city'].isnull()) & (geopy_data['town'].notnull())].shape[0])

# %%
print(geopy_data['addresstype'].unique())
print('Number of rows in which addresstype is null: ', geopy_data[geopy_data['addresstype'].isnull()].shape[0])

# %% [markdown]
# ### Additional dataset: Wikipedia

# %% [markdown]
# We also downloaded data from [Wikipedia](https://en.wikipedia.org/wiki/County_(United_States)) containing the states and counties (or the equivalent) for each state in the USA. This data was used in cases where no consistency was found with GeoPy data.
#
# This dataset made it possible to verify the data consistency for the *state* and *county* fields without the need for *latitude* and *longitude* values.

# %%
counties_path = os.path.join(DATA_FOLDER_PATH, 'wikipedia/counties.csv')

additional_data = pd.read_csv(counties_path)
additional_data.head()

# %%
additional_data.dtypes

# %% [markdown]
# ### Studying Data Consistency

# %% [markdown]
# We concatenate geographic data from our dataset and GeoPy data into a single DataFrame.

# %%
data_check_consistency = pd.DataFrame(columns=['state', 'city_or_county', 'address', 'latitude', 'longitude', 'display_name', 
    'village_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy', 'importance_geopy', 'addresstype_geopy', 
    'coord_presence', 'suburb_geopy'])
data_check_consistency[['state', 'city_or_county', 'address', 'latitude', 'longitude']] = incidents_data[[
    'state', 'city_or_county', 'address', 'latitude', 'longitude']]
    
data_check_consistency[['address_geopy', 'village_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy', 
    'importance_geopy', 'addresstype_geopy', 'coord_presence', 'suburb_geopy']] = geopy_data.loc[incidents_data.index][['display_name', 'village', 'town', 'city', 
    'county', 'state', 'importance', 'addresstype', 'coord_presence', 'suburb']]



# %%
data_check_consistency.head(2)

# %% [markdown]
# Convert latitude and longitude to float type.

# %%
# convert latitude and longitude to float
data_check_consistency['latitude'] = data_check_consistency['latitude'].astype(float)
data_check_consistency['longitude'] = data_check_consistency['longitude'].astype(float)

# %% [markdown]
# Print the first 10 occurrences of city_or_county values where parentheses are present in the string:

# %%
[c for c in data_check_consistency['city_or_county'].unique() if '(' in c][:10]

# %%
len([c for c in data_check_consistency['city_or_county'].unique() if '(' in c and 'county' not in c])

# %% [markdown]
# We can note that in 1782 entries, both city and county values are present in city_or_county, so we need to take this into consideration.

# %% [markdown]
# We created some functions to check the consistency of geographic data using the external sources we mentioned earlier. We also replaced the values for State, County, and City with a string in title case, without punctuation or numbers, to obtain a dataset of clean and consistent data.
#
# Below, we provide a brief summary of all the functions used to check data consistency and replace values when necessary.

# %% [markdown]
# **String Preprocessing**
# - Initially, we convert the values for state, county, and city to lowercase for both our dataset and external dataset.
# - For cases where the *city_or_county* field contained values for both city and county, we split the string into two and used both new strings to match with both county and city.
# - We removed the words 'city of' and 'county' from the strings in the *city_or_county* field.
# - We removed punctuation and numerical values from the string, if present, in the *state* and *city_or_county* fields.
#
# **If We Had GeoPy Data**
# - We attempted direct comparisons with GeoPy data:
#     - Our data's *state* was compared with *state* from GeoPy.
#     - To assign the value of *county*, we compared with *county_geopy* and with *suburb_geopy*.
#     - The *city* was compared with *city_geopy*, *town_geopy* and *village_geopy*.
# - If direct comparisons were not positive:
#     - We checked for potential typos in the string using the Damerau-Levenshtein distance (definition below).
#     - Thresholds to decide the maximum distance for two strings to be considered equal were set after several preliminary tests. We decided to use different thresholds for state and city/county.
# - In cases where previous comparisons were not sufficient, we also used the *address* field from our dataset, comparing it with GeoPy's *display_name*, from which commonly occurring words throughout the column were removed (e.g., "Street," "Avenue," "Boulevard"). Again, we used the Damerau-Levenshtein distance with an appropriate threshold to verify address consistency.
#
# In cases where we were able to evaluate data consistency through these comparisons, we set the values for the fields *state*, *county*, *city*, *latitude*, *longitude*, *importance*, *address_type* using GeoPy values. Additionally, we also saved values reflecting the consistency with the fields evaluated earlier in: *state_consistency*, *county_consistency*, *address_consistency* (0 if not consistent, 1 if consistent, -1 if null values are presents)
#
# If the fields in our dataset were not consistent through the previously described checks or could not be verified due to the absence of latitude and longitude values, we attempted to assess consistency using Wikipedia data, with similar checks as before. In this case, we could only retrieve the *state* and *county* fields.

# %% [markdown]
# General formula for calculating the **Damerau-Levenshtein distance** between two strings $s$ and $t$ \
# $D(i, j) = \min
# \begin{cases}
# D(i-1, j) + 1 \\
# D(i, j-1) + 1 \\
# D(i-1, j-1) + \delta \\
# D(i-2, j-2) + \delta & \text{if } s[i] = t[j] \text{ and } s[i-1] = t[j-1]
# \end{cases}$
#
# where:
# - $D(i, j)$ is the Damerau-Levenshtein distance between the first $i$ letters of a string $s$ and the first $j$ letters of a string $t$.
# - $\delta$ is 0 if the current letters $s[i]$ and $t[j]$ are equal, otherwise, it is 1.
# - $D(i-2, j-2) + \delta$ represents transposition (swapping two adjacent letters) if the current letters $s[i]$ and $t[j]$ are equal, and the preceding letters $s[i-1]$ and $t[j-1]$ are also equal.
#
#

# %%
from data_preparation_utils import check_geographical_data_consistency

if LOAD_DATA_FROM_CHECKPOINT: # load data
    clean_geo_data = load_checkpoint('checkpoint_geo_temporary')
else: # compute data
    clean_geo_data = data_check_consistency.apply(lambda row: check_geographical_data_consistency(row, 
        additional_data=additional_data), axis=1)
    checkpoint(clean_geo_data, 'checkpoint_geo_temporary') # save data

# %% [markdown]
# ### Visualize Consistent Geographical Data

# %%
clean_geo_data.head(2)

# %%
print('Number of rows with all null values: ', clean_geo_data.isnull().all(axis=1).sum())
print('Number of rows with null value for state: ', clean_geo_data['state'].isnull().sum())
print('Number of rows with null value for county: ', clean_geo_data['county'].isnull().sum())
print('Number of rows with null value for city: ', clean_geo_data['city'].isnull().sum())
print('Number of rows with null value for latitude: ', clean_geo_data['latitude'].isnull().sum())
print('Number of rows with null value for longitude: ', clean_geo_data['longitude'].isnull().sum())

# %%
clean_geo_data['state'].unique().shape[0]

# %%
sns.heatmap(clean_geo_data.isnull(), cbar=False, xticklabels=True)

# %% [markdown]
# After this check, all the entries in the dataset have at least the state value not null and consistent. Only 12,796 data points, which account for 4.76% of the dataset, were found to have inconsistent latitude and longitude values.

# %% [markdown]
# Below, we have included some plots to visualize the inconsistency values in the dataset.

# %%
clean_geo_data.groupby(['state_consistency','county_consistency','address_consistency']).count().sort_index(ascending=False)

# %%
dummy = {}
stats_columns = ['#null_val', '#not_null', '#value_count']
for col in clean_geo_data.columns:
    dummy[col] = []
    dummy[col].append(clean_geo_data[col].isna().sum())
    dummy[col].append(len(clean_geo_data[col]) - clean_geo_data[col].isna().sum())
    dummy[col].append(len(clean_geo_data[col].value_counts()))
    
clean_geo_stat_stats = pd.DataFrame(dummy, index=stats_columns).transpose()
clean_geo_stat_stats

# %%
a = len(clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].notna())])
b = len(clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].isna())])
c = len(clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())])
d = len(clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].isna())])
e = len(clean_geo_data.loc[(clean_geo_data['latitude'].isna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].notna())])
f = len(clean_geo_data.loc[(clean_geo_data['latitude'].isna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].isna())])
g = len(clean_geo_data.loc[(clean_geo_data['latitude'].isna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())])
h = len(clean_geo_data.loc[(clean_geo_data['latitude'].isna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].isna())])

print('LAT/LONG     COUNTY     CITY             \t#samples')
print( 'not null    not null   not null         \t', a)
print( 'not null    not null   null             \t', b)
print( 'not null    null       not null         \t', c)
print( 'not null    null       null             \t', d)
print( 'null        not null   not null         \t', e)
print( 'null        not null   null             \t', f)
print( 'null        null       null             \t', g)
print( 'null        null       null             \t', h)
print('\n')
print( 'TOT samples                             \t', a+b+c+d+e+f+g+h)
print( 'Samples with not null values for lat/lon\t', a+b+c+d)
print( 'Samples with null values for lat/lon    \t', e+f+g+h)

# %%
dummy_data = clean_geo_data[clean_geo_data['latitude'].notna()]
print('Number of entries with not null values for latitude and longitude: ', len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state', zoom=2,)

# %%
dummy_data = clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & 
    (clean_geo_data['city'].notna())]
print('Number of entries with not null values for county but not for lat/lon and city: ', len(dummy_data))
plot_scattermap_plotly(dummy_data, 'state', zoom=2, title='Missing county')

# %% [markdown]
# Visualize the number of entries for each city where we have the *city* value but not the *county*

# %%
clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())].groupby('city').count()

# %%
clean_geo_data[(clean_geo_data['latitude'].notna()) & (clean_geo_data['city'].isna()) & (clean_geo_data['county'].isna())]

# %%
dummy_data = clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].isna())]
print('Number of rows with null values for city, but not for lat/lon and county: ', len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state', zoom=2, title='Missing city')

# %% [markdown]
# **Final evaluation**:
# We segmented the dataset into distinct groups based on the consistency we could establish among the latitude, longitude, state, county, and city fields. We also considered the address field in our analysis, but its variability and limited identifiability led us to set it aside for further use. In the end, we formulated strategies to address the missing data in these groups, considering the quality of available information.
#
# Below, we provide a breakdown of these groups, along with their sizes:
#
# ---------- GOOD GROUPS ----------
# * 174,796 entries: These are the fully consistent and finalized rows in the dataset.
# * 26,635 entries: Rows where only the city is missing, but it can be easily inferred from the location (k-nn).
# * 15,000 entries: Rows where only the county is missing, but it can be easily inferred from the location (k-nn).
# * 33 entries: Rows where both the city and county are missing. Even in this group, the missing information can be inferred from the location, as they are all closely clustered around Baltimore.
#
# ---------- BAD GROUPS ----------
# * 3,116 entries: Rows where latitude, longitude, and city are missing. They can be inferred (though not perfectly) from the county-state pair.
# * 19,844 entries: Rows where only the state field is present, making it challenging to retrieve missing information.
#
# The dataset does not contain any missing combinations beyond those mentioned.
# Out of the total of 216,464 lines, either the information is definitive or can be derived with a high degree of accuracy.

# %% [markdown]
# ### Infer Missing City Values

# %% [markdown]
# For entries where we have missing values for *city* but not for *latitude* and *longitude*, we attempt to assign the *city* value based on the entry's distance from the centroid.

# %% [markdown]
# Visualize data group by *state*, *county* and *city*

# %%
clean_geo_data.groupby(['state', 'county', 'city']).size().reset_index(name='count')

# %% [markdown]
# Compute the centroid for each city and visualize the first 10 centroids in alphabetical order.

# %%
centroids = clean_geo_data.loc[clean_geo_data['latitude'].notna() & clean_geo_data['city'].notna()][[
    'latitude', 'longitude', 'city', 'state', 'county']].groupby(['state', 'county', 'city']).mean()
centroids.head(10)

# %%
print('Number of distinct cities:', len(centroids.index.to_list()))

# %% [markdown]
# Create new DataFrame:

# %%
info_city = pd.DataFrame(columns=['5', '15', '25', '35', '45', '55', '65', '75', '85', '95', 
    'tot_points', 'min', 'max', 'avg', 'centroid_lat', 'centroid_lon'], index=centroids.index)
info_city.head(2)

# %% [markdown]
# The code below generates descriptive statistics related to geographical distances and updates the 'info_city' DataFrame for different city, county, and state combinations with the aim of using this data to infer missing city values.
#
# For each tuple (state, county, city) in 'centroids', it extracts all values for latitude and longitude corresponding coordinates from the 'clean_geo_data' DataFrame, if present. 
#
# It then calculates the distance between these coordinates and the centroid using the geodesic distance (in kilometers) and saves this distance in a sorted list. 
#
# After calculating percentiles (at 0.05 intervals), maximum, minimum, and average distances, all of these values are saved in the new DataFrame along with latitude and longitude coordinates.

# %%
if LOAD_DATA_FROM_CHECKPOINT: # load data
    info_city = load_checkpoint('checkpoint_geo_temporary2')
else: # compute data
    for state, county, city in centroids.index:
        dummy = []
        for lat, long in zip(clean_geo_data.loc[(clean_geo_data['city'] == city) & 
            (clean_geo_data['state'] == state) & (clean_geo_data['county'] == county) & 
            clean_geo_data['latitude'].notna()]['latitude'], 
            clean_geo_data.loc[(clean_geo_data['city'] == city) & 
            (clean_geo_data['state'] == state) & (clean_geo_data['county'] == county) & 
            clean_geo_data['longitude'].notna()]['longitude']):
            dummy.append(geopy_distance.geodesic([lat, long], centroids.loc[state, county, city]).km)
            
        dummy = sorted(dummy)
        pc = np.quantile(dummy, np.arange(0, 1, 0.05))
        for i in range(len(info_city.columns) - 6):
            info_city.loc[state, county, city][i] = pc[i*2 + 1]
        info_city.loc[state, county, city][len(info_city.columns) - 6] = len(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 5] = min(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 4] = max(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 3] = sum(dummy)/len(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 2] = centroids.loc[state, county, city]['latitude']
        info_city.loc[state, county, city][len(info_city.columns) - 1] = centroids.loc[state, county, city]['longitude']
    checkpoint(info_city, 'checkpoint_geo_temporary2') # save data 

# %%
info_city.head()

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
info_city.loc[info_city['tot_points'] > 1].info()

# %%
plot_scattermap_plotly(info_city, 'tot_points', x_column='centroid_lat', 
    y_column='centroid_lon', hover_name=False, zoom=2, title='Number of points per city')

# %% [markdown]
# We utilize the previously calculated data to infer missing values for the *city* field in entries of the dataset where latitude and longitude are available. The *city* field is assigned if the distance of the entry from the centroid falls within the third quartile of all points assigned to that centroid.

# %%
def substitute_city(row, info_city):
    if pd.isna(row['city']) and not np.isnan(row['latitude']):
        for state, county, city in info_city.index:
            if row['state'] == state and row['county'] == county:
                if info_city.loc[state, county, city]['tot_points'] > 1:
                    max_radius = info_city.loc[state, county, city]['75'] # terzo quantile
                    centroid_coord = [info_city.loc[state, county, city]['centroid_lat'], 
                        info_city.loc[state, county, city]['centroid_lon']]
                    if (geopy_distance.geodesic([row['latitude'], row['longitude']], centroid_coord).km <= 
                        max_radius):
                        row['city'] = city
                        break
                    
    return row

# %%
if LOAD_DATA_FROM_CHECKPOINT: # load data
    final_geo_data = load_checkpoint('checkpoint_geo')
else: # compute data
    final_geo_data = clean_geo_data.apply(lambda row: substitute_city(row, info_city), axis=1)
    checkpoint(final_geo_data, 'checkpoint_geo') # save data

# %%
final_geo_data.head(2)

# %%
print('Number of rows with null values for city before: ', clean_geo_data['city'].isnull().sum())
print('Number of rows with null values for city: ', final_geo_data['city'].isnull().sum())

# %% [markdown]
# From this process, we infer 2248 *city* values.

# %% [markdown]
# ### Visualize new data

# %%
a = len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].notna())])
b = len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].isna())])
c = len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].isna()) & (final_geo_data['city'].notna())])
d = len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].isna()) & (final_geo_data['city'].isna())])
e = len(final_geo_data.loc[(final_geo_data['latitude'].isna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].notna())])
f = len(final_geo_data.loc[(final_geo_data['latitude'].isna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].isna())])
g = len(final_geo_data.loc[(final_geo_data['latitude'].isna()) & (final_geo_data['county'].isna()) & (final_geo_data['city'].notna())])
h = len(final_geo_data.loc[(final_geo_data['latitude'].isna()) & (final_geo_data['county'].isna()) & (final_geo_data['city'].isna())])

print('LAT/LONG     COUNTY     CITY             \t#samples')
print( 'not null    not null   not null         \t', a)
print( 'not null    not null   null             \t', b)
print( 'not null    null       not null         \t', c)
print( 'not null    null       null             \t', d)
print( 'null        not null   not null         \t', e)
print( 'null        not null   null             \t', f)
print( 'null        null       null             \t', g)
print( 'null        null       null             \t', h)
print('\n')
print( 'TOT samples                             \t', a+b+c+d+e+f+g+h)
print( 'Samples with not null values for lat/lon\t', a+b+c+d)
print( 'Samples with null values for lat/lon    \t', e+f+g+h)

# %%
plot_utils.plot_scattermap_plotly(final_geo_data.loc[(final_geo_data['latitude'].notna()) & 
    (final_geo_data['county'].notna()) & (final_geo_data['city'].isna())], 'state', zoom=2, title='Missing city')

# %%
#TODO: plottare le città che ha inserto e i centroidi??

# %%
final_geo_data.head(3)


# %% [markdown]
# ## Geo CONTROLLARE

# %% [markdown]
# We now check the values of the attribute `state` comparing them with the values in the dictionary `usa_code` that maps the name of USA states to their alphanumerical codes that we will later use to plot maps:

# %%
states = incidents_data['state'].unique()
not_existing_states = False
missing_states = False
for state in states:
    if state not in usa_code:
        not_existing_states = True
        print(f"State {state} does not exist")
for state in usa_code:
    if state not in states:
        missing_states = True
        print(f"State {state} is missing")
if not_existing_states == False:
    print("All the values of the attribute 'states' are actually USA states (there are no misspelling or other errors).")
if missing_states == False:
    print("There is at least one incident for each USA state.")

# %% [markdown]
# We convert the attribute `state` to uppercase so that we can later merge this dataset with the one containing the data about elections:

# %%
incidents_data['state'] = incidents_data['state'].str.upper()

# %% [markdown]
# We now check if, given a certain value for the attributes `latitude` and a `longitude`, the attribute `city_or_county` has always the same value:

# %%
incidents_data.groupby(['latitude', 'longitude'])['city_or_county'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# That is not true and is due to the fact that sometimes the attribute `city_or_county` takes on the value of the city, other times the value of the county (as in the first row displayed above). Furthermore, we notice even when the attribute refers to the same county it could be written in different ways (e.g. "Bethel (Newtok)", "Bethel (Napaskiak)", "Bethel"). 

# %% [markdown]
# We now check if a similar problem occurs for the attribute `address`:

# %%
incidents_data.groupby(['latitude', 'longitude'])['address'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# Still this attribute may be written in different ways (e.g. "Avenue" may also be written as "Ave", or "Highway" as "Hwy"). There could also be some errors (e.g. the same point corresponds to the address "33rd Avenue", "Kamehameha Highway" and "Kilauea Avenue extension").
#
# We plot on a map the location of the incidents:

# %%
fig = px.scatter_mapbox(
    lat=incidents_data['latitude'],
    lon=incidents_data['longitude'],
    zoom=0, 
    height=500,
    width=800
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %% [markdown]
# There are some points in China and in Russia that are clearly wrong. We display the rows of the dataset that correspond to one of these points:

# %%
incidents_data[(incidents_data['latitude'] == 37.6499) & (incidents_data['longitude'] == 97.4331)]

# %% [markdown]
# That point has probably the correct values for the attributes `state` and `city_or_county`.

# %%
# TODO: le oservazioni sopra dovrebbero giustificare la scelta di usare geopy, importare qui il codice e spiegare cosa è stato fatto.

# %% [markdown]
# We check if the attribute `congressional_district` is numbered consistently (with '0' for states with only one congressional district). To do so we use the data from the dataset containing the data about elections in the period of interest (congressional districts are redrawn when (year%10)==0):

# %%
at_large_states = elections_data[
    (elections_data['year'].between(2013, 2018, inclusive="both")) & 
    (elections_data['congressional_district']==0)
    ]['state'].unique()
at_large_states

# %% [markdown]
# Now we check if states with a '0' as congressional district are the same states with only one congressional district in the dataset containing the data about elections:

# %%
zero_congress_states_inc = incidents_data[incidents_data['congressional_district']==0]['state'].unique()
set(zero_congress_states_inc).issubset(set(at_large_states))

# %% [markdown]
# We check if states with a single congressional district are always numbered with '0' in the dataset containing the data about elections:

# %%
incidents_data[(incidents_data['state'] == at_large_states.any()) & (incidents_data['congressional_district']!=0)].size==0

# %% [markdown]
# Since they are not, we fix this issue:

# %%
incidents_data.loc[incidents_data['state'].isin(at_large_states), 'congressional_district'] = 0

# %% [markdown]
# We check if the range of the attributes `congressional_district` is consistent with the number of congressional districts in the dataset containing the data about elections:

# %%
wrong_congr_states = elections_data.groupby('state')['congressional_district'].max()>=incidents_data.groupby('state')['congressional_district'].max()
for state in wrong_congr_states[wrong_congr_states==False].index:
    print(f"State {state} has more districts in the incidents data than in the elections data")

# %%
incidents_data[
    (incidents_data['state']=='KENTUCKY') &
    (incidents_data['congressional_district'] > 
        elections_data[(elections_data['state']=='KENTUCKY') & (elections_data['year']>2012)]['congressional_district'].max())
]

# %%
# TODO: questi errori ci sono sempre una volta corretta la geografia?

# %% [markdown]
# We check whether given a certain value for the attributes `latitude` and a `longitude`, the attribute `congressional_district` has always the same value:

# %%
incidents_data[incidents_data['congressional_district'].notnull()].groupby(['latitude', 'longitude'])['congressional_district'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# All these points are probably errors, due to the fact that they are near the border between two congressional districts. We correct them setting the most frequent value for the attribute `congressional_district` (setting that value also for the entries with missing values):

# %%
corrected_congr_districts = incidents_data[
    ~incidents_data['congressional_district'].isna()
    ].groupby(['latitude', 'longitude'])['congressional_district'].agg(lambda x: x.value_counts().index[0])
incidents_data = incidents_data.merge(corrected_congr_districts, on=['latitude', 'longitude'], how='left')
# where latitude and longitude are null, keep the original value
incidents_data['congressional_district_y'].fillna(incidents_data['congressional_district_x'], inplace=True)
incidents_data.rename(columns={'congressional_district_y':'congressional_district'}, inplace=True)
incidents_data.drop(columns=['congressional_district_x'], inplace=True)

# %% [markdown]
# In the same city or county there could be different values for the attribute `congressional_district` (this is not an error, is actually possible according to the USA law):

# %%
incidents_data[incidents_data['congressional_district'].notna()].groupby(['state', 'city_or_county'])['congressional_district'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# We print the unique values the attribute `state_house_district` can take on:

# %%
house_districts = incidents_data['state_house_district'].unique()
house_districts.sort()
house_districts

# %% [markdown]
# Also this attribute has some errors because the maximum number of state house districts should be 204 (for New Hampshire, see [here](https://ballotpedia.org/State_Legislative_Districts)). For now we won't correct this error beacuse this attribute is not useful for our analysis.
#
# We check if given a certain value for the attributes `latitude` and a `longitude`, the attribute `state_house_district` has always the same value:

# %%
incidents_data[incidents_data['state_house_district'].notnull()].groupby(
    ['latitude', 'longitude'])['state_house_district'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# We correct the errors:

# %%
corrected_house_districts = incidents_data[
    incidents_data['state_house_district'].notnull()
    ].groupby(['latitude', 'longitude'])['state_house_district'].agg(lambda x: x.value_counts().index[0])
incidents_data = incidents_data.merge(corrected_house_districts, on=['latitude', 'longitude'], how='left')
incidents_data['state_house_district_y'].fillna(incidents_data['state_house_district_x'], inplace=True)
incidents_data.rename(columns={'state_house_district_y':'state_house_district'}, inplace=True)
incidents_data.drop(columns=['state_house_district_x'], inplace=True)

# %% [markdown]
# We now print the unique values the attribute `state_senate_district` can take on:

# %%
senate_districts = incidents_data['state_senate_district'].unique()
senate_districts.sort()
senate_districts

# %% [markdown]
# And again we notice some errors because the maximum number of state senate districts should be 67 (for Minnesota, see [here](https://ballotpedia.org/State_Legislative_Districts)). For now we won't correct this error beacuse this attribute is not useful for our analysis.
#
# We correct other possible errors as above:

# %%
corrected_senate_districts = incidents_data[
    incidents_data['state_senate_district'].notnull()
    ].groupby(['latitude', 'longitude'])['state_senate_district'].agg(lambda x: x.value_counts().index[0])
incidents_data = incidents_data.merge(corrected_senate_districts, on=['latitude', 'longitude'], how='left')
incidents_data['state_senate_district_y'].fillna(incidents_data['state_senate_district_x'], inplace=True)
incidents_data.rename(columns={'state_senate_district_y':'state_senate_district'}, inplace=True)
incidents_data.drop(columns=['state_senate_district_x'], inplace=True)

# %% [markdown]
# We check whether given a `state`, `city_or_county` and `state_senate_district`, the value of the attribute `congressional_district` is always the same:

# %%
incidents_data[incidents_data['congressional_district'].notnull()].groupby(
    ['state', 'city_or_county', 'state_senate_district'])['congressional_district'].unique()[lambda x: x.str.len() > 1].shape[0]==0

# %% [markdown]
# Hence we cannot recover the missing values for the attribute `congressional_district` from the values of `state_senate_district`. We check the same for the attribute `state_house_district`:

# %%
incidents_data[incidents_data['congressional_district'].notnull()].groupby(
    ['state', 'city_or_county', 'state_house_district'])['congressional_district'].unique()[lambda x: x.str.len() > 1].shape[0]==0

# %% [markdown]
# We cannot recover the missing values for the attribute `congressional_district` from the values of `state_house_district` either.
#
# We could, instead, recover the missing values from the entries with "similar" `latitude` and `longitude`. To explore this possibility we first plot on a map the dislocation of the incidents, coloring them according to the value of the attribute `congressional_district`:

# %%
plot_scattermap_plotly(
    incidents_data,
    'congressional_district',
    black_nan=True,
    zoom=2,
    height=800,
    width=800,
    title="USA Congressional districts"
    )

# %% [markdown]
# Many points with missing `congressional_district` are often "surrounded" by points belonging to the same congressional district. We could, therefore, use KNN classifier to recover those values.
#
# We'll do this first for the state of Alabama, showing the results with some plots. Later we will do the same for all the other states. We plot the distribution of the values of the attribute `congressional_district` for the state of Alabama:

# %%
plot_scattermap_plotly(
    incidents_data[incidents_data['state']=='ALABAMA'],
    attribute='congressional_district',
    black_nan=True,
    width=500,
    height=600,
    zoom=5.5,
    title="Alabama incidents by Congressional Districts",
    legend_title="Congressional District"
)

# %% [markdown]
# We define a function to prepare the data for the classification task:

# %%
def build_X_y_for_district_inference(incidents_data):
    X_train = np.concatenate((
        incidents_data[
            (incidents_data['congressional_district'].notna()) &
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna())
            ]['latitude'].values.reshape(-1, 1),
        incidents_data[
            (incidents_data['congressional_district'].notna()) & 
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna())
            ]['longitude'].values.reshape(-1, 1)),
        axis=1
    )
    X_test = np.concatenate((
        incidents_data[
            (incidents_data['congressional_district'].isna()) & 
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna())
            ]['latitude'].values.reshape(-1, 1),
        incidents_data[
            (incidents_data['congressional_district'].isna()) &
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna())
            ]['longitude'].values.reshape(-1, 1)),
        axis=1
    )
    y_train = incidents_data[
        (incidents_data['congressional_district'].notna()) & 
        (incidents_data['latitude'].notna()) & 
        (incidents_data['longitude'].notna())
        ]['congressional_district'].values
    return X_train, X_test, y_train

# %% [markdown]
# We define the function to compute the geodesic distance to pass to the KNN classifier:

# %%
def geodesic_distance(point1, point2):
    return geopy_distance.geodesic(point1, point2).km

# %% [markdown]
# Now we are ready to apply the classifier (using K=1):

# %%
X_train, X_test, y_train = build_X_y_for_district_inference(incidents_data[incidents_data['state']=="ALABAMA"])
knn_clf = KNeighborsClassifier(n_neighbors=1, metric=geodesic_distance)
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
incidents_data['KNN_congressional_district'] = incidents_data['congressional_district']
incidents_data.loc[
    (incidents_data['state']=="ALABAMA") &
    (incidents_data['congressional_district'].isna()) &
    (incidents_data['latitude'].notna()) & 
    (incidents_data['longitude'].notna()),
    'KNN_congressional_district'
    ] = knn_pred

# %% [markdown]
# We plot the results:

# %%
plot_scattermap_plotly(
    incidents_data[incidents_data['state']=='ALABAMA'],
    attribute='KNN_congressional_district',
    width=500,
    height=600,
    zoom=5.5,
    title="Alabama incidents by Congressional Districts",
    legend_title="Congressional District"
)

# %% [markdown]
# To improve the visualization, we plot on the map the decision boundaries of the classifier. To do so, we convert latitude and longitude to a 2D space:

# %%
transformer = Transformer.from_crs("EPSG:4326", "EPSG:26929", always_xy=True)

X_train_converted = []

for i in range(X_train.shape[0]):
    x, y = transformer.transform(X_train[i][1], X_train[i][0])
    X_train_converted.append([x,y])

X_train_converted = np.array(X_train_converted)

# %% [markdown]
# And now we train the classifier using the euclidean distance:

# %%
knn_eu_clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn_eu_clf.fit(X_train_converted, y_train)

# %% [markdown]
# We plot the boundaries of the classifier:

# %%
alabama_color_map = {
    1:'red',
    2:'orange',
    3:'yellow',
    4:'green',
    5:'lightblue',
    6:'blue',
    7:'purple'
}
plot_clf_decision_boundary(knn_eu_clf, X_train_converted, y_train, alabama_color_map, "KNN Alabama borders")

# %% [markdown]
# We can now compare the boundaries built by the classifier with the actual boundaries (this map was taken [here](https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif/lossless-page1-1256px-United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif.png)):
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif/lossless-page1-1256px-United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif.png" alt="Alt Text" width="600"/>

# %% [markdown]
# The result is satisfactory. However, it is important to highlight that if there are no examples available for a specific district, we won't assign the correct label to the points in that districts. We check how many congressional districts have 2 or less examples:

# %%
incidents_data.groupby(['state', 'congressional_district']).size()[lambda x: x <= 2]

# %% [markdown]
# By the way, missclassification can still occurr, depending on the position of the available examples w.r.t the position of the points to classify. Aware of this limitation, we proceed to apply this method to the other states and plot the result:

# %%
# for state in incidents_data['state'].unique():
#     if state != "ALABAMA":
#         print(f"{state} done.")
#         X_train, X_test, y_train = build_X_y_for_district_inference(incidents_data[incidents_data['state']==state])
#         if X_test.shape[0] == 0:
#             continue
#         knn_clf.fit(X_train, y_train)
#         knn_pred = knn_clf.predict(X_test)
#         incidents_data.loc[
#             (incidents_data['state']==state) &
#             (incidents_data['congressional_district'].isna()) &
#             (incidents_data['latitude'].notna()) & 
#             (incidents_data['longitude'].notna()),
#             'KNN_congressional_district'
#         ] = knn_pred
# plot_scattermap_plotly(
#     incidents_data,
#     'congressional_district',
#     zoom=2,
#     height=800,
#     width=800,
#     title="USA Congressional districts (after inference)"
#     )

# %% [markdown]
# We drop the original column with congressional districts and we replace it with the one with the one we just computed:

# %%
incidents_data.drop(columns=['congressional_district'], inplace=True)
incidents_data.rename(columns={'KNN_congressional_district':'congressional_district'}, inplace=True)

# %% [markdown]
# We now plot on a map the location of the incidents, coloring them according to the value of the attribute `state_senate_district` and `state_house_district`, to assess wheter we can apply the same method to recover missing values:

# %%
plot_scattermap_plotly(
    incidents_data,
    'state_senate_district',
    black_nan=True,
    zoom=2,
    height=800,
    width=800,
    title="USA State senate districts"
    )

plot_scattermap_plotly(
    incidents_data,
    'state_house_district',
    black_nan=True,
    zoom=2,
    height=800,
    width=800,
    title="USA State house districts"
    )

# %% [markdown]
# These attributes have a lot of missing values, sometimes spread over large areas where there are no other points. Given this scarcity of training examples, we cannot apply the same method to recover the missing values.

# %% [markdown]
# ## Age, gender and number of participants data

# %% [markdown]
# ### Features

# %% [markdown]
# Columns of the dataset are considered in order to verify the correctness and consistency of data related to age, gender, and the number of participants for each incident:
# - *participant_age1*
# - *participant_age_group1*
# - *participant_gender1*
# - *min_age_participants*
# - *avg_age_participants*
# - *max_age_participants*
# - *n_participants_child*
# - *n_participants_teen*
# - *n_participants_adult*
# - *n_males*
# - *n_females*
# - *n_killed*
# - *n_injured*
# - *n_arrested*
# - *n_unharmed*
# - *n_participants*

# %%
# participant_age1,participant_age_group1,participant_gender1,min_age_participants,avg_age_participants,max_age_participants,n_participants_child,n_participants_teen,n_participants_adult,n_males,n_females,n_killed,n_injured,n_arrested,n_unharmed,n_participants
age_data = incidents_data[['participant_age1', 'participant_age_group1', 'participant_gender1', 
    'min_age_participants', 'avg_age_participants', 'max_age_participants',
    'n_participants_child', 'n_participants_teen', 'n_participants_adult', 
    'n_males', 'n_females',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants']]

# %%
age_data.head(10)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
age_data.info()

# %%
age_data['participant_age_group1'].unique()

# %% [markdown]
# Display the maximum and minimum ages, among the possible valid values, in the dataset. We have set a maximum threshold of 122 years, as it is the age reached by [Jeanne Louise Calment](https://www.focus.it/scienza/scienze/longevita-vita-umana-limite-biologico#:~:text=Dal%201997%2C%20anno%20in%20cui,ha%20raggiunto%20un%20limite%20biologico), the world's oldest person.

# %%
def max_min_value(attribute):
    age = []
    for i in age_data[attribute].unique():
        try: 
            i = int(float(i))
            if i <= 122 and i > 0: age.append(i)
        except: pass
    print(f'Max value for attribute {attribute}: {np.array(age).max()}')
    print(f'Max value for attribute {attribute}: {np.array(age).min()}')

max_min_value('participant_age1')
max_min_value('min_age_participants')
max_min_value('max_age_participants')
max_min_value('avg_age_participants')

# %%
age_data[age_data['max_age_participants'] == '101.0']

# %% [markdown]
# We have set the maximum age threshold at 101 years.

# %% [markdown]
# We check if we have entries with non-null values for participant_age1 but NaN for participant_age_group1. 

# %%
age_data[age_data['participant_age1'].notna() & age_data['participant_age_group1'].isna()]

# %% [markdown]
# These 126 values can be inferred.

# %% [markdown]
# ### Studying Data Consistency

# %% [markdown]
# We create some functions to identify and, if possible, correct missing and inconsistent data.
# Below, we provide a brief summary of all the functions used to check data consistency:

# %% [markdown]
# First of all, we convert all the values to type int if the values were consistent (i.e., values related to age and the number of participants for a particular category must be a positive number), all the values that are out of range or contain alphanumeric strings were set to *NaN*.

# %% [markdown]
# Checks done to evaluate the consistency of data related to the minimum, maximum, and average ages of participants, as well as the composition of the age groups:
#
# - min_age_participants $<$ avg_age_participants $<$ max_age_participants
# - n_participants_child $+$ n_participants_teen $+$ n_participants_adult $>$ 0
#
# - $if$ min_age_participants $<$ 12 $then$ n_participants_child $>$ 0
# - $if$ 12 $\leq$ min_age_participants $<$ 18 $then$ n_participants_teen $>$ 0
# - $if$ min_age_participants $\geq$ 18 $then$ n_participants_adult $>$ 0
#
# - $if$ max_age_participants $<$ 12 $then$ n_participants_child $>$ 0 and n_participants_teen $=$ 0 and n_participants_adult $=$ 0
# - $if$ max_age_participants $<$ 18 $then$ n_participants_teen $>$ 0 or n_participants_child $>$ 0 and n_participants_adult $=$ 0
# - $if$ max_age_participants $\geq$ 18 $then$ n_participants_adult $>$ 0
#
# Note that: child = 0-11, teen = 12-17, adult = 18+

# %% [markdown]
# Checks done to evaluate the consistency of data related to number of participants divided by gender and other participants class:
#
# - n_participants $\geq$ 0
# - n_participants $==$ n_males $+$ n_females
# - n_killed $+$ n_injured $\leq$ n_participants
# - n_arrested $\leq$ n_participants
# - n_unharmed $\leq$ n_participants

# %% [markdown]
# We also considered data of participants1, a randomly chosen participant whose data related to gender and age are reported in the dataset. For participants, we have the following features: *participant_age1*, *participant_age_group1*, *participant_gender1*.
#
# Values related to participant_age_group1 and participant_gender1 have been binarized using one-hot encoding, thus creating the boolean features *participant1_child*, *participant1_teen*, *participant1_adult*, *participant1_male*, *participant1_female*.
#
# The following checks are done in order to verify the consistency of the data among them and with respect to the other features of the incident:
#
# - $if$ participant_age1 $<$ 12 $then$ participant_age_group1 $=$ *Child*
# - $if$ 12 $\leq$ participant_age1 $<$ 18 $then$ participant_age_group1 $=$ *Teen*
# - $if$ participant_age1 $\geq$ 18 $then$ participant_age_group1 $==$ *Adult*
#
# - $if$ participant_age_group1 $==$ *Child* $then$ n_participants_child $>$ 0
# - $if$ participant_age_group1 $==$ *Teen* $then$ n_participants_teen $>$ 0
# - $if$ participant_age_group1 $==$ *Adult* $then$ n_participants_adult $>$ 0
#
# - $if$ participant_gender1 $==$ *Male* $then$ n_males $>$ 0
# - $if$ participant_gender1 $==$ *Female* $then$ n_females $>$ 0

# %% [markdown]
# In the initial phase, only the values that were not permissible were set to *NaN*. 
#
# We kept track of the consistency of admissible values by using variables (which could take on the boolean value *True* if they were consistent, *False* if they were not, or *NaN* in cases where data was not present). 
#
# These variables were temporarily included in the dataframe so that we could later replace them with consistent values, if possible, or remove them if they were outside the acceptable range.
#
# Variables:
# - *consistency_age*: Values related to the minimum, maximum, and average ages consistent with the number of participants by age groups.
# - *consistency_n_participant*: The number of participants for different categories consistent with each other.
# - *consistency_gender*: The number of participants by gender consistent with the total number of participants.
# - *consistency_participant1*: Values of features related to participant1 consistent with each other.
#
# - *consistency_participants1_wrt_n_participants*: If *consistency_participants1_wrt_n_participants*, *participant1_age_range_consistency_wrt_all_data*, and *participant1_gender_consistency_wrt_all_data* are all *True*.
#
# - *participant1_age_consistency_wrt_all_data*: Age of participant1 consistent with the minimum and maximum age values of the participants.
# - *participant1_age_range_consistency_wrt_all_data*: Value of the age range (*Child*, *Teen*, or *Adult*) consistent with the age groups of the participants.
# - *participant1_gender_consistency_wrt_all_data*: Gender value of participant1 consistent with the gender breakdown values of the group.
#
# - *nan_values*: Presence of "NaN" values in the row.

# %%
from data_preparation_utils import check_age_gender_data_consistency

if LOAD_DATA_FROM_CHECKPOINT: # load data
    age_temporary_data = load_checkpoint('checkpoint_age_temporary')
else: # compute data
    age_temporary_data = age_data.apply(lambda row: check_age_gender_data_consistency(row), axis=1)
    checkpoint(age_temporary_data, 'checkpoint_age_temporary') # save data

# %% [markdown]
# ### Data Exploration without Out-of-Range Data

# %%
age_temporary_data.head(2)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
age_temporary_data.info()

# %% [markdown]
# We assess the correctness of the checks performed by printing the consistency variable for the first 5 rows and providing a concise summary of their most frequent values.

# %%
age_temporary_data[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].head(5)

# %%
age_temporary_data[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].describe()

# %% [markdown]
# Below, we print the number of rows with 'NaN' or inconsistent data.

# %%
print('Number of rows with null values: ', age_temporary_data[age_temporary_data['nan_values'] == True].shape[0])
print('Number of rows with inconsistent values in age data: ', age_temporary_data[age_temporary_data['consistency_age'] == False].shape[0])
print('Number of rows with inconsistent values in number of participants data: ', age_temporary_data[age_temporary_data[
    'consistency_n_participant'] == False].shape[0])
print('Number of rows with inconsistent values in gender data: ', age_temporary_data[age_temporary_data['consistency_gender'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 data: ', age_temporary_data[age_temporary_data[
    'consistency_participant1'] == False].shape[0])

# %%
print('Number of rows with inconsistent values for participants1: ', age_temporary_data[age_temporary_data[
    'consistency_participant1'] == False].shape[0])
print('Number of rows with NaN values for participants1: ', age_temporary_data[age_temporary_data[
    'consistency_participant1'] == np.nan].shape[0])
print('Number of rows with inconsistent values in participants1 wrt all other data: ', age_temporary_data[age_temporary_data[
    'consistency_participants1_wrt_n_participants'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age data: ', age_temporary_data[age_temporary_data[
    'participant1_age_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age range data: ', age_temporary_data[age_temporary_data[
    'participant1_age_range_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt gender data: ', age_temporary_data[age_temporary_data[
    'participant1_gender_consistency_wrt_all_data'] == False].shape[0])

# %%
age_temporary_data[(age_temporary_data['consistency_participant1'] == True) & (age_temporary_data[
    'participant1_age_range_consistency_wrt_all_data'] == False)].shape[0]

# %%
print('Number of rows with null values in age data: ', age_temporary_data[age_temporary_data['consistency_age'].isna()].shape[0])
print('Number of rows with null values in number of participants data: ', age_temporary_data[age_temporary_data[
    'consistency_n_participant'].isna()].shape[0])
print('Number of rows with null values in gender data: ', age_temporary_data[age_temporary_data['consistency_gender'].isna()].shape[0])
print('Number of rows with null values in participants1 data: ', age_temporary_data[age_temporary_data[
    'consistency_participant1'].isna()].shape[0])

# %%
print('Number of rows with all null data: ', age_temporary_data.isnull().all(axis=1).sum())

# %% [markdown]
# We can notice that:
# - The data in our dataset related to participant1, excluding the 1099 cases where age and age group data were inconsistent with each other and 190 cases where age range is not consistent, always appear to be consistent with the data in the rest of the dataset and can thus be used to fill in missing or incorrect data.
# - In the data related to age and gender, some inconsistencies are present, but they account for only 1.88% and 6.01% of the total dataset rows, respectively.
# - In 93779 rows, at least one field had a *NaN* value.

# %% [markdown]
# Since we noticed that some age data contained impossible values, we have set the age range between 0 and 100 years old. Below, we have verified this by printing the range.

# %%
print('Range age: ', age_temporary_data['min_age_participants'].min(), '-', age_temporary_data['max_age_participants'].max())

# %%
age_temporary_data[age_temporary_data['consistency_participant1'] == False].head(5)

# %% [markdown]
# We printed the distribution of participants1 in the age range when age was equal to 18 to verify that the majority of the data were categorized as adults.

# %%
age_data[age_data['participant_age1'] == 18]['participant_age_group1'].value_counts()

# %% [markdown]
# We plotted the age distribution of participant1 and compared it to the distribution of the minimum and maximum participants' age for each group.

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

ax0.hist(age_temporary_data['participant_age1'], bins=100, edgecolor='black', linewidth=0.8)
ax0.set_xlabel('Age')
ax0.set_ylabel('Frequency')
ax0.set_title('Distribution of age participant1')

ax1.hist(age_temporary_data['min_age_participants'], bins=100, edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of min age participants')

ax2.hist(age_temporary_data['max_age_participants'], bins=100, edgecolor='black', linewidth=0.8)
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of max age participants')

plt.show()

# %% [markdown]
# Observing the similar shapes of the distributions provides confirmation that the data pertaining to participant1 is accurate and reliable. Therefore, we can confidently use participant1's data to fill gaps in cases involving groups with a single participant.

# %% [markdown]
# We visualized the number of unique values for the cardinality of participants in each incident and provided a brief summary of this feature below.

# %%
print('Values of n_participants: ', age_temporary_data['n_participants'].unique())
display(age_temporary_data['n_participants'].describe())

# %% [markdown]
# From the data above, it is evident that the third quartile is equal to two participants, and the maximum number of participants per incident reaches the value of 103.
#
# Below, we have presented the distribution of the number of participants for each incident. In order to make the histograms more comprehensible, we have chosen to represent the data on two separate histograms.

# %%
# distribuition number of participants
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

ax0.hist(age_temporary_data['n_participants'], bins=15, range=(0,15), edgecolor='black', linewidth=0.8)
ax0.set_xlabel('Number of participants')
ax0.set_ylabel('Frequency')
ax0.set_title('Distribution of number of participants (1-15 participants)')

ax1.hist(age_temporary_data['n_participants'], bins=15, range=(15,103), edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Number of participants')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of number of participants (15-103 participants)')
plt.show()

# %% [markdown]
# Note that: the chart on the left shows the distribution of data for groups with a number of participants between 0 and 15, while the one on the right displays data for groups between 15 and 103. The y-axes are not equal.

# %% [markdown]
# In the table below, we can see how many data related to the *number of participants* are clearly out of range, divided by age groups.

# %%
age_temporary_data[age_temporary_data['n_participants_adult'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
age_temporary_data[age_temporary_data['n_participants_teen'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
age_temporary_data[age_temporary_data['n_participants_child'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %% [markdown]
# Based on the tables above, we have evidence to set the maximum number of participants to 103.

# %% [markdown]
# We have provided additional information below for two of the rows with values out of range.

# %%
age_temporary_data.loc[35995]

# %%
age_temporary_data.iloc[42353]

# %% [markdown]
# This data visualization has been helpful in understanding the exceptions in the dataset and correcting them when possible, using other data from the same entry.
#
# In cases where we were unable to obtain consistent data for a certain value, we have set the corresponding field to *NaN*.

# %% [markdown]
# ### Fix Inconsistent Data

# %% [markdown]
# We have created a new DataFrame in which we have recorded the corrected and consistent data. Note that all these checks are performed based on the assumptions made in previous stages of the analysis.
#
# For entries with missing or inconsistent data, when possible, we have inferred or derived the missing values from other available data. Specifically:
#
# - In cases where we had the number of males (n_males) and number of females (n_females), we calculated the total number of participants as n_participants = n_males + n_females.
# - In instances with a single participant and consistent data for *participants1*, we used that data to derive values related to age (max, min, average) and gender.

# %%
from data_preparation_utils import  set_gender_age_consistent_data

if LOAD_DATA_FROM_CHECKPOINT: # load data
    new_age_data = load_checkpoint('checkpoint_age')
else: # compute data
    new_age_data = age_temporary_data.apply(lambda row: set_gender_age_consistent_data(row), axis=1)
    checkpoint(age_temporary_data, 'checkpoint_age') # save data

# %% [markdown]
# We display the first 2 rows and a concise summary of the DataFrame:

# %%
new_age_data.head(2)

# %%
new_age_data.info()

# %%
print('Number of rows in which all data are null: ', new_age_data.isnull().all(axis=1).sum())
print('Number of rows with some null data: ', new_age_data.isnull().any(axis=1).sum())
print('Number of rows in which number of participants is null: ', new_age_data[new_age_data['n_participants'].isnull()].shape[0])
print('Number of rows in which number of participants is 0: ', new_age_data[new_age_data['n_participants'] == 0].shape[0])
print('Number of rows in which number of participants is null and n_killed is not null: ', new_age_data[
    new_age_data['n_participants'].isnull() & new_age_data['n_killed'].notnull()].shape[0])

# %%
print('Total rows with null value for n_participants: ', new_age_data['n_participants'].isnull().sum())
print('Total rows with null value for n_participants_child: ', new_age_data['n_participants_child'].isnull().sum())
print('Total rows with null value for n_participants_teen: ', new_age_data['n_participants_teen'].isnull().sum())
print('Total rows with null value for n_participants_adult: ', new_age_data['n_participants_adult'].isnull().sum())
print('Total rows with null value for n_males: ', new_age_data['n_males'].isnull().sum())
print('Total rows with null value for n_females: ', new_age_data['n_females'].isnull().sum())

# %% [markdown]
# We can observe that for any entries in the dataset, all data related to age and gender are *NaN*, while for 98973 entries, almost one value is *NaN*. From the plot below, we can visualize the null values (highlighted).
#
# It's important to note that we have complete data for *n_killed* and *n_injured* entries, and the majority of missing data are related to age-related features.

# %%
sns.heatmap(new_age_data.isnull(), cbar=False)

# %% [markdown]
# Below, we have provided the distribution of the total number of participants and the number of participants divided by age range for each incident. Once again, to make the histograms more comprehensible, we have opted to present the data on separate histograms.

# %%
# distribuition number of participants
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

ax0.hist(age_temporary_data['n_participants'], bins=15, range=(0,15), edgecolor='black', linewidth=0.8)
ax0.set_xlabel('Number of participants')
ax0.set_ylabel('Frequency')
ax0.set_title('Distribution of number of participants (1-15 participants)')

ax1.hist(age_temporary_data['n_participants'], bins=15, range=(15,103), edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Number of participants')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of number of participants (15-103 participants)')
plt.show()

# %%
print('Max number of participants: ', new_age_data['n_participants'].max())
print('Max number of children: ', new_age_data['n_participants_child'].max())
print('Max number of teens: ', new_age_data['n_participants_teen'].max())
print('Max number of adults: ', new_age_data['n_participants_adult'].max())

# %%
new_age_data[new_age_data['n_participants_adult'] > 60][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
# distribuition number of participants divided by age group
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 8), sharey=False)

ax0.hist(age_temporary_data['n_participants_child'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='blue', label='Children')
ax0.hist(age_temporary_data['n_participants_teen'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='magenta', label='Teens')
ax0.hist(age_temporary_data['n_participants_adult'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='green', label='Adults')
ax0.set_xlabel('Number of participants')
ax0.set_ylabel('Frequency')
ax0.legend()
ax0.set_title('Distribution of number of participants (1-10 participants)')

ax1.hist(age_temporary_data['n_participants_child'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='blue', label='Children')
ax1.hist(age_temporary_data['n_participants_teen'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='magenta', label='Teens')
ax1.hist(age_temporary_data['n_participants_adult'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='green', label='Adults')
ax1.set_xlabel('Number of participants')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.set_title('Distribution of number of participants (10-30 participants)')

ax2.hist(age_temporary_data['n_participants_child'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='blue', label='Children')
ax2.hist(age_temporary_data['n_participants_teen'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='magenta', label='Teens')
ax2.hist(age_temporary_data['n_participants_adult'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='green', label='Adults')
ax2.set_xlabel('Number of participants')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.set_title('Distribution of number of participants (30-103 participants)')

plt.show()

# %% [markdown]
# We observe that in incidents involving children and teenagers under the age of 18, the total number of participants was less than 7 and 27, respectively. In general, incidents involving a single person are much more frequent than other incidents, and most often, they involve teenagers and children, with a smaller percentage involving adults. On the other hand, incidents with multiple participants mostly consist of adults, and as the number of participants increases, the frequency of such incidents decreases. 
#
# Note that the y-axis of the histograms is not equal.

# %% [markdown]
# We also plot the distribution of the number of incidents divided by gender:

# %%
# distribuition number of participants divided by gender
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 8), sharey=False)

ax0.hist(age_temporary_data['n_males'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='blue', label='Males')
ax0.hist(age_temporary_data['n_females'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='red', label='Females')
ax0.set_xlabel('Number of participants')
ax0.set_ylabel('Frequency')
ax0.legend()
ax0.set_title('Distribution of number of participants (1-10 participants)')

ax1.hist(age_temporary_data['n_males'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='blue', label='Males')
ax1.hist(age_temporary_data['n_females'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='red', label='Females')
ax1.set_xlabel('Number of participants')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.set_title('Distribution of number of participants (10-30 participants)')

ax2.hist(age_temporary_data['n_males'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='blue', label='Males')
ax2.hist(age_temporary_data['n_females'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='red', label='Females')
ax2.set_xlabel('Number of participants')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.set_title('Distribution of number of participants (30-103 participants)')  

plt.show()

# %% [markdown]
# From the plot, we can notice that when women are involved in incidents, most of the time, there is only one woman, while in incidents with more than two participants of the same gender, it is more frequent for the participants to be men.
#
# Note that for 1567 entries in the dataset, we have the total number of participants, but we do not have the number of males and females

# %% [markdown]
# Below, we plot the distribution of the average age of participants in each incident.

# %%
plt.figure(figsize=(20, 8))
plt.hist(new_age_data['avg_age_participants'], bins=100, density=False, edgecolor='black', linewidth=0.8)
plt.xlim(0, 100)
plt.xlabel('Participants average age')
plt.ylabel('Frequency')
plt.title('Distribution of participants average age')
plt.show()

# %%
new_age_data.describe()


# %% [markdown]
# ## TAGS EXPLORATION:

# %%
#merge characteristics list

#characteristics1_frequency = incidents_data.pivot_table(columns=['incident_characteristics1'], aggfunc='size').sort_values(ascending=False)
#characteristics2_frequency = incidents_data.pivot_table(columns=['incident_characteristics2'], aggfunc='size').sort_values(ascending=False)
characteristics1_frequency = pd.Series.to_dict(incidents_data.pivot_table(columns=['incident_characteristics1'], aggfunc='size'))
characteristics2_frequency = pd.Series.to_dict(incidents_data.pivot_table(columns=['incident_characteristics2'], aggfunc='size'))

characteristics_frequency = {}
keys1 = list(characteristics1_frequency.keys())
keys2 = list(characteristics2_frequency.keys())

i = 0
j = 0
while i < len(characteristics1_frequency) and j < len(characteristics2_frequency):
    if keys1[i] > keys2[j]:
        characteristics_frequency[keys2[j]] = characteristics2_frequency[keys2[j]]
        j += 1
    elif keys1[i] == keys2[j]:
        characteristics_frequency[keys2[j]] = characteristics2_frequency[keys2[j]] + characteristics1_frequency[keys1[i]]
        i += 1
        j += 1
    else:
        characteristics_frequency[keys1[i]] = characteristics1_frequency[keys1[i]]
        i += 1

if(len(characteristics1_frequency) < len(characteristics2_frequency)):
    for j in range(len(characteristics1_frequency), len(characteristics2_frequency)):
        characteristics_frequency[keys2[j]] = characteristics2_frequency[keys2[j]]
elif(len(characteristics2_frequency) < len(characteristics1_frequency)):
    for i in range(len(characteristics2_frequency), len(characteristics1_frequency)):
        characteristics_frequency[keys1[i]] = characteristics1_frequency[keys1[i]]

characteristics_frequency = dict(sorted(characteristics_frequency.items(), key=lambda x:x[1])) # sort by value

# %%
characteristics_frequency_df = pd.DataFrame({'characteristics': list(characteristics_frequency.keys()), 'occurrences': list(characteristics_frequency.values())})

characteristics_frequency_df

# %%

fig = pd.DataFrame(characteristics_frequency_df).plot(kind='barh', figsize=(5, 18))
fig.set_yticklabels(characteristics_frequency_df['characteristics'])
fig.set_xscale("log")
plt.title("Counts of 'incident_characteristics'")
plt.xlabel('Count')
plt.ylabel('Incident characteristics')
plt.tight_layout()

# %%
characteristics_count_matrix = pd.crosstab(incidents_data['incident_characteristics2'], incidents_data['incident_characteristics1'])
fig, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(characteristics_count_matrix, cmap='coolwarm', ax=ax, xticklabels=True, yticklabels=True, linewidths=.5)
ax.set_xlabel('incident_characteristics2')
ax.set_ylabel('incident_characteristics1')  
ax.set_title('Counts of incident characteristics')
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(characteristics_count_matrix[["Shot - Dead (murder, accidental, suicide)"]].sort_values(by="Shot - Dead (murder, accidental, suicide)", inplace=False, ascending=False).tail(-1),
            cmap='coolwarm', yticklabels=True)

# %%
incidents_data[incidents_data['state']=='DISTRICT OF COLUMBIA'].size

# %% [markdown]
# We join the poverty data with the incidents data:

# %%
incidents_data['year'] = incidents_data['date'].dt.year # FIX: già fatto?
incidents_data = incidents_data.merge(poverty_data, on=['state', 'year'], how='left', validate="m:1")
incidents_data.head()

# %% [markdown]
# We join the elections data with the incidents data:

# %%
elections_data_copy = elections_data.copy()
elections_data_copy['year'] = elections_data_copy['year'] + 1
elections_data = pd.concat([elections_data, elections_data_copy], ignore_index=True)
incidents_data = incidents_data.merge(elections_data, on=['state', 'year', 'congressional_district'], how='left')
incidents_data.head()

# %%
incidents_data['month'] = incidents_data['date'].dt.month
incidents_data.groupby('month').size().plot(
    kind='bar',
    figsize=(10, 5),
    title='Number of incidents per month',
    xlabel='Month',
    ylabel='Number of incidents'
)
plt.xticks(range(12), calendar.month_name[1:13], rotation=45);

# %%
incidents_data['day_of_week'] = incidents_data['date'].dt.dayofweek
incidents_data.groupby('day_of_week').size().plot(
    kind='bar',
    figsize=(10, 5),
    title='Number of incidents per day of the week',
    xlabel='Day of the week',
    ylabel='Number of incidents'
)
plt.xticks(range(7), calendar.day_name[0:7], rotation=45);

# %%
usa_population = pd.read_csv(DATA_FOLDER_PATH + 'wikipedia/2010_United_States_census.csv')

# %%
usa_population.info()

# %%
usa_population.head()

# %%
usa_population.drop(columns=['Population as of 2000 census', 'Change', 'Percent change'], inplace=True)
usa_population.rename(columns={'Population as of 2010 census':'population', 'State': 'state'}, inplace=True)
usa_population['state'] = usa_population['state'].str.upper()
usa_population['population'] = usa_population['population'].str.replace(',', '').astype('int64')
incidents_data = incidents_data.merge(usa_population, on=['state'], how='left')
incidents_data.head()

# %%
incidents_per_state = incidents_data[incidents_data['year']<=2020].groupby(['state', 'population']).size()
incidents_per_state = ((incidents_per_state / incidents_per_state.index.get_level_values('population'))*100000).to_frame(name='incidents_per_100k_inhabitants').sort_values(by='incidents_per_100k_inhabitants', ascending=True)
incidents_per_state.reset_index(inplace=True)
incidents_per_state.plot(
    kind='barh',
    x='state',
    y='incidents_per_100k_inhabitants',
    figsize=(15, 10),
    ylabel='State',
    xlabel='Incidents per 100k inhabitants',
    title='Incidents per 100k inhabitants per state'
)

# %%
incidents_data[incidents_data['state']=='DISTRICT OF COLUMBIA'].groupby(['latitude', 'longitude', 'date']).size()[lambda x: x > 1].sort_values(ascending=False)

# %%
incidents_data.groupby(['latitude', 'longitude', 'date']).size()[lambda x: x>1]

# %%
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

word_cloud_all_train = WordCloud(
    width=1500,
    height=1200,
    stopwords=stopwords,
    collocations=False,
    background_color='white'
    ).generate(' '.join(incidents_data[incidents_data['notes'].notna()]['notes'].tolist()));

plt.imshow(word_cloud_all_train)
plt.axis('off')
plt.title('Word cloud of notes')

# %%
incidents_per_month_per_state = incidents_data.groupby(['state', 'month', 'year']).size()
incidents_per_month_per_state = incidents_per_month_per_state.to_frame(name='incidents').reset_index()
incidents_per_month_per_state = incidents_per_month_per_state.sort_values(by=['year', 'month', 'state'], ignore_index=True)
incidents_per_month_per_state['incidents_per_100k_inhabitants'] = incidents_per_month_per_state.apply(
    lambda row: (row['incidents'] / usa_population[usa_population['state']==row['state']]['population'].iloc[0])*100000,
    axis=1
)
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(
    incidents_per_month_per_state[incidents_per_month_per_state.year<=2020].pivot(
        index='state',
        columns=['year', 'month'],
        values='incidents_per_100k_inhabitants'
    ).fillna(0),
    cmap='coolwarm',
    ax=ax,
    xticklabels=True,
    yticklabels=True,
    linewidths=.5
)
ax.set_xlabel('Month-Year')
ax.set_ylabel('State')
ax.set_title('Number of incidents per month per state')

xticks = []
for label in ax.get_xticklabels():
    txt_label = label.get_text()
    month = txt_label[txt_label.find('-')+1:]
    year = txt_label[:txt_label.find('-')]
    xticks.append(year+' - '+calendar.month_name[int(month)])

ax.set_xticklabels(xticks);

plt.xticks(rotation=90)
plt.tight_layout() # 601,723 / 672,602

# %%
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(
    incidents_per_month_per_state[(incidents_per_month_per_state.year<=2020) & (incidents_per_month_per_state['state']!='DISTRICT OF COLUMBIA')].pivot(
        index='state',
        columns=['year', 'month'],
        values='incidents_per_100k_inhabitants'
    ).fillna(0),
    cmap='coolwarm',
    ax=ax,
    xticklabels=True,
    yticklabels=True,
    linewidths=.5
)
ax.set_xlabel('Month-Year')
ax.set_ylabel('State')
ax.set_title('Number of incidents per month per state')

xticks = []
for label in ax.get_xticklabels():
    txt_label = label.get_text()
    month = txt_label[txt_label.find('-')+1:]
    year = txt_label[:txt_label.find('-')]
    xticks.append(year+' - '+calendar.month_name[int(month)])

ax.set_xticklabels(xticks);

plt.xticks(rotation=90)
plt.tight_layout()

# %%
incidents_per_year_per_state = incidents_data.groupby(['state', 'year']).size()
incidents_per_year_per_state = incidents_per_year_per_state.to_frame(name='incidents').reset_index()
incidents_per_year_per_state['incidents_per_100k_inhabitants'] = incidents_per_year_per_state.apply(
    lambda row: (row['incidents'] / usa_population[usa_population['state']==row['state']]['population'].iloc[0])*100000,
    axis=1
)
fig = px.line(
    incidents_per_year_per_state[incidents_per_year_per_state.year<=2020].pivot(
        index='year',
        columns='state',
        values='incidents_per_100k_inhabitants'
    ),
    title='Number of incidents in the US over the years')
fig.show()

# %%
fig = px.line(
    incidents_per_year_per_state[(incidents_per_year_per_state.year<=2020) & (incidents_per_year_per_state['state']!='DISTRICT OF COLUMBIA')].pivot(
        index='year',
        columns='state',
        values='incidents_per_100k_inhabitants'
    ),
    title='Number of incidents in the US over the years')
fig.show()

# %%
incidents_per_state_2016 = incidents_data[incidents_data['year']==2016].groupby(['state', 'population', 'povertyPercentage']).size()
incidents_per_state_2016 = incidents_per_state_2016.to_frame(name='incidents').reset_index()
incidents_per_state_2016['incidents_per_100k_inhabitants'] = (incidents_per_state_2016['incidents'] / incidents_per_state_2016['population'])*100000
incidents_per_state_2016 = incidents_per_state_2016.merge(poverty_elections_data[['state', 'winningparty']], on=['state'], how='left')
incidents_per_state_2016['winningparty'] = incidents_per_state_2016['winningparty'].apply(lambda x: 'DEMOCRAT' if x==0 else 'REPUBLICAN') # TODO: fix, sembra che plotly abbia sovrascritto questo campo
fig = px.scatter(
    incidents_per_state_2016,
    x='povertyPercentage',
    y='incidents_per_100k_inhabitants',
    color='winningparty',
    hover_name='state',
    hover_data={'povertyPercentage': True, 'incidents_per_100k_inhabitants': True}
)
fig.show() # TODO: controllare se è giusto parlare di partito "vincente" e scrivere meglio

# %%
# TODO: da togliere, era per fare una prova (in generale ricordare che con px si possono usare solo scale colore continu e non è interattivo)
incidents_per_state = incidents_data.groupby(['year', 'state', 'population', 'povertyPercentage']).size()
incidents_per_state = incidents_per_state.to_frame(name='incidents').reset_index()
incidents_per_state['incidents_per_100k_inhabitants'] = (incidents_per_state['incidents'] / incidents_per_state['population'])*100000
incidents_per_state.state = incidents_per_state.state.astype("category").cat.codes
fig = px.parallel_coordinates(
    incidents_per_state,
    dimensions=['year', 'povertyPercentage', 'incidents_per_100k_inhabitants'],
    color="state"
)
fig.show()

# %% [markdown]
# We re-order the columns and we save the cleaned dataset:

# %%
incidents_data = incidents_data[[
    'date',
    'state',
    'px_code',
    'city_or_county',
    'address',
    'latitude',
    'longitude',
    'congressional_district',
    'state_house_district',
    'state_senate_district',
    'participant_age1',
    'participant_age_group1',
    'participant_gender1',
    'min_age_participants',
    'avg_age_participants',
    'max_age_participants',
    'n_participants_child',
    'n_participants_teen',
    'n_participants_adult',
    'n_males',
    'n_females',
    'n_killed',
    'n_injured',
    'n_arrested',
    'n_unharmed',
    'n_participants',
    'notes',
    'incident_characteristics1',
    'incident_characteristics2',
    'povertyPercentage',
    'party',
    'candidatevotes',
    'totalvotes',
    'candidateperc'
    ]]
#incidents_data.to_csv(DATA_FOLDER_PATH + 'incidents_cleaned.csv')

# %%
# da capire meglio come inserire il tutto

# create all the tags for each record
from tags_mapping import *

tagged_incidents_data = build_tagged_dataframe('../data/')

tagged_incidents_data

# %%
# add tag consistency column
tag_consistency_attr_name = "tag_consistency"
col = [True] * tagged_incidents_data.shape[0] #tag consistency assumed true
tagged_incidents_data.insert(tagged_incidents_data.shape[1], tag_consistency_attr_name, col)

# %%
from data_preparation_utils import check_consistency_tag

#consistency check
unconsistencies = 0
for index, record in tagged_incidents_data.iterrows():
    if not check_consistency_tag(record):
        tagged_incidents_data.at[index, tag_consistency_attr_name] = False
        unconsistencies += 1

print(unconsistencies)



# %%
# fare in modo che i tag vengano messi sui dati puliti
# concatenare date, particpanti, geo, e integrare con le cose dei distretti
