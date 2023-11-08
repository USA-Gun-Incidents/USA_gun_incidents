# -*- coding: utf-8 -*-
# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
#
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa

# %% [markdown]
# # Task 1 - Data Understanding and Preparation


# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px
import plotly.offline as pyo
import plotly.subplots as sp
import plotly.graph_objs as go
import math
import os
import calendar
import sys
sys.path.append(os.path.abspath('..')) # TODO: ???
from plot_utils import *
from sklearn.neighbors import KNeighborsClassifier
from geopy import distance as geopy_distance
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from pyproj import Transformer
import zipfile

# %% [markdown]
# We define constants and settings for the notebook:

# %%
# %matplotlib inline

DATA_FOLDER_PATH = '../data/'

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %% [markdown]
# ## Poverty Data

# %% [markdown]
# We load the dataset:

# %%
poverty_path = DATA_FOLDER_PATH + 'poverty_by_state_year.csv'
poverty_df = pd.read_csv(poverty_path)

# %% [markdown]
# We assess the correct loading of the dataset printing the first 2 rows:

# %%
poverty_df.head(n=2)

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
poverty_df.info()

# %% [markdown]
# We notice that:
# - the inferred types of the attributes are correct
# - the presence of missing values within the attribute `povertyPercentage`

# %% [markdown]
# We display descriptive statistics:

# %%
poverty_df.describe(include='all')

# %% [markdown]
# We notice that:
# - the data are provided also for the United States as a whole
# - `year` spans from 2004 to 2020

# %% [markdown]
# We check whether the tuple <`state`, `year`> uniquely identify each row:

# %%
poverty_df.groupby(['state', 'year']).size().max()==1

# %% [markdown]
# Since it does not, we display the duplicated <`state`, `year`> tuples:

# %%
poverty_df.groupby(['state', 'year']).size()[lambda x: x> 1]

# %% [markdown]
# We display the data for Wyoming, the only one with this issue:

# %%
poverty_df[(poverty_df['state']=='Wyoming')]

# %% [markdown]
# We notice that the entry relative to 2010 is missing. Since the other entries are ordered by year, we correct this error setting the year of the row with a povertyPercentage equal to 10.0 to 2010.

# %%
poverty_df.loc[
    (poverty_df['state'] == 'Wyoming') &
    (poverty_df['year'] == 2009) &
    (poverty_df['povertyPercentage'] == 10),
    'year'] = 2010

# %% [markdown]
# We check if each state has the expected number or rows:

# %%
(poverty_df.groupby('state').size()==(poverty_df['year'].max()-poverty_df['year'].min()+1)).all()

# %% [markdown]
# Since the tuple <`state`, `year`> uniquely identifies each row we can conclude that there are no missing rows.
#
# Now, we count how many rows have missing values:

# %%
poverty_df[poverty_df['povertyPercentage'].isnull()].shape[0]

# %% [markdown]
# Given that there are 52 unique values for the `state` attribute, data for a specific year is probably missing. To check this, we list the years with missing values.

# %%
poverty_df[poverty_df['povertyPercentage'].isnull()]['year'].unique()

# %% [markdown]
# As expected we have no data from 2012. Later we will fix this issue.
#
# Now we visualize the distribution of poverty percentage for each state.

# %%
poverty_df.boxplot(column='povertyPercentage', by='state', figsize=(20, 10), rot=90, xlabel='state', ylabel='Poverty (%)')
plt.suptitle('Poverty Percentage by State')
plt.title('')
plt.tight_layout()

# %% [markdown]
# This plot shows that Arkansas, Kentucky, Nebraska and North Dakota seems to be affected by fliers. We check this by plotting their poverty percentage over the years.

# %%
poverty_df[
    poverty_df['state'].isin(['Arkansas', 'Kentucky', 'Nebraska', 'North Dakota', 'United States'])
    ].pivot(index='year', columns='state', values='povertyPercentage').plot(kind='line')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Poverty (%)')
plt.title('Poverty (%) over the years')

# %% [markdown]
# The plot above shows that those fliers could be realistic values, we don't need to correct them.

# %%
poverty_df.groupby('year')['povertyPercentage'].mean().plot(kind='line', figsize=(15, 5), label='USA average', color='black', style='--')
plt.fill_between(
    poverty_df.groupby('year')['povertyPercentage'].mean().index,
    poverty_df.groupby('year')['povertyPercentage'].mean() - poverty_df.groupby('year')['povertyPercentage'].std(),
    poverty_df.groupby('year')['povertyPercentage'].mean() + poverty_df.groupby('year')['povertyPercentage'].std(),
    alpha=0.2,
    color='gray'
)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Poverty (%)')
plt.title('Average poverty (%) over the years')

# %% [markdown]
# We now plot the average poverty percentage over the years for each state:

# %%
poverty_df.groupby(['state'])['povertyPercentage'].mean().sort_values().plot(kind='bar', figsize=(15, 5))
plt.title(f'Average Poverty (%) in the period {poverty_df.year.min()}-{poverty_df.year.max()}')
plt.xlabel('State')
plt.ylabel('Average Poverty (%)')

# %% [markdown]
# It is evident that New Hampshire's average poverty rate is markedly lower than that of the other states, whereas Mississippi's average poverty rate is notably higher than the rest. 
#
# To inspect and compare the poverty percentage of each state over the year, we plot an interactive line chart:

# %%
fig = px.line(
    poverty_df.pivot(index='year', columns='state', values='povertyPercentage'),
    title='Poverty percentage in the US over the years')
fig.show()

# %% [markdown]
# We can oberserve that New Hampshire always had the lowest poverty percentage, whereas Mississippi had the highest till 2009, then it was surpassed by New Mexico and Louisiana.
#
# To imputate the missing data from 2012, we calculate the average of the `povertyPercentage` values for the preceding and succeeding year.

# %%
poverty_perc_2012 = poverty_df[poverty_df['year'].isin([2011, 2013])].groupby(['state'])['povertyPercentage'].mean()
poverty_df['povertyPercentage'] = poverty_df.apply(
    lambda x: poverty_perc_2012[x['state']] if x['year']==2012 else x['povertyPercentage'], axis=1
)

# %% [markdown]
# Now we plot again the interactive line chart:

# %%
fig = px.line(
    poverty_df.pivot(index='year', columns='state', values='povertyPercentage'),
    title='Poverty percentage in the US over the years')
pyo.plot(fig, filename='../html/lines_poverty.html', auto_open=False)
fig.show()

# %% [markdown]
# We also visualize how the poverty percentage changed with an animated map (to do this we need the alphanumeric codes associated to each state):

# %%
usa_states_df = pd.read_csv(
    'https://www2.census.gov/geo/docs/reference/state.txt',
    sep='|',
    dtype={'STATE': str, 'STATE_NAME': str}
)
usa_name_alphcode = usa_states_df.set_index('STATE_NAME').to_dict()['STUSAB']
poverty_df.sort_values(by=['state', 'year'], inplace=True)
poverty_df['px_code'] = poverty_df['state'].map(usa_name_alphcode) # retrieve the code associated to each state (the map is defined in the file data_preparation_utils.py)
fig = px.choropleth(
    poverty_df[poverty_df['state']!='United States'],
    locations='px_code',
    locationmode="USA-states",
    color='povertyPercentage',
    color_continuous_scale="rdbu",
    range_color=(
        min(poverty_df[poverty_df['state']!='United States']['povertyPercentage']),
        max(poverty_df[poverty_df['state']!='United States']['povertyPercentage'])),
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
pyo.plot(fig, filename='../html/animation_poverty.html', auto_open=False)
fig.show()

# %%
# TODO: usare unica color bar e aggiustare dimensioni in modo che si leggano gli stati?
# per ora lasciamo il codice qua sotto che magari ci serve per il report

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
# ## Elections Data

# %% [markdown]
# We load the dataset:

# %%
elections_path = DATA_FOLDER_PATH + 'year_state_district_house.csv'
elections_df = pd.read_csv(elections_path)

# %% [markdown]
# We assess the correct loading of the dataset printing the first 2 rows:

# %%
elections_df.head(n=2)

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
elections_df.info()

# %% [markdown]
# We notice that:
# - the inferred types are correct
# - there are no missing values (however, we should still assess whether there are any missing rows for specific years, states, or congressional districts)

# %% [markdown]
# We display descriptive statistics:

# %%
elections_df.describe(include='all')

# %% [markdown]
# We notice that:
# - year spans from 2004 to 2020
# - there are 6 unique parties
# - the minimum of candidatevotes and totalvotes are negative numbers, meaning that there are actually missing values

# %% [markdown]
# First we check if the triple <`year`, `state`, `congressional_district`> uniquely identifies each row:

# %%
elections_df.groupby(['year', 'state', 'congressional_district']).size().max() == 1

# %% [markdown]
# Then, we check if `candidatevotes` are always less or equal than `totalvotes`:

# %%
elections_df[elections_df['candidatevotes'] <= elections_df['totalvotes']].size == elections_df.size

# %% [markdown]
# We list the unique values in the column `state`:

# %%
states = elections_df['state'].unique()
states.sort()
print(f'States: {states}')
print(f'Number of states: {states.size}')

# %% [markdown]
# All the states (District of Columbia included) are present.
#
# We now display the states and the years for which there are missing rows:

# %%
years = [i for i in range(elections_df['year'].min(), elections_df['year'].max(), 2)]
for year in years:
    for state in states:
        if elections_df[(elections_df['state']==state) & (elections_df['year']==year)].size == 0:
            print(f"No data for '{state}' in {year}")

# %% [markdown]
# Except for District of Columbia, there are no missing rows.
# For District of Columbia we have only the following row:

# %%
elections_df[elections_df['state']=='DISTRICT OF COLUMBIA']

# %% [markdown]
# Missing values are probably due to the fact that District of Columbia is a non voting delegate district. Anyway, we gathered the missing values from Wikipedia. We noticed that as for the 2020 elecetions, the number of votes received by the winning party coincides, but the number of totalvotes is different (see [here](https://en.wikipedia.org/wiki/2020_United_States_House_of_Representatives_election_in_the_District_of_Columbia)). To be consistent with the other data, we replace the totalvotes value from 2020 with the one from Wikipedia.
#
# Now we import those data:

# %%
dc_elections_df = pd.read_csv('../data/wikipedia/district_of_columbia_house.csv')
dc_elections_df.head(n=2)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
dc_elections_df.info()

# %% [markdown]
# The inferred types are correct.
#
# We now merge the two dataframes:

# %%
elections_df.drop(elections_df[elections_df['state']=='DISTRICT OF COLUMBIA'].index, inplace=True)
elections_df = pd.concat([elections_df, dc_elections_df], ignore_index=True)
elections_df.sort_values(by=['year', 'state', 'congressional_district'], inplace=True, ignore_index=True)

# %% [markdown]
# We now check if congressional districts are numbered correctly (with '0' for states with only one congressional district, or with incremental values starting from '1' otherwise):

# %%
correct_numbering = True
for state in states:
    if state == 'DISTRICT OF COLUMBIA':
        continue
    for year in years:
        districts = elections_df[(elections_df['state']==state) & (elections_df['year']==year)]['congressional_district'].unique()
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
elections_df[
    (elections_df['totalvotes']>0)&(elections_df['year']>2012)
].boxplot(column='totalvotes', by='state', figsize=(20, 10), rot=90, xlabel='State', ylabel='Total votes')
plt.suptitle('Total votes from 2014')
plt.title('')
plt.tight_layout()

# %% [markdown]
# We can observe that for both total and candidate votes Florida, Louisian and Oklahoma have lower outliers, while Maine has an upper outlier. 
#
# We display the rows relative to Maine:

# %%
elections_df[(elections_df['year']>2013) & (elections_df['state']=='MAINE')]

# %% [markdown]
# We found in [Wikipedia](https://en.wikipedia.org/wiki/2022_United_States_House_of_Representatives_elections_in_Maine) that in Maine, that year, the Democratic party received 165136 votes out of a total of 311278 votes. We correct the error:

# %%
elections_df.loc[
    (elections_df['state']=='MAINE') &
    (elections_df['year']==2022) &
    (elections_df['congressional_district']==2),
    'candidatevotes'] = 165136
elections_df.loc[
    (elections_df['state']=='MAINE') &
    (elections_df['year']==2022) &
    (elections_df['congressional_district']==2),
    'totalvotes'] = 311278

# %%
elections_df[
    (elections_df['year']>2013) &
    (elections_df['state'].isin(['FLORIDA', 'LOUSIANA', 'OKLAHOMA'])) &
    ((elections_df['candidatevotes']<100) | (elections_df['totalvotes']<100))
]

# %% [markdown]
# We found in Wikipedia (e.g. [here](https://en.wikipedia.org/wiki/2014_United_States_House_of_Representatives_elections_in_Florida)), that for all the years and states above, no candidates filed to challenge the incumbent representative for their seat. Therefore, we will copy the `candidatevotes` and `totalvotes` values from the previous year:

# %%
for index, row in elections_df.iterrows():
    if row['candidatevotes'] < 2:
        replacement = elections_df[
            (elections_df['year']==row['year']-2) &
            (elections_df['state']==row['state']) &
            (elections_df['congressional_district']==row['congressional_district'])
        ]
        if replacement.size > 0:
            elections_df.at[index, 'candidatevotes'] = replacement['candidatevotes'].iloc[0]
            elections_df.at[index, 'totalvotes'] = replacement['totalvotes'].iloc[0]

# %% [markdown]
# We now plot the distribution of `totalvotes` (summing over the districts) after cleaning the data:

# %%
elections_df[
    elections_df['year']>2012
].groupby(['year', 'state']).agg('sum', numeric_only=True).boxplot(column='totalvotes', by='state', figsize=(20, 10), rot=90, xlabel='State', ylabel='Total votes')
plt.suptitle('Total votes from 2014')
plt.title('')
plt.tight_layout()

# %% [markdown]
# It is evident that in some states the number of votes fluctuates significantly from year to year.
#
# We get the unique names of the parties for the years of interest:

# %%
elections_df[
    (elections_df['year']>2012)
]['party'].unique()

# %% [markdown]
# The Democratic Farmer Labor is the affiliate of the Democratic Party in the U.S. state of Minnesota [[Wikipedia](https://en.wikipedia.org/wiki/Minnesota_Democratic–Farmer–Labor_Party)], hence we replace this party name with 'DEMOCRATIC' to ease later analysis.

# %%
elections_df['party'] = elections_df['party'].apply(
    lambda x: 'DEMOCRAT' if x=='DEMOCRATIC-FARMER-LABOR' else x
)

# %% [markdown]
# We now compute the percentage of votes obtained by the winner party and we plot the distribution of these percentages for the years of interest:

# %%
elections_df['candidateperc'] = (elections_df['candidatevotes']/elections_df['totalvotes'])*100

# %%
hist_box_plot(elections_df[elections_df['year']>2012], col='candidateperc', title='Percentage of winner votes')

# %% [markdown]
# It seems that in some districts the winner party obtained 100% of the votes. We disaply those districts:

# %%
elections_df[(elections_df['candidateperc']==100) & (elections_df['year']>2012)]

# %% [markdown]
# Wikipedia reports the same data, in those cases there was not an opponent party.
#
# The histogram above also shows that in some disticts the winner party obtained less than 50% of the votes. We display some of those districts:

# %%
elections_df[(elections_df['candidateperc']<=30) & (elections_df['year']>2012)]

# %% [markdown]
# Searching in [Wikipedia](https://en.wikipedia.org/wiki/2016_United_States_House_of_Representatives_elections_in_Louisiana) we found that the number of candidatevotes refers to the votes obtained by the winner at the final runoff (in which less people went to vote) while the number of totalvotes refers to the voter at the runoff plus the votes for the other candidates at the primary election. We won't correct these errors but we will keep it in mind for later analysis.

# %% [markdown]
# Now we compute, for each year and state, the party with the highest percentage of votes, so to have a better understanding of the political orientation of each state:

# %%
# FIX: data l'osservazione sopra questo dato e questo plot non hanno più significato
winning_party_per_state = elections_df.groupby(['year', 'state', 'party'])['candidateperc'].mean()
winning_party_per_state = winning_party_per_state.groupby(['year', 'state']).idxmax().apply(lambda x: x[2])
winning_party_per_state = winning_party_per_state.to_frame()
winning_party_per_state.reset_index(inplace=True)
winning_party_per_state.rename(columns={'candidateperc': 'majority_state_party'}, inplace=True)
winning_party_per_state['px_code'] = winning_party_per_state['state'].str.title().map(usa_name_alphcode) # District of Columbia won't be plotted because 'of' is written with capital 'O'
winning_party_per_state

# %% [markdown]
# We now plot on a map the winning party over the years:

# %%
fig = px.choropleth(
    winning_party_per_state[winning_party_per_state['year']>2004],
    locations='px_code',
    locationmode="USA-states",
    color='majority_state_party',
    scope="usa",
    animation_frame='year',
    title="Results of the elections over the years", 
    hover_name='state',
    hover_data={'px_code': False}
)
fig.update_layout(
    legend_title_text='Party'
)
pyo.plot(fig, filename='../html/animation_elections.html', auto_open=False)
fig.show()

# %% [markdown]
# ## Incidents Data

# %% [markdown]
# ### Preliminaries

# %% [markdown]
# We load the dataset:

# %%
incidents_path = DATA_FOLDER_PATH + 'incidents.csv'
incidents_df = pd.read_csv(incidents_path, low_memory=False)

# %% [markdown]
# We assess the correct loading of the dataset printing the first 2 rows:

# %%
incidents_df.head(n=2)

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
incidents_df.info()

# %% [markdown]
# We notice that:
# - `congressional_district`, `state_house_district`, `state_senate_district`, `participant_age1`, `n_males`, `n_females`, `n_arrested`, `n_unharmed`, `n_participants` are stored as `float64` while should be `int64`
# - `min_age_participants`, `avg_age_participants`, `max_age_participants`, `n_participants_child`, `n_participants_teen`, `n_participants_adult` are stored as `object` while should be `int64`, this probably indicates the presence of out of syntactic errors (not in the domain)
# - the presence of missing values within many attributes; the only attributes without missing values are the following: `date`, `state`, `city_or_county`, `n_killed`, `n_injured`, `n_participants`
#
# We display descriptive statistics of the DataFrame so to better understand how to cast the data:

# %%
incidents_df.describe(include='all')

# %% [markdown]
# We cast the attributes to the correct type:

# %%
# NUMERIC ATTRIBUTES

numerical_features = [
    'participant_age1',
    'n_males',
    'n_females',
    'n_killed',
    'n_injured',
    'n_arrested',
    'n_unharmed', 
    'n_participants',
    'min_age_participants',
    'avg_age_participants',
    'max_age_participants',
    'n_participants_child',
    'n_participants_teen',
    'n_participants_adult',
    # (the following attributes should be categorical, but for convenience we keep them numeric)
    'congressional_district',
    'state_house_district',
    'state_senate_district'
    ]
incidents_df[numerical_features] = incidents_df[numerical_features].apply(pd.to_numeric, errors='coerce')

# DATE
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')

# CATEGORICAL ATTRIBUTES
# nominal
incidents_df['participant_gender1'] = incidents_df['participant_gender1'].astype("category")
# ordinal
incidents_df['participant_age_group1'] = incidents_df['participant_age_group1'].astype(
    pd.api.types.CategoricalDtype(categories = ["Child 0-11", "Teen 12-17", "Adult 18+"], ordered = True))

# %% [markdown]
# We display again information about the dataset to check the correctness of the casting and the number of missing values:

# %%
incidents_df.info()

# %% [markdown]
# We observe that the downcasting of many attributes has not succeeded. This is due to the presence of missing or out of range values.
#
# We drop duplicates:

# %%
print(f"# of rows before dropping duplicates: {incidents_df.shape[0]}")
incidents_df.drop_duplicates(inplace=True) #, ignore_index=True) # TODO: geopy assume qui non sia stato resettato??
print(f"# of rows after dropping duplicates: {incidents_df.shape[0]}")

# %% [markdown]
# Now we visualize missing values:

# %%
fig, ax = plt.subplots(figsize=(12,8)) 
sns.heatmap(incidents_df.isnull(), cbar=False, xticklabels=True, ax=ax)

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
# We display descriptive statistics:

# %%
incidents_df.describe(include='all', datetime_is_numeric=True)

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
#
# In the following sections of this notebook we will analyze each attribute in detail.
#
# To avoid re-running some cells, we save checkpoints of the dataframe at different stages of the analysis and load the dataframe from the last checkpoint using the following functions:

# %%
LOAD_DATA_FROM_CHECKPOINT = True
CHECKPOINT_FOLDER_PATH = 'checkpoints/'

def save_checkpoint(df, checkpoint_name):
    df.to_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv')

def load_checkpoint(checkpoint_name, casting=None, date_cols=None):
    df = pd.read_csv(
        CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv',
        low_memory=False,
        index_col=0,
        parse_dates=date_cols,
        dtype=casting
        )
    return df

# %% [markdown]
# ### Date attribute: exploration and preparation

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

plot_dates(incidents_df['date'], title='Dates distribution')
print('Range data: ', incidents_df['date'].min(), ' - ', incidents_df['date'].max())
print('Unique years: ', sorted(incidents_df['date'].dt.year.unique()))
num_oor = incidents_df[incidents_df['date'].dt.year>2018].shape[0]
print(f'Number of rows with out of range value for the attribute date: {num_oor} ({num_oor/incidents_df.shape[0]*100:.2f}%)')

# %% [markdown]
# These plots show that the number of incidents with an out of range value for the attribute date is non negligible (9.6%) and, excluding these points, there are no incidents happened after the year 2018.
# Instead of discarding rows with out-of-range dates, we will try to correct the errors to prevent excessive data loss.
# Since there are no other features that could suggest the timeframe of the incident, we can only proceed using one of the following approaches:
# - check if those records have duplicates with a correct date
# - suppose dates were entered manually using a numeric keypad and that the errors are typos (e.g. 2030 is actually 2020)
# - replace the errors with the mean or median value
#
# Let's check if there are duplicates with a correct date:

# %%
incidents_future = incidents_df[incidents_df['date'].dt.year>2018].drop(columns=['date'])
incidents_past = incidents_df[incidents_df['date'].dt.year<2019].drop(columns=['date'])
incidents_past[incidents_past.isin(incidents_future).any(axis=1)].size!=0

# %% [markdown]
# Since there are no duplicates, we proceed with the second and third approach:

# %%
incidents_df['year'] = incidents_df['date'].dt.year
mean_date = incidents_df[incidents_df['year']<2019]['date'].mean()
median_date = incidents_df[incidents_df['year']<2019]['date'].median()

incidents_df['date_minus10'] = incidents_df['date']
incidents_df['date_minus10'] = incidents_df['date'].apply(lambda x : x - pd.DateOffset(years=10) if x.year>2018 else x)
incidents_df['date_minus11'] = incidents_df['date']
incidents_df['date_minus11'] = incidents_df['date'].apply(lambda x : x - pd.DateOffset(years=11) if x.year>2018 else x)
incidents_df['date_mean'] = incidents_df['date']
incidents_df['date_mean'] = incidents_df['date'].apply(lambda x : mean_date if x.year>2018 else x)
incidents_df['date_mean'] = pd.to_datetime(incidents_df['date_mean'], format='%Y-%m-%d') # discard hours, minutes and seconds
incidents_df['date_median'] = incidents_df['date']
incidents_df['date_median'] = incidents_df['date'].apply(lambda x : median_date if x.year>2018 else x)

# %%
plot_dates(incidents_df['date_minus10'], 'Dates distribution (year - 10 for oor)')
plot_dates(incidents_df['date_minus11'], 'Dates distribution (year - 11 for oor)', color='orange')
plot_dates(incidents_df['date_mean'], 'Dates distribution (oor replaced with mean)', color='green')
plot_dates(incidents_df['date_median'], 'Dates distribution (oor replaced with median)', color='red')

# %% [markdown]
# Unfortunately, these methods lead to unsatisfactory results, as they all remarkably alter the distribution. Therefore, we will keep the errors and take them into account in subsequent analyses. 

# %%
incidents_df.drop(columns=['date_minus10', 'date_minus11', 'date_mean', 'date_median'], inplace=True)
incidents_df['date_original'] = incidents_df['date']
incidents_df['date'] = incidents_df['date'].apply(lambda x : pd.NaT if x.year>2018 else x)
incidents_df['year'] = incidents_df['date'].dt.year.astype('UInt64')
incidents_df['month'] = incidents_df['date_original'].dt.month.astype('int64')
incidents_df['month_name'] = incidents_df['date_original'].dt.month_name()
incidents_df['day'] = incidents_df['date_original'].dt.day.astype('int64')
incidents_df['day_of_week'] = incidents_df['date_original'].dt.dayofweek.astype('UInt64')
incidents_df['day_of_week_name'] = incidents_df['date_original'].dt.day_name()

# %%
incidents_df.groupby('month').size().plot(
    kind='bar',
    figsize=(10, 5),
    title='Number of incidents per month',
    xlabel='Month',
    ylabel='Number of incidents'
)
plt.xticks(range(12), calendar.month_name[1:13], rotation=45);

# %%
incidents_df.groupby('day_of_week').size().plot(
    kind='bar',
    figsize=(10, 5),
    title='Number of incidents per day of the week',
    xlabel='Day of the week',
    ylabel='Number of incidents'
)
plt.xticks(range(7), calendar.day_name[0:7], rotation=45);


# %%
def group_by_day(df, date_col):
    counts_by_day = df[date_col].groupby([df[date_col].dt.year, df[date_col].dt.month, df[date_col].dt.day]).size().rename_axis(['year', 'month', 'day']).to_frame('Number of incidents').reset_index()
    counts_by_day[['year', 'month', 'day']] = counts_by_day[['year', 'month', 'day']].astype('int64')
    # add missing days
    for day in pd.date_range(start='2017-01-01', end='2017-12-31'): # 2017%4!=0, has not 29 days in February
        for year in counts_by_day['year'].unique():
            row_exists = (
                (counts_by_day['year']==year) &
                (counts_by_day['month']==day.month) &
                (counts_by_day['day']==day.day)
                ).any()
            if not row_exists:
                counts_by_day = pd.concat([
                        counts_by_day,
                        pd.DataFrame({'year': [year], 'month': [day.month], 'day': [day.day], 'Number of incidents': [0]})
                    ])
    counts_by_day.sort_values(by=['year', 'month', 'day'], inplace=True)
    counts_by_day['Day'] = counts_by_day.apply(lambda x: f'{x["day"]} {calendar.month_name[x["month"]]}', axis=1)
    return counts_by_day


# %%
incidents_counts_by_day = group_by_day(
    incidents_df[~((incidents_df['date'].dt.day==29) & (incidents_df['date'].dt.month==2))], # exclude 29 february
    'date_original'
)
fig = px.line(
    incidents_counts_by_day,
    x='Day',
    y='Number of incidents',
    title='Number of incidents per day',
    labels={'Day': 'Day of the year', 'Number of incidents': 'Number of incidents'},
    facet_col='year',
    width=1200,
    height=800,
    facet_col_wrap=3
)
fig.update_xaxes(tickangle=-90)
fig.show()
pyo.plot(fig, filename='../html/incidents_per_day.html', auto_open=False)

# %%
incidents_df['date_changed'] = incidents_df['date_original']
incidents_df['date_changed'] = incidents_df['date_changed'].apply(lambda x : x.replace(year=2018) if ((x.year==2028) | (x.year==2029) | (x.year==2030)) else x)
incidents_counts_by_day = group_by_day(
    incidents_df[~((incidents_df['date'].dt.day==29) & (incidents_df['date'].dt.month==2))],
    'date_changed'
)
fig = px.line(
    incidents_counts_by_day,
    x='Day',
    y='Number of incidents',
    title='Number of incidents per day',
    labels={'Day': 'Day of the year', 'Number of incidents': 'Number of incidents'},
    facet_col='year',
    width=1500,
    height=800,
    facet_col_wrap=3
)
fig.update_xaxes(tickangle=-90)
fig.show()


# %%
first_qtl_2016 = incidents_counts_by_day[incidents_counts_by_day['year']==2016]['Number of incidents'].quantile(q=0.05)
last_qtl_2016 = incidents_counts_by_day[incidents_counts_by_day['year']==2016]['Number of incidents'].quantile(q=0.95)
first_qtl_2017 = incidents_counts_by_day[incidents_counts_by_day['year']==2017]['Number of incidents'].quantile(q=0.05)
last_qtl_2017 = incidents_counts_by_day[incidents_counts_by_day['year']==2017]['Number of incidents'].quantile(q=0.95)

# %%
incidents_counts_by_day[(incidents_counts_by_day['year']==2016) & (incidents_counts_by_day['Number of incidents']<first_qtl_2016)].sort_values(by='Number of incidents', ascending=True)

# %%
incidents_counts_by_day[(incidents_counts_by_day['year']==2017) & (incidents_counts_by_day['Number of incidents']<first_qtl_2017)].sort_values(by='Number of incidents', ascending=True)

# %%
incidents_counts_by_day[(incidents_counts_by_day['year']==2016) & (incidents_counts_by_day['Number of incidents']>last_qtl_2016)].sort_values(by='Number of incidents', ascending=False)

# %%
incidents_counts_by_day[(incidents_counts_by_day['year']==2017) & (incidents_counts_by_day['Number of incidents']>last_qtl_2017)].sort_values(by='Number of incidents', ascending=False)

# %%
# TODO:
# commentare 1 gennaio, 29 febbraio, 4 luglio, 31 ottobre, 25 dicembre, ringraziamento
# aggiungere linea orizzontale per media (e quartili?)
# stampare le giornate con numero inncidenti <25qrt e >75qrt di ogni anno e mapparli su festività (ragionare al contrario)

# %%
incidents_df['month'] = incidents_df['date'].dt.month
incidents_df.groupby('month').size().plot(
    kind='bar',
    figsize=(10, 5),
    title='Number of incidents per month',
    xlabel='Month',
    ylabel='Number of incidents'
)
plt.xticks(range(12), calendar.month_name[1:13], rotation=45);

# %%
incidents_df['day_of_week'] = incidents_df['date'].dt.dayofweek
incidents_df.groupby('day_of_week').size().plot(
    kind='bar',
    figsize=(10, 5),
    title='Number of incidents per day of the week',
    xlabel='Day of the week',
    ylabel='Number of incidents'
)
plt.xticks(range(7), calendar.day_name[0:7], rotation=45);

# %% [markdown]
# ### Geospatial features: exploration and preparation

# %% [markdown]
# We check if the values of the attribute `state` are admissible comparing them with an official list of states:

# %%
states = incidents_df['state'].unique()
not_existing_states = False
missing_states = False

for state in states:
    if state not in usa_name_alphcode:
        not_existing_states = True
        print(f"State {state} does not exist")

for state in usa_name_alphcode:
    if state not in states:
        missing_states = True
        print(f"State {state} is missing")

if not_existing_states == False:
    print("All the values of the attribute 'states' are actually USA states (there are no misspelling or other errors).")
if missing_states == False:
    print("There is at least one incident for each USA state.")

# %% [markdown]
# We now check if, given a certain value for the attributes `latitude` and `longitude`, the attribute `city_or_county` has always the same value:

# %%
incidents_df.groupby(['latitude', 'longitude'])['city_or_county'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# That is not true and is due to the fact that sometimes the attribute `city_or_county` takes on the value of the city, other times the value of the county (as in the first row displayed above). Furthermore, we notice that even when the attribute refers to the same county it could be written in different ways (e.g. "Bethel (Newtok)", "Bethel (Napaskiak)", "Bethel"). 

# %% [markdown]
# We now check if a similar problem occurs for the attribute `address`:

# %%
incidents_df.groupby(['latitude', 'longitude'])['address'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# Still this attribute may be written in different ways (e.g. "Avenue" may also be written as "Ave", or "Highway" as "Hwy"). There could also be some errors (e.g. the same point corresponds to the address "33rd Avenue", "Kamehameha Highway" and "Kilauea Avenue extension").
#
# We plot on a map the location of the incidents:

# %%
fig = px.scatter_mapbox(
    lat=incidents_df['latitude'],
    lon=incidents_df['longitude'],
    zoom=0, 
    height=500,
    width=800
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %% [markdown]
# There are some points in China that are clearly wrong. We display the rows of the dataset that correspond to one of these points:

# %%
incidents_df[(incidents_df['latitude'] == 37.6499) & (incidents_df['longitude'] == 97.4331)]

# %% [markdown]
# That point has probably the correct values for the attributes `state` and `city_or_county`.

# %%
#FIXME: abbiamo usato geolocator = Nominatim(user_agent="?????"), assicurarsi abbia confini 2013-2020

# %% [markdown]
# To fix these inconsistencies we used the library [GeoPy]((https://geopy.readthedocs.io/en/stable/)). This library allows to retrieve the address (state, county, suburb, city, town, village, location name, and other features) corresponding to a given latitude and longitude. We queried the library using all the latitudes and longitudes of the points in the dataset and we saved the results in the CSV file we now load:

# %%
geopy_path = os.path.join(DATA_FOLDER_PATH, 'geopy/geopy_new.csv') # TODO: questo potraà diventare geopy (cancellare il vecchio geopy)
geopy_df = pd.read_csv(geopy_path, index_col=['index'], low_memory=False, dtype={})
geopy_df.head(n=2)

# %% [markdown]
# The rows in this dataframe correspond to the rows in the original dataset. Its column *coord_presence* is false if the corresponding row in the original dataset did not have latitude and longitude values.
#
# Among all the attributes returned by GeoPy, we selected and used the following:
#
# - *lat* and *lon*: Latitude and longitude of the location
# - *importance*: Numerical value $\in [0,1]$, indicates the importance of the location (in comparison to other locations)
# - *addresstype*: Address type (e.g., "house," "street," "postcode")
# - *state*: State of the location
# - *county*: County of the location
# - *suburb*: Suburb of the location
# - *city*: City of the location
# - *town*: Town of the location
# - *village*: Village of the location
# - *display_name*: User-friendly representation of the location, often formatted as a complete address. Used by us to cross-reference with the address in case we are unable to find a match between our data and the GeoPy data set using other information from the address.

# %%
print(f"Number of rows in which surburb is null: {geopy_df.loc[geopy_df['suburb_geopy'].isna()].shape[0]}\n")
print('Coordinate presence:')
display(geopy_df['coord_presence'].value_counts())
print('Importance presence:')
display(geopy_df['importance_geopy'].notna().value_counts())
print(f"Number of rows in which city is null and town is not null: {geopy_df[(geopy_df['city_geopy'].isnull()) & (geopy_df['town_geopy'].notnull())].shape[0]}\n")
print("Values of addresstype:")
print(geopy_df['addresstype_geopy'].unique())
print(f"\nNumber of rows in which addresstype is null: {geopy_df[geopy_df['addresstype_geopy'].isnull()].shape[0]}")

# %% [markdown]
# We also downloaded from [Wikipedia](https://en.wikipedia.org/wiki/County_(United_States)) the list of the counties (or their equivalent) in each state. 
#
# This data was used in cases where no consistency was found with GeoPy data. FIXME: come?
#
# When latitude and longitude where not available we used this information to check whether the county actually belonged to the state. FIXME: è questo che volevi dire con "This dataset made it possible to verify the data consistency for the *state* and *county* fields without the need for *latitude* and *longitude* values"?

# %%
counties_path = os.path.join(DATA_FOLDER_PATH, 'wikipedia/counties.csv')
counties_df = pd.read_csv(counties_path)
counties_df.head()

# %% [markdown]
# We now check and correct the consistency of the geographic data:

# %%
from TASK_1.data_preparation_utils import check_geographical_data_consistency

if LOAD_DATA_FROM_CHECKPOINT:
    incidents_df = load_checkpoint('checkpoint_1', date_cols=['date', 'date_original'])
else:
    geo_df = incidents_df[['state', 'city_or_county', 'address', 'latitude', 'longitude']]
    geo_df = pd.concat([geo_df, geopy_df.loc[incidents_df.index]], axis=1) # TODO: geopy ha più righe perchè tiene anche quelle dei vecchi duplicati????
    geo_df = geo_df.apply(lambda row: check_geographical_data_consistency(row, additional_data=counties_df), axis=1)
    incidents_df[geo_df.columns] = geo_df[geo_df.columns]
    save_checkpoint(incidents_df, 'checkpoint_1')

# %% [markdown]
# The function called above performs the following operations:
#
# - Converts to lowercase the values for *state*, *county*, and *city* in all the dataframes
# - If *city_or_county* contains values for both city and county, splits them into two different fields
# - Removes from *city_or_county* the words 'city of' and 'county' to avoid potential inconsistencies during distance calculations (this precaution is taken to identify if two strings are the same but contain typos, ensuring more accurate and consistent comparisons)
# - Removes from *city_or_county* punctuation and numerical values
# - Removes frequent words from *address* and *display_name* (e.g., "Street," "Avenue," "Boulevard") to avoid potential inconsistencies during distance calculations
#
# When latitude and longitude are available and therefore Geopy provided information for the corresponding location:
# - checks for equality between *state* and *state_geopy*
# - checks for equality between *county* and *county_geopy* or between *county* and *suburb_geopy*
# - checks for equality between *city* and *city_geopy*, or between *city* and *town_geopy*, or between *city* and *village_geopy*
#
# If these comparison fails, it checks for potential typos in the string. This is done using the Damerau-Levenshtein distance (see the definition below), with a threshold to decide the maximum distance for two strings to be considered equal. The thresholds were set after several preliminary tests. We decided to use different thresholds for state and city/county.
#
# If the comparison still fails, it compares the *address* field from our dataset with GeoPy's *display_name*. Again, the Damerau-Levenshtein distance with an appropriate threshold is used to verify address consistency.
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

# %% [markdown]
# #### Visualize Consistent Geographical Data

# %%
print('Number of rows with all null values: ', incidents_df.isnull().all(axis=1).sum())
print('Number of rows with null value for state: ', incidents_df['state'].isnull().sum())
print('Number of rows with null value for county: ', incidents_df['county'].isnull().sum())
print('Number of rows with null value for city: ', incidents_df['city'].isnull().sum())
print('Number of rows with null value for latitude: ', incidents_df['latitude'].isnull().sum())
print('Number of rows with null value for longitude: ', incidents_df['longitude'].isnull().sum())

# %%
sns.heatmap(incidents_df.isnull(), cbar=False, xticklabels=True)

# %% [markdown]
# After this check, all the entries in the dataset have at least the state value not null and consistent. Only 12,796 data points, which account for 4.76% of the dataset, were found to have inconsistent latitude and longitude values.  FIXME: ogni volta che c'è un percentuale (o qualsiasi numero in generale) nel markdown bisognerebbe averli calcolati e stampati nelle celle di codice precedenti

# %% [markdown]
# Below, we have included some plots to visualize the inconsistent values in the dataset.

# %%
incidents_df.groupby(['state_consistency','county_consistency','address_consistency']).count().sort_index(ascending=False)

# %%
stats = {}
stats_columns = ['#null_val', '#not_null', '#value_count']
for col in ['state', 'county', 'city', 'latitude', 'longitude', 'state_consistency',
       'county_consistency', 'address_consistency', 'location_importance', 'address_type']:
    stats[col] = []
    stats[col].append(incidents_df[col].isna().sum())
    stats[col].append(len(incidents_df[col]) - incidents_df[col].isna().sum())
    stats[col].append(len(incidents_df[col].value_counts()))
    
clean_geo_stat_stats = pd.DataFrame(stats, index=stats_columns).transpose()
clean_geo_stat_stats

# %%
geo_null_counts = [] # FIXME: mettere in dataframe come sopra
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].notna()) & (incidents_df['city'].notna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].notna()) & (incidents_df['city'].isna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].isna()) & (incidents_df['city'].notna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].isna()) & (incidents_df['city'].isna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].isna()) & (incidents_df['county'].notna()) & (incidents_df['city'].notna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].isna()) & (incidents_df['county'].notna()) & (incidents_df['city'].isna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].isna()) & (incidents_df['county'].isna()) & (incidents_df['city'].notna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].isna()) & (incidents_df['county'].isna()) & (incidents_df['city'].isna())]))

print('LAT/LONG     COUNTY     CITY             \t#samples')
print( 'not null    not null   not null         \t', geo_null_counts[0])
print( 'not null    not null   null             \t', geo_null_counts[1])
print( 'not null    null       not null         \t', geo_null_counts[2])
print( 'not null    null       null             \t', geo_null_counts[3])
print( 'null        not null   not null         \t', geo_null_counts[4])
print( 'null        not null   null             \t', geo_null_counts[5])
print( 'null        null       null             \t', geo_null_counts[6])
print( 'null        null       null             \t', geo_null_counts[7])
print('\n')
print( 'TOT samples                             \t', sum(geo_null_counts))
print( 'Samples with not null values for lat/lon\t', geo_null_counts[0]+geo_null_counts[1]+geo_null_counts[2])
print( 'Samples with null values for lat/lon    \t', geo_null_counts[4]+geo_null_counts[5]+geo_null_counts[6]+geo_null_counts[7])

# %%
dummy_df = incidents_df[incidents_df['latitude'].notna()] # FIXME: possiamo sostituire tutte le variabili 'dummy' con nomi più significativi? (anche se temporanee)
print('Number of entries with not null values for latitude and longitude: ', len(dummy_df))
plot_scattermap_plotly(dummy_df, 'state', zoom=2,)

# %%
dummy_df = incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].isna()) & 
    (incidents_df['city'].notna())]
print('Number of entries with not null values for county but not for lat/lon and city: ', len(dummy_df))
plot_scattermap_plotly(dummy_df, 'state', zoom=2, title='Missing county')

# %% [markdown]
# Visualize the number of entries for each city where we have the *city* value but not the *county* FIXME: dove stampiamo df dire 'display', visualize è più per le immagini

# %%
incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].isna()) & (incidents_df['city'].notna())].groupby('city').count()

# %%
incidents_df[(incidents_df['latitude'].notna()) & (incidents_df['city'].isna()) & (incidents_df['county'].isna())]

# %%
dummy_df = incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].notna()) & (incidents_df['city'].isna())]
print('Number of rows with null values for city, but not for lat/lon and county: ', len(dummy_df))
plot_scattermap_plotly(dummy_df, 'state', zoom=2, title='Missing city')

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
# #### Infer Missing City Values

# %% [markdown]
# For entries where we have missing values for *city* but not for *latitude* and *longitude*, we attempt to assign the *city* value based on the entry's distance from the centroid.

# %% [markdown]
# Visualize data group by *state*, *county* and *city*

# %%
incidents_df.groupby(['state', 'county', 'city']).size().reset_index(name='count')

# %% [markdown]
# Compute the centroid for each city and visualize the first 10 centroids in alphabetical order.

# %%
centroids = incidents_df.loc[incidents_df['latitude'].notna() & incidents_df['city'].notna()][[
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
    info_city = load_checkpoint('checkpoint_cities') # FIXME: uniformare ad altri?
else: # compute data
    for state, county, city in centroids.index:
        dummy = []
        for lat, long in zip(incidents_df.loc[(incidents_df['city'] == city) & 
            (incidents_df['state'] == state) & (incidents_df['county'] == county) & 
            incidents_df['latitude'].notna()]['latitude'], 
            incidents_df.loc[(incidents_df['city'] == city) & 
            (incidents_df['state'] == state) & (incidents_df['county'] == county) & 
            incidents_df['longitude'].notna()]['longitude']):
            dummy.append(geopy_distance.geodesic([lat, long], centroids.loc[state, county, city]).km)
            
        dummy = sorted(dummy)
        pc = np.quantile(dummy, np.arange(0, 1, 0.05))
        for i in range(len(info_city.columns) - 6):
            info_city.loc[state, county, city][i] = pc[i*2 + 1]
        info_city.loc[state, county, city][len(info_city.columns) - 6] = len(dummy) # FIXME: invece di -6,-5,... si possono mettere le stringhe?
        info_city.loc[state, county, city][len(info_city.columns) - 5] = min(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 4] = max(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 3] = sum(dummy)/len(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 2] = centroids.loc[state, county, city]['latitude']
        info_city.loc[state, county, city][len(info_city.columns) - 1] = centroids.loc[state, county, city]['longitude']
    save_checkpoint(info_city, 'checkpoint_cities') # save data 

# %%
info_city.head()

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
info_city.loc[info_city['tot_points'] > 1].info()

# %%
plot_scattermap_plotly(info_city, 'tot_points', x_column='centroid_lat', 
    y_column='centroid_lon', hover_name=False, zoom=2, title='Number of points per city') 
# FIXME: discretizzare e.g. <x, between(x, y), ...

# %% [markdown]
# We utilize the previously calculated data to infer missing values for the *city* field in entries of the dataset where latitude and longitude are available. The *city* field is assigned if the distance of the entry from the centroid falls within the third quartile of all points assigned to that centroid.

# %%
def substitute_city(row, info_city):
    if pd.isna(row['city']) and not np.isnan(row['latitude']):
        for state, county, city in info_city.index:
            if row['state'] == state and row['county'] == county: # FIXME: mettere in &
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
if LOAD_DATA_FROM_CHECKPOINT:
    incidents_df = load_checkpoint('checkpoint_2', date_cols=['date', 'date_original'])
else:
    incidents_df = incidents_df.apply(lambda row: substitute_city(row, info_city), axis=1)
    save_checkpoint(incidents_df, 'checkpoint_2')

# %%
incidents_df.head(2)

# %%
print('Number of rows with null values for city before: ', incidents_df['city'].isnull().sum())
print('Number of rows with null values for city: ', incidents_df['city'].isnull().sum())

# %% [markdown]
# From this process, we infer 2248 *city* values.

# %% [markdown]
# #### Visualize new data

# %%
geo_null_counts = [] # FIXME: mettere in dataframe come sopra
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].notna()) & (incidents_df['city'].notna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].notna()) & (incidents_df['city'].isna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].isna()) & (incidents_df['city'].notna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].isna()) & (incidents_df['city'].isna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].isna()) & (incidents_df['county'].notna()) & (incidents_df['city'].notna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].isna()) & (incidents_df['county'].notna()) & (incidents_df['city'].isna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].isna()) & (incidents_df['county'].isna()) & (incidents_df['city'].notna())]))
geo_null_counts.append(len(incidents_df.loc[(incidents_df['latitude'].isna()) & (incidents_df['county'].isna()) & (incidents_df['city'].isna())]))

print('LAT/LONG     COUNTY     CITY             \t#samples')
print( 'not null    not null   not null         \t', geo_null_counts[0])
print( 'not null    not null   null             \t', geo_null_counts[1])
print( 'not null    null       not null         \t', geo_null_counts[2])
print( 'not null    null       null             \t', geo_null_counts[3])
print( 'null        not null   not null         \t', geo_null_counts[4])
print( 'null        not null   null             \t', geo_null_counts[5])
print( 'null        null       null             \t', geo_null_counts[6])
print( 'null        null       null             \t', geo_null_counts[7])
print('\n')
print( 'TOT samples                             \t', sum(geo_null_counts))
print( 'Samples with not null values for lat/lon\t', geo_null_counts[0]+geo_null_counts[1]+geo_null_counts[2]+geo_null_counts[3])
print( 'Samples with null values for lat/lon    \t', geo_null_counts[4]+geo_null_counts[5]+geo_null_counts[6]+geo_null_counts[7])

# %%
plot_scattermap_plotly(incidents_df.loc[(incidents_df['latitude'].notna()) & 
    (incidents_df['county'].notna()) & (incidents_df['city'].isna())], 'state', zoom=2, title='Missing city')

# %%
#TODO: plottare le città inferite e i centroidi dello stesso colore e quelle che rimangono nan di nero

# %% [markdown]
# We check if the attribute `congressional_district` is numbered consistently (with '0' for states with only one congressional district). To do so we use the dataset containing the data about elections in the period of interest (congressional districts are redrawn when (year%10)==0):

# %%
at_large_states = elections_df[
    (elections_df['year'].between(2013, 2018, inclusive="both")) & 
    (elections_df['congressional_district']==0)
    ]['state'].unique()
at_large_states

# %% [markdown]
# Now we check if states with a '0' as congressional district are the same states with only one congressional district in the dataset containing the data about elections:

# %%
zero_congress_states_inc = incidents_df[incidents_df['congressional_district']==0]['state'].unique()
set(zero_congress_states_inc).issubset(set(at_large_states))

# %% [markdown]
# We check if states with a single congressional district are always numbered with '0' in the dataset containing the data about elections:

# %%
incidents_df[(incidents_df['state'] == at_large_states.any()) & (incidents_df['congressional_district']!=0)].size==0

# %% [markdown]
# Since they are not, we fix this issue:

# %%
incidents_df.loc[incidents_df['state'].isin(at_large_states), 'congressional_district'] = 0

# %%
incidents_df['state'].unique()

# %%
elections_df['state'].unique()

# %% [markdown]
# We check if the range of the attributes `congressional_district` is consistent with the number of congressional districts in the dataset containing the data about elections:

# %%
incidents_df['state'] = incidents_df['state'].str.upper()
wrong_congr_states = elections_df.groupby('state')['congressional_district'].max()>=incidents_df.groupby('state')['congressional_district'].max()
for state in wrong_congr_states[wrong_congr_states==False].index:
    print(f"State {state} has more districts in the incidents data than in the elections data")

# %% [markdown]
# We display the rows with inconsistent congressional district in Kentucky:

# %%
incidents_df[
    (incidents_df['state']=='KENTUCKY') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='KENTUCKY') & (elections_df['year']>2012)]['congressional_district'].max())
]

# %% [markdown]
# Searching online we found that Kentucky has 6 congressional districts, so we'll set to nan the congressional district for the row above:

# %%
incidents_df.loc[
    (incidents_df['state']=='KENTUCKY') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='KENTUCKY') & (elections_df['year']>2012)]['congressional_district'].max()),
    'congressional_district'] = np.nan

# %% [markdown]
# We display the rows with inconsistent congressional district in Oregon:

# %%
incidents_df[
    (incidents_df['state']=='OREGON') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='OREGON') & (elections_df['year']>2012)]['congressional_district'].max())
]

# %% [markdown]
# Searching online we found that Oregon has 5 congressional districts, so we'll set to nan the congressional district for the rows above:


# %%
incidents_df.loc[
    (incidents_df['state']=='OREGON') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='OREGON') & (elections_df['year']>2012)]['congressional_district'].max()),
    'congressional_district'] = np.nan 

# %% [markdown]
# We display the rows with inconsistent congressional district in West Virginia:

# %%
incidents_df[
    (incidents_df['state']=='WEST VIRGINIA') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='WEST VIRGINIA') & (elections_df['year']>2012)]['congressional_district'].max())
]

# %% [markdown]
# Searching online we found that West Virginia has 3 congressional districts, so we'll set to nan the congressional district for the row above:

# %%
incidents_df.loc[
    (incidents_df['state']=='WEST VIRGINIA') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='WEST VIRGINIA') & (elections_df['year']>2012)]['congressional_district'].max()),
    'congressional_district'] = np.nan

# %% [markdown]
# We check whether given a certain value for the attributes `latitude` and a `longitude`, the attribute `congressional_district` has always the same value:

# %%
incidents_df[incidents_df['congressional_district'].notnull()].groupby(['latitude', 'longitude'])['congressional_district'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# All these points are probably errors, due to the fact that they are near the border between two congressional districts. We correct them setting the most frequent value for the attribute `congressional_district` (setting that value also for the entries with missing values):

# %%
corrected_congr_districts = incidents_df[
    ~incidents_df['congressional_district'].isna()
    ].groupby(['latitude', 'longitude'])['congressional_district'].agg(lambda x: x.value_counts().index[0])
incidents_df = incidents_df.merge(corrected_congr_districts, on=['latitude', 'longitude'], how='left')
# where latitude and longitude are null, keep the original value
incidents_df['congressional_district_y'].fillna(incidents_df['congressional_district_x'], inplace=True)
incidents_df.rename(columns={'congressional_district_y':'congressional_district'}, inplace=True)
incidents_df.drop(columns=['congressional_district_x'], inplace=True)

# %% [markdown]
# In the same city or county there could be different values for the attribute `congressional_district` (this is not an error, is actually possible according to the USA law):

# %%
incidents_df[incidents_df['congressional_district'].notna()].groupby(['state', 'city_or_county'])['congressional_district'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# We print the unique values the attribute `state_house_district` can take on:

# %%
house_districts = incidents_df['state_house_district'].unique()
house_districts.sort()
house_districts

# %% [markdown]
# Also this attribute has some errors because the maximum number of state house districts should be 204 (for New Hampshire, see [here](https://ballotpedia.org/State_Legislative_Districts)). For now we won't correct this error beacuse this attribute is not useful for our analysis.
#
# We check if given a certain value for the attributes `latitude` and a `longitude`, the attribute `state_house_district` has always the same value:

# %%
incidents_df[incidents_df['state_house_district'].notnull()].groupby(
    ['latitude', 'longitude'])['state_house_district'].unique()[lambda x: x.str.len() > 1]

# %% [markdown]
# We correct the errors:

# %%
corrected_house_districts = incidents_df[
    incidents_df['state_house_district'].notnull()
    ].groupby(['latitude', 'longitude'])['state_house_district'].agg(lambda x: x.value_counts().index[0])
incidents_df = incidents_df.merge(corrected_house_districts, on=['latitude', 'longitude'], how='left')
incidents_df['state_house_district_y'].fillna(incidents_df['state_house_district_x'], inplace=True)
incidents_df.rename(columns={'state_house_district_y':'state_house_district'}, inplace=True)
incidents_df.drop(columns=['state_house_district_x'], inplace=True)

# %% [markdown]
# We now print the unique values the attribute `state_senate_district` can take on:

# %%
senate_districts = incidents_df['state_senate_district'].unique()
senate_districts.sort()
senate_districts

# %% [markdown]
# And again we notice some errors because the maximum number of state senate districts should be 67 (for Minnesota, see [here](https://ballotpedia.org/State_Legislative_Districts)). For now we won't correct this error beacuse this attribute is not useful for our analysis.
#
# We correct other possible errors as above:

# %%
corrected_senate_districts = incidents_df[
    incidents_df['state_senate_district'].notnull()
    ].groupby(['latitude', 'longitude'])['state_senate_district'].agg(lambda x: x.value_counts().index[0])
incidents_df = incidents_df.merge(corrected_senate_districts, on=['latitude', 'longitude'], how='left')
incidents_df['state_senate_district_y'].fillna(incidents_df['state_senate_district_x'], inplace=True)
incidents_df.rename(columns={'state_senate_district_y':'state_senate_district'}, inplace=True)
incidents_df.drop(columns=['state_senate_district_x'], inplace=True)

# %% [markdown]
# We check whether given a `state`, `city_or_county` and `state_senate_district`, the value of the attribute `congressional_district` is always the same:

# %%
incidents_df[incidents_df['congressional_district'].notnull()].groupby(
    ['state', 'city_or_county', 'state_senate_district'])['congressional_district'].unique()[lambda x: x.str.len() > 1].shape[0]==0

# %% [markdown]
# Hence we cannot recover the missing values for the attribute `congressional_district` from the values of `state_senate_district`. We check the same for the attribute `state_house_district`:

# %%
incidents_df[incidents_df['congressional_district'].notnull()].groupby(
    ['state', 'city_or_county', 'state_house_district'])['congressional_district'].unique()[lambda x: x.str.len() > 1].shape[0]==0

# %% [markdown]
# We cannot recover the missing values for the attribute `congressional_district` from the values of `state_house_district` either.
#
# We could, instead, recover the missing values from the entries with "similar" `latitude` and `longitude`. To explore this possibility we first plot on a map the dislocation of the incidents, coloring them according to the value of the attribute `congressional_district`:

# %%
plot_scattermap_plotly(
    incidents_df,
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
    incidents_df[incidents_df['state']=='ALABAMA'],
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
def build_X_y_for_district_inference(incidents_df):
    X_train = np.concatenate((
        incidents_df[
            (incidents_df['congressional_district'].notna()) &
            (incidents_df['latitude'].notna()) & 
            (incidents_df['longitude'].notna())
            ]['latitude'].values.reshape(-1, 1),
        incidents_df[
            (incidents_df['congressional_district'].notna()) & 
            (incidents_df['latitude'].notna()) & 
            (incidents_df['longitude'].notna())
            ]['longitude'].values.reshape(-1, 1)),
        axis=1
    )
    X_test = np.concatenate((
        incidents_df[
            (incidents_df['congressional_district'].isna()) & 
            (incidents_df['latitude'].notna()) & 
            (incidents_df['longitude'].notna())
            ]['latitude'].values.reshape(-1, 1),
        incidents_df[
            (incidents_df['congressional_district'].isna()) &
            (incidents_df['latitude'].notna()) & 
            (incidents_df['longitude'].notna())
            ]['longitude'].values.reshape(-1, 1)),
        axis=1
    )
    y_train = incidents_df[
        (incidents_df['congressional_district'].notna()) & 
        (incidents_df['latitude'].notna()) & 
        (incidents_df['longitude'].notna())
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
X_train, X_test, y_train = build_X_y_for_district_inference(incidents_df[incidents_df['state']=="ALABAMA"])
knn_clf = KNeighborsClassifier(n_neighbors=1, metric=geodesic_distance)
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
incidents_df['KNN_congressional_district'] = incidents_df['congressional_district']
incidents_df.loc[
    (incidents_df['state']=="ALABAMA") &
    (incidents_df['congressional_district'].isna()) &
    (incidents_df['latitude'].notna()) & 
    (incidents_df['longitude'].notna()),
    'KNN_congressional_district'
    ] = knn_pred

# %% [markdown]
# We plot the results:

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['state']=='ALABAMA'],
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
transformer = Transformer.from_crs("EPSG:4326", "EPSG:26929", always_xy=True) # EPSG:26929 identifies the projected coordinate system for Alabama East (had to choose between E,W,N,S)

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
alabama_color_map = { #TODO: NON SO DOVE SI ROMPE, adesso in y_train ha più di 7 valori
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
incidents_df.groupby(['state', 'congressional_district']).size()[lambda x: x <= 2]

# %% [markdown]
# By the way, missclassification can still occurr, depending on the position of the available examples w.r.t the position of the points to classify. Aware of this limitation, we proceed to apply this method to the other states and plot the result:

# %%
if LOAD_DATA_FROM_CHECKPOINT:
    incidents_df = load_checkpoint('checkpoint_3', date_cols=['date', 'date_original'])
else:
    for state in incidents_df['state'].unique():
        if state != "ALABAMA":
            print(f"{state} done.")
            X_train, X_test, y_train = build_X_y_for_district_inference(incidents_df[incidents_df['state']==state])
            if X_test.shape[0] == 0:
                continue
            knn_clf.fit(X_train, y_train)
            knn_pred = knn_clf.predict(X_test)
            incidents_df.loc[
                (incidents_df['state']==state) &
                (incidents_df['congressional_district'].isna()) &
                (incidents_df['latitude'].notna()) & 
                (incidents_df['longitude'].notna()),
                'KNN_congressional_district'
            ] = knn_pred
    incidents_df.drop(columns=['congressional_district'], inplace=True)
    incidents_df.rename(columns={'KNN_congressional_district':'congressional_district'}, inplace=True)
    save_checkpoint(incidents_df, 'checkpoint_3')

plot_scattermap_plotly(
    incidents_df,
    'congressional_district',
    zoom=2,
    height=800,
    width=800,
    title="USA Congressional districts (after inference)"
)

# %% [markdown]
# We now plot on a map the location of the incidents, coloring them according to the value of the attribute `state_senate_district` and `state_house_district`, to assess wheter we can apply the same method to recover missing values:

# %%
plot_scattermap_plotly(
    incidents_df,
    'state_senate_district',
    black_nan=True,
    zoom=2,
    height=800,
    width=800,
    title="USA State senate districts"
    )

plot_scattermap_plotly(
    incidents_df,
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
# ### Age, gender and number of participants: exploration and preparation

# %% [markdown]
# #### Features
# %%
incidents_df.groupby(['address']).size().sort_values(ascending=False)[:50].plot( #TODO: TOGLIERE(?)
    kind='bar',
    figsize=(10,6),
    title='Counts of the addresses with the 50 highest number of incidents'
)



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
participants_columns = ['participant_age1', 'participant_age_group1', 'participant_gender1', 
    'min_age_participants', 'avg_age_participants', 'max_age_participants',
    'n_participants_child', 'n_participants_teen', 'n_participants_adult', 
    'n_males', 'n_females',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants']
age_df = incidents_df[participants_columns]

# %%
age_df.head(10)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
age_df.info()

# %%
age_df['participant_age_group1'].unique()

# %% [markdown]
# Display the maximum and minimum ages, among the possible valid values, in the dataset. We have set a maximum threshold of 122 years, as it is the age reached by [Jeanne Louise Calment](https://www.focus.it/scienza/scienze/longevita-vita-umana-limite-biologico#:~:text=Dal%201997%2C%20anno%20in%20cui,ha%20raggiunto%20un%20limite%20biologico), the world's oldest person.

# %%
def max_min_value(attribute): # FIXME: convertire in float, escludere <= 122 e > 0 e usare la funzione max sulle colonne di interesse
    age = []
    for i in age_df[attribute].unique():
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
age_df[age_df['max_age_participants'] == '101.0']

# %% [markdown]
# We have set the maximum age threshold at 101 years.

# %% [markdown]
# We check if we have entries with non-null values for participant_age1 but NaN for participant_age_group1. 

# %%
age_df[age_df['participant_age1'].notna() & age_df['participant_age_group1'].isna()]

# %% [markdown]
# These 126 values can be inferred.

# %% [markdown]
# #### Studying Data Consistency

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
# Checks done to evaluate the consistency of data related to number of participants per gender and other participants class:
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
from TASK_1.data_preparation_utils import check_age_gender_data_consistency

if True:#LOAD_DATA_FROM_CHECKPOINT: # load data
    age_temporary_df = load_checkpoint('checkpoint_tmp')#, ['date', 'date_original']) # TODO: questa cosa è temporanea
else: # compute data
    age_temporary_df = age_df.apply(lambda row: check_age_gender_data_consistency(row), axis=1)
    save_checkpoint(age_temporary_df, 'checkpoint_tmp') # save data

# %% [markdown]
# #### Data Exploration without Out-of-Range Data

# %%
age_temporary_df.head(2)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
age_temporary_df.info()

# %% [markdown]
# We assess the correctness of the checks performed by printing the consistency variable for the first 5 rows and providing a concise summary of their most frequent values.

# %%
age_temporary_df[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].head(5)

# %%
age_temporary_df[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].describe()

# %% [markdown]
# Below, we print the number of rows with 'NaN' or inconsistent data.

# %%
print('Number of rows with null values: ', age_temporary_df[age_temporary_df['nan_values'] == True].shape[0])
print('Number of rows with inconsistent values in age data: ', age_temporary_df[age_temporary_df['consistency_age'] == False].shape[0])
print('Number of rows with inconsistent values in number of participants data: ', age_temporary_df[age_temporary_df[
    'consistency_n_participant'] == False].shape[0])
print('Number of rows with inconsistent values in gender data: ', age_temporary_df[age_temporary_df['consistency_gender'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 data: ', age_temporary_df[age_temporary_df[
    'consistency_participant1'] == False].shape[0])

# %%
print('Number of rows with inconsistent values for participants1: ', age_temporary_df[age_temporary_df[
    'consistency_participant1'] == False].shape[0])
print('Number of rows with NaN values for participants1: ', age_temporary_df[age_temporary_df[
    'consistency_participant1'] == np.nan].shape[0])
print('Number of rows with inconsistent values in participants1 wrt all other data: ', age_temporary_df[age_temporary_df[
    'consistency_participants1_wrt_n_participants'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age data: ', age_temporary_df[age_temporary_df[
    'participant1_age_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age range data: ', age_temporary_df[age_temporary_df[
    'participant1_age_range_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt gender data: ', age_temporary_df[age_temporary_df[
    'participant1_gender_consistency_wrt_all_data'] == False].shape[0])

# %%
age_temporary_df[(age_temporary_df['consistency_participant1'] == True) & (age_temporary_df[
    'participant1_age_range_consistency_wrt_all_data'] == False)].shape[0]

# %%
print('Number of rows with null values in age data: ', age_temporary_df[age_temporary_df['consistency_age'].isna()].shape[0])
print('Number of rows with null values in number of participants data: ', age_temporary_df[age_temporary_df[
    'consistency_n_participant'].isna()].shape[0])
print('Number of rows with null values in gender data: ', age_temporary_df[age_temporary_df['consistency_gender'].isna()].shape[0])
print('Number of rows with null values in participants1 data: ', age_temporary_df[age_temporary_df[
    'consistency_participant1'].isna()].shape[0])

# %%
print('Number of rows with all null data: ', age_temporary_df.isnull().all(axis=1).sum())

# %% [markdown]
# We can notice that:
# - The data in our dataset related to participant1, excluding the 1099 cases where age and age group data were inconsistent with each other and 190 cases where age range is not consistent, always appear to be consistent with the data in the rest of the dataset and can thus be used to fill in missing or incorrect data.
# - In the data related to age and gender, some inconsistencies are present, but they account for only 1.88% and 6.01% of the total dataset rows, respectively.
# - In 93779 rows, at least one field had a *NaN* value.

# %% [markdown]
# Since we noticed that some age data contained impossible values, we have set the age range between 0 and 100 years old. Below, we have verified this by printing the range.

# %%
print('Range age: ', age_temporary_df['min_age_participants'].min(), '-', age_temporary_df['max_age_participants'].max())

# %%
age_temporary_df[age_temporary_df['consistency_participant1'] == False].head(5)

# %% [markdown]
# We printed the distribution of participants1 in the age range when age was equal to 18 to verify that the majority of the data were categorized as adults.

# %%
age_df[age_df['participant_age1'] == 18]['participant_age_group1'].value_counts()

# %% [markdown]
# We plotted the age distribution of participant1 and compared it to the distribution of the minimum and maximum participants' age for each group.

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

ax0.hist(age_temporary_df['participant_age1'], bins=100, edgecolor='black', linewidth=0.8)
ax0.set_xlabel('Age')
ax0.set_ylabel('Frequency')
ax0.set_title('Distribution of age participant1')

ax1.hist(age_temporary_df['min_age_participants'], bins=100, edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of min age participants')

ax2.hist(age_temporary_df['max_age_participants'], bins=100, edgecolor='black', linewidth=0.8)
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of max age participants')

plt.show()

# %% [markdown]
# Observing the similar shapes of the distributions provides confirmation that the data pertaining to participant1 is accurate and reliable. Therefore, we can confidently use participant1's data to fill gaps in cases involving groups with a single participant.

# %% [markdown]
# We visualized the number of unique values for the cardinality of participants in each incident and provided a brief summary of this feature below.

# %%
print('Values of n_participants: ', age_temporary_df['n_participants'].unique())
display(age_temporary_df['n_participants'].describe())

# %% [markdown]
# From the data above, it is evident that the third quartile is equal to two participants, and the maximum number of participants per incident reaches the value of 103.
#
# Below, we have presented the distribution of the number of participants for each incident.

# %%
#distribution munber of participants
plt.figure(figsize=(20, 5))
plt.hist(age_temporary_df['n_participants'], bins=102, edgecolor='black', linewidth=0.8)
plt.xlabel('Number of participants')
plt.ylabel('Frequency (log scale)')
plt.xticks(np.arange(1, 104, 2))
plt.yscale('log')
plt.title('Distribution of number of participants')
plt.show()

# %% [markdown]
# Note that: y-axes is in logaritmic scale.

# %% [markdown]
# In the table below, we can see how many data related to the *number of participants* are clearly out of range, per age groups.

# %%
age_temporary_df[age_temporary_df['n_participants_adult'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
age_temporary_df[age_temporary_df['n_participants_teen'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
age_temporary_df[age_temporary_df['n_participants_child'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %% [markdown]
# Based on the tables above, we have evidence to set the maximum number of participants to 103.

# %% [markdown]
# We have provided additional information below for two of the rows with values out of range.

# %%
age_temporary_df.loc[35995]

# %%
age_temporary_df.iloc[42353]

# %% [markdown]
# This data visualization has been helpful in understanding the exceptions in the dataset and correcting them when possible, using other data from the same entry.
#
# In cases where we were unable to obtain consistent data for a certain value, we have set the corresponding field to *NaN*.

# %% [markdown]
# #### Fix Inconsistent Data

# %% [markdown]
# We have created a new DataFrame in which we have recorded the corrected and consistent data. Note that all these checks are performed based on the assumptions made in previous stages of the analysis.
#
# For entries with missing or inconsistent data, when possible, we have inferred or derived the missing values from other available data. Specifically:
#
# - In cases where we had the number of males (n_males) and number of females (n_females), we calculated the total number of participants as n_participants = n_males + n_females.
# - In instances with a single participant and consistent data for *participants1*, we used that data to derive values related to age (max, min, average) and gender.

# %%
from TASK_1.data_preparation_utils import set_gender_age_consistent_data

if LOAD_DATA_FROM_CHECKPOINT:
    with zipfile.ZipFile('checkpoints/checkpoint_4.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('checkpoints/') # TODO: magari fare all'inizio una chiamata che decomprime tutti i *.zip
    incidents_df = load_checkpoint('checkpoint_4', date_cols=['date', 'date_original'])
else:
    new_age_df = age_temporary_df.apply(lambda row: set_gender_age_consistent_data(row), axis=1)
    incidents_df[new_age_df.columns] = new_age_df[new_age_df.columns]
    save_checkpoint(incidents_df, 'checkpoint_4')

# %% [markdown]
# We display the first 2 rows and a concise summary of the DataFrame:

# %%
incidents_df.head(2)

# %%
incidents_df.info()

# %%
print('Number of rows in which all data are null: ', incidents_df.isnull().all(axis=1).sum())
print('Number of rows with some null data: ', incidents_df.isnull().any(axis=1).sum())
print('Number of rows in which number of participants is null: ', incidents_df[incidents_df['n_participants'].isnull()].shape[0])
print('Number of rows in which number of participants is 0: ', incidents_df[incidents_df['n_participants'] == 0].shape[0])
print('Number of rows in which number of participants is null and n_killed is not null: ', incidents_df[
    incidents_df['n_participants'].isnull() & incidents_df['n_killed'].notnull()].shape[0])

# %%
print('Total rows with null value for n_participants: ', incidents_df['n_participants'].isnull().sum())
print('Total rows with null value for n_participants_child: ', incidents_df['n_participants_child'].isnull().sum())
print('Total rows with null value for n_participants_teen: ', incidents_df['n_participants_teen'].isnull().sum())
print('Total rows with null value for n_participants_adult: ', incidents_df['n_participants_adult'].isnull().sum())
print('Total rows with null value for n_males: ', incidents_df['n_males'].isnull().sum())
print('Total rows with null value for n_females: ', incidents_df['n_females'].isnull().sum())

# %% [markdown]
# We can observe that for any entries in the dataset, all data related to age and gender are *NaN*, while for 98973 entries, almost one value is *NaN*. From the plot below, we can visualize the null values (highlighted).
#
# It's important to note that we have complete data for *n_killed* and *n_injured* entries, and the majority of missing data are related to age-related features.

# %%
sns.heatmap(incidents_df.isnull(), cbar=False)

# %% [markdown]
# Below, we have provided the distribution of the total number of participants and the number of participant per age range for each incident. Once again, to make the histograms more comprehensible use a logaritmic scale for y-axes.

# %%
# distribuition number of participants
plt.figure(figsize=(20, 5))
plt.hist(incidents_df['n_participants'], bins=104, edgecolor='black', linewidth=0.8)
plt.xlabel('Number of participants')
plt.ylabel('Frequency (log scale)')
plt.xticks(np.arange(1, 104, 2))
plt.yscale('log')
plt.title('Distribution of number of participants')
plt.show()

# %%
print('Max number of participants: ', incidents_df['n_participants'].max())
print('Max number of children: ', incidents_df['n_participants_child'].max())
print('Max number of teens: ', incidents_df['n_participants_teen'].max())
print('Max number of adults: ', incidents_df['n_participants_adult'].max())

# %%
incidents_df[incidents_df['n_participants_adult'] > 60][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
# distribuition number of participants per age group 
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(20, 12), sharex=True, sharey=True)

ax0.bar(incidents_df['n_participants_child'].value_counts().index, incidents_df['n_participants_child'].value_counts(),
    alpha=0.8, color='magenta', edgecolor='black', linewidth=0.8, label='Children')
ax0.legend()
ax1.bar(incidents_df['n_participants_teen'].value_counts().index, incidents_df['n_participants_teen'].value_counts(),
    alpha=0.8, color='red', edgecolor='black', linewidth=0.8, label='Teen')
ax1.legend()
ax2.bar(incidents_df['n_participants_adult'].value_counts().index, incidents_df['n_participants_adult'].value_counts(),
    color='orange', edgecolor='black', linewidth=0.8, label='Adult')
ax2.legend()

plt.xlim(-1, 64)
plt.xticks(range(0, 64))
plt.yscale('log')
plt.xlabel('Number of participants')
ax0.set_ylabel('Number of incidents')
ax1.set_ylabel('Numer of incidents')
ax2.set_ylabel('Numer of incidents')
ax0.set_title('Number of participants for each incident per age')
plt.show()

# %% [markdown]
# We observe that in incidents involving children and teenagers under the age of 18, the total number of participants was less than 7 and 27, respectively. In general, incidents involving a single person are much more frequent than other incidents, and most often, they involve teenagers and children, with a smaller percentage involving adults. On the other hand, incidents with multiple participants mostly consist of adults, and as the number of participants increases, the frequency of such incidents decreases. 
#
# Note that the y-axis of the histograms are in logaritmic scale.

# %% [markdown]
# We also plot the distribution of the number of incidents per gender:

# %%
# distribuition number of participants per gender
plt.figure(figsize=(20, 5))
plt.bar(incidents_df['n_males'].value_counts().index-0.2, incidents_df['n_males'].value_counts(), 0.4,
    edgecolor='black', linewidth=0.8, label='Males participants')
plt.bar(incidents_df['n_females'].value_counts().index+0.2, incidents_df['n_females'].value_counts(), 0.4,
    edgecolor='black', linewidth=0.8, label='Females participants')
plt.xticks(range(0, 64))
plt.yscale('log')
plt.xlabel('Number of participants')
plt.ylabel('Number of incidents')
plt.legend()
plt.title('Number of participants for each incident per gender')
plt.show()

# %% [markdown]
# Note that for 1567 entries in the dataset, we have the total number of participants, but we do not have the number of males and females
# and that the y-axis of the histogram is in logaritmic scale.

# %% [markdown]
# Below, we plot the distribution of the average age of participants in each incident.

# %%
plt.figure(figsize=(20, 8))
plt.hist(incidents_df['avg_age_participants'], bins=100, density=False, edgecolor='black', linewidth=0.8) # FIXME: provare + binning (magare anche sturges's rule)
plt.xlim(0, 100)
plt.xlabel('Participants average age')
plt.ylabel('Frequency')
plt.title('Distribution of participants average age')
plt.show()

# %%
incidents_df.describe()

# %% [markdown]
# ### Incident characteristics features: exploration and preparation

# %%
# FIXME: aggiungere commenti + ricontrollare quando si usa incedeints_df e quando final_incidents_df

# %%
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

word_cloud_all_train = WordCloud(
    width=1500,
    height=1200,
    stopwords=stopwords,
    collocations=False,
    background_color='white'
    ).generate(' '.join(incidents_df[incidents_df['notes'].notna()]['notes'].tolist()));

plt.imshow(word_cloud_all_train)
plt.axis('off')
plt.title('Word cloud of notes')

# %% [markdown]
# We check if given the first characteristic of a record, the second one is different. This to ensure that the info we have are not redundant.

# %%
# check if ch1 and ch2 are always different
incidents_df[incidents_df['incident_characteristics1']==incidents_df['incident_characteristics2']].shape[0]==0

# %% [markdown]
# We check the frequency of each characteristic in the whole dataset, then we plot it.

# %%
# merge characteristics list
ch1_counts = incidents_df['incident_characteristics1'].value_counts()
ch2_counts = incidents_df['incident_characteristics2'].value_counts()
ch_counts = ch1_counts.add(ch2_counts, fill_value=0).sort_values(ascending=True)
ch_counts

# %%
fig = ch_counts.plot(kind='barh', figsize=(5, 18))
fig.set_xscale("log")
plt.title("Counts of 'incident_characteristics'")
plt.xlabel('Count')
plt.ylabel('Incident characteristics')
plt.tight_layout()

# %%
ch1_females_counts = incidents_df[incidents_df['n_females']>1]['incident_characteristics1'].value_counts()
ch2_females_counts = incidents_df[incidents_df['n_females']>1]['incident_characteristics2'].value_counts()
ch_females_counts = ch1_females_counts.add(ch2_females_counts, fill_value=0).sort_values(ascending=False).plot(
    kind='bar',
    title='Characteristics counts of incidents with females involved',
    figsize=(20,10)
)

# %%
characteristics_count_matrix = pd.crosstab(incidents_df['incident_characteristics2'], incidents_df['incident_characteristics1'])
fig, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(characteristics_count_matrix, cmap='coolwarm', ax=ax, xticklabels=True, yticklabels=True, linewidths=.5)
ax.set_xlabel('incident_characteristics1')
ax.set_ylabel('incident_characteristics2')  
ax.set_title('Counts of incident characteristics')
plt.tight_layout()

# %% [markdown]
# We plot, for a specific characteristic, the most common characteristic that is paired to it.

# %%
fig, ax = plt.subplots(figsize=(20, 15)) # FIX: questo plot possiamo toglierlo? dovrebbe contenere la stessa informazione di quello sotto
sns.heatmap(characteristics_count_matrix[["Shot - Dead (murder, accidental, suicide)"]].sort_values(by="Shot - Dead (murder, accidental, suicide)", inplace=False, ascending=False).tail(-1),
            cmap='coolwarm', yticklabels=True)

# %%
characteristics_count_matrix[["Shot - Dead (murder, accidental, suicide)"]].sort_values(
    by="Shot - Dead (murder, accidental, suicide)",
    inplace=False,
    ascending=False).plot(
        kind='bar',
        figsize=(20,10)
    )

# %% [markdown]
# We can see that the most of the other characteristics are not paired to the one we're analyzing, in particular there are very few ones which are paired to it for a significant number of times.

# %% [markdown]
# ### Tags
# We create some binary tags to standardize the characteristics of each incident, in this way we can easyly get detailed information on the type of all the incidents.<br>
# We based the tags creation only on the info we could get from the characteristics. The logic behind the conversion is that the tag is true if and only if we could get the semantic information of the tag with 100% certainty; if the tag is false, it means that or the tag represent info wich are false for the specific record, or that we don't have enough data to assume something for that particular incident.<br>
# The tags we cretaed are the following:
# - <b>firearm</b>: it tells if a firearm is involved in the incident
# - <b>air_gun</b>: it tells if an air gun is involved in the incident
# - <b>shots</b>: it tells if it was an incident involving one or more shootings
# - <b>aggression</b>: it tells if there was an aggression (both using a gun or not) in the incident
# - <b>suicide</b>: it tells if the incident involve a suicide (attempts are included)
# - <b>injuries</b>: it tells if there was one ore more injuried subjects in the incident
# - <b>death</b>: it tells if there was one ore more deaths in the incident
# - <b>road</b>: it tells if the incident involves a bad street behavior
# - <b>illegal_holding</b>: it tells if the incident involve a stealing act or an illegaly possessed gun
# - <b>house</b>: it tells if the incident is happened in a house
# - <b>school</b>: it tells if the incident is happened next to a school
# - <b>children</b>: it tells if the incident involves one or more children
# - <b>drugs</b>: it tells if the incident involves drug interests
# - <b>officers</b>: it tells if one or more officiers are involved in the incident
# - <b>organized</b>: it tells if the incident was planned by an organization or a group
# - <b>social_reasons</b>: it tells if the incident involves social discriminations or terrorism
# - <b>defensive</b>: it tells if there was a defensive use of a gun during the incident
# - <b>workplace</b>: it tells if the incident happened in a workplace
# - <b>abduction</b>: it tells if the incident involves any form of abduction
# - <b>unintentional</b>: it tells if the incident was unintentional

# %% [markdown]
# We set all the tags and check their consistency w.r.t. the other data of the record.

# %%
from TASK_1.data_preparation_utils import add_tags, check_tag_consistency, check_characteristics_consistency, IncidentTag

tags_columns = [tag.name for tag in IncidentTag]
tags_columns.append('tag_consistency')

if LOAD_DATA_FROM_CHECKPOINT:
    with zipfile.ZipFile('checkpoints/checkpoint_5.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('checkpoints/')
    incidents_df = load_checkpoint('checkpoint_5', date_cols=['date', 'date_original'])
else:
    incidents_df = add_tags(incidents_df)
    incidents_df['tag_consistency'] = True
    incidents_df = incidents_df.apply(lambda row: check_tag_consistency(row), axis=1)
    incidents_df = incidents_df.apply(lambda row: check_characteristics_consistency(row), axis=1)
    save_checkpoint(incidents_df, 'checkpoint_5')

# %%
incidents_df['tag_consistency'].value_counts()

# %% [markdown]
# We correct the inconsistencies and we save again the dataset. Then we check again to see if there are any improvement.

# %%
from TASK_1.data_preparation_utils import set_tags_consistent_data

if LOAD_DATA_FROM_CHECKPOINT:
    with zipfile.ZipFile('checkpoints/checkpoint_6.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('checkpoints/')
    incidents_df = load_checkpoint('checkpoint_6', date_cols=['date', 'date_original'])
else:
    incidents_df = incidents_df.apply(lambda row: set_tags_consistent_data(row), axis=1)
    incidents_df = incidents_df.apply(lambda row: check_tag_consistency(row), axis=1)
    incidents_df = incidents_df.apply(lambda row: check_characteristics_consistency(row), axis=1)
    save_checkpoint(incidents_df, 'checkpoint_6')

# %%
incidents_df['tag_consistency'].value_counts()

# %% [markdown]
# We create 5 partitions: Murder, Suicide, Defensive, Accidental and Others. Then we show grafically, in percentage, how many incidents belong to each class.

# %%
tags_counts = {}
tags_counts['Murder'] = incidents_df[
    (incidents_df['death']==True) &
    ((incidents_df['aggression']==True) |
    (incidents_df['social_reasons']==True))].shape[0] # not accidental nor defensive
tags_counts['Suicide'] = incidents_df[
    (incidents_df['death']==True) &
    (incidents_df['suicide']==True)].shape[0] # warninig: if murder/suicide is counted twice
tags_counts['Defensive'] = incidents_df[
    (incidents_df['death']==True) &
    (incidents_df['defensive']==True)].shape[0]
tags_counts['Accidental'] = incidents_df[
    (incidents_df['death']==True) &
    (incidents_df['unintentional']==True)].shape[0]
tags_counts['Others or not known'] = incidents_df[
    (incidents_df['death']==True) &
    (incidents_df['aggression']==False) &
    (incidents_df['social_reasons']==False) &
    (incidents_df['unintentional']==False)].shape[0]

fig, ax = plt.subplots()
total = sum(tags_counts.values())
ax.pie(tags_counts.values())
legend_labels = [f'{label}: {(size/total)*100:.1f}%' for label, size in tags_counts.items()]
plt.legend(legend_labels)
plt.title("Gun incidents")
plt.show()

# %% [markdown]
# We can see that the biggest part of the incidents involves Murder, while even if Suicide, Defensive and Accidental are the most common, they're very few compare to murders. The other big slice of the pie belongs to Others, showing that there are a lot of different incidents that are less common.

# %% [markdown]
# We show which are the most common tags. In particular we display, for each tag, in how many records it's set to True w.r.t. all the records.

# %%
ax = (incidents_df[tags_columns].apply(lambda col: col.value_counts()).T.sort_values(by=True)/incidents_df.shape[0]*100).plot(kind='barh', stacked=True, alpha=0.8, edgecolor='black')
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=8)
plt.title("Incidents characteristic (%)")

# %% [markdown]
# We can see that the most common tags are firearm, shots, aggression and injuries (above 50% of the records), in particular firearm is True for almost every record (97.8 %). On the other hand there are tags that are very rare (less than 1% of the records): air_gun, school, social_reasons and abduction.

# %% [markdown]
# We try to see if there are correlations between accidental incidents and the presence of children.

# %%
# compute correlation between accidental incidents and presence of children
incidents_df['unintentional'].corr(incidents_df['n_participants_child']>0) # not correlated

# %% [markdown]
# We can see that the two events are not correlated
# %%
incidents_df.groupby(['address']).size().sort_values(ascending=False)[:50].plot(
    kind='bar',
    figsize=(10,6),
    title='Counts of the addresses with the 50 highest number of incidents'
) # many airports!!

# %% [markdown]
# We display the most common characteristics for incidents involving women.

# %%
ch1_females_counts = incidents_df[incidents_df['n_females']>1]['incident_characteristics1'].value_counts()
ch2_females_counts = incidents_df[incidents_df['n_females']>1]['incident_characteristics2'].value_counts()
ch_females_counts = ch1_females_counts.add(ch2_females_counts, fill_value=0).sort_values(ascending=False).plot(
    kind='bar',
    title='Characteristics counts of incidents with females involved',
    figsize=(20,10)
)

# %% [markdown]
# We can see that the distribution is very similar to the one involving both men and women. Some of the main differences are that, for women, the frequency of suicides is higher than normal, while the officiers involving incidents have the opposite behavior.

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['n_killed']>=4],
        zoom=2,
        title='Mass shootings'
)

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['children']==True],
    zoom=2,
    title='Incidents involving children'
)

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['suicide']==True],
    zoom=2,
    title='Suicides'
)

# %% [markdown]
# We are aware of the fact that we could use classifier to inferr missing values. We chose not to do it because we think such method do not align with the nature of gun incidents. Citando il libro "Classification is the task of learning a target function f that maps each attribute set x to one of the predefined class labels y", il problema è che non può esistere una tale funzione (possono esserci (e immagino siano anche molti comuni) record uguali su tutti gli attributi tranne uno, per cui l'inferenza è impossibile).

# %% [markdown]
# ## Joint analysis of the datasets

# %% [markdown]
# We join the poverty data with the incidents data:

# %%
poverty_df['state'] = poverty_df['state'].str.upper()
incidents_df = incidents_df.merge(poverty_df, on=['state', 'year'], how='left', validate="m:1")
incidents_df.head()

# %% [markdown]
# We join the elections data with the incidents data:

# %%
elections_df_copy = elections_df.copy()
elections_df_copy['year'] = elections_df_copy['year'] + 1
elections_df = pd.concat([elections_df, elections_df_copy], ignore_index=True)
incidents_df = incidents_df.merge(elections_df, on=['state', 'year', 'congressional_district'], how='left')
incidents_df.head()

# %% [markdown]
# We read and join the data about the USA population from the 2010 census downloaded from [Wikipedia](https://en.wikipedia.org/wiki/2010_United_States_census). This time we won't use the ACS population data because we simply need aggregated data for each state over the period of interest.

# %%
usa_population_df = pd.read_csv(DATA_FOLDER_PATH + 'wikipedia/2010_United_States_census.csv')

# %%
usa_population_df.info()

# %%
usa_population_df.head()

# %%
usa_population_df.drop(columns=['Population as of 2000 census', 'Change', 'Percent change', 'Rank'], inplace=True) # FIX: fare solo cu col
usa_population_df.rename(columns={'Population as of 2010 census':'population_state_2010', 'State': 'state'}, inplace=True)
usa_population_df['state'] = usa_population_df['state'].str.upper()
usa_population_df['population_state_2010'] = usa_population_df['population_state_2010'].str.replace(',', '').astype('int64')
incidents_df = incidents_df.merge(usa_population_df, on=['state'], how='left')
incidents_df.head()

# %%
incidents_per_state = incidents_df[incidents_df['year']<=2020].groupby(['state', 'population_state_2010']).size()
incidents_per_state = ((incidents_per_state / incidents_per_state.index.get_level_values('population_state_2010'))*100000).to_frame(name='incidents_per_100k_inhabitants').sort_values(by='incidents_per_100k_inhabitants', ascending=True)
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
incidents_df[incidents_df['state']=='DISTRICT OF COLUMBIA'].groupby(['latitude', 'longitude', 'date']).size()[lambda x: x > 1].sort_values(ascending=False)

# %%
incidents_df.groupby(['latitude', 'longitude', 'date']).size()[lambda x: x>1]

# %%
incidents_per_month_per_state = incidents_df.groupby(['state', 'month_name', 'year']).size()
incidents_per_month_per_state = incidents_per_month_per_state.to_frame(name='incidents').reset_index()
incidents_per_month_per_state = incidents_per_month_per_state.sort_values(by=['year', 'month_name', 'state'], ignore_index=True)
incidents_per_month_per_state['incidents_per_100k_inhabitants'] = incidents_per_month_per_state.apply(
    lambda row: (row['incidents'] / usa_population_df[usa_population_df['state']==row['state']]['population_state_2010'].iloc[0])*100000,
    axis=1
)
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(
    incidents_per_month_per_state[incidents_per_month_per_state.year<=2020].pivot(
        index='state',
        columns=['year', 'month_name'],
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

plt.xticks(rotation=90)
plt.tight_layout()

# %%
incidents_df[incidents_df['state']=='DISTRICT OF COLUMBIA']['incident_characteristics1'].value_counts().plot(kind='barh', figsize=(20, 10))

# %%
incidents_per_month_per_state = incidents_df[incidents_df['incident_characteristics1']!='Non-Shooting Incident'].groupby(['state', 'month_name', 'year']).size()
incidents_per_month_per_state = incidents_per_month_per_state.to_frame(name='incidents').reset_index()
incidents_per_month_per_state = incidents_per_month_per_state.sort_values(by=['year', 'month_name', 'state'], ignore_index=True)
incidents_per_month_per_state['incidents_per_100k_inhabitants'] = incidents_per_month_per_state.apply(
    lambda row: (row['incidents'] / usa_population_df[usa_population_df['state']==row['state']]['population_state_2010'].iloc[0])*100000,
    axis=1
)
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(
    incidents_per_month_per_state[incidents_per_month_per_state.year<=2020].pivot(
        index='state',
        columns=['year', 'month_name'],
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
ax.set_title('Number of incidents per month per state (excluding non-shooting incidents)')

plt.xticks(rotation=90)
plt.tight_layout()

# %%
incidents_df[(incidents_df['state']=='DISTRICT OF COLUMBIA') & (incidents_df['year']==2014) & 
    (incidents_df['month']==1)]

# %%
incidents_df[(incidents_df['state']=='DISTRICT OF COLUMBIA') & (incidents_df['year']==2014) & 
    (incidents_df['month']==1)]['incident_characteristics1'].value_counts().plot(kind='barh', figsize=(20, 10))

# %%
incidents_df[(incidents_df['state']=='DISTRICT OF COLUMBIA')& (incidents_df['date']=="2014-01-01")]

# %% [markdown]
# https://mpdc.dc.gov/sites/default/files/dc/sites/mpdc/publication/attachments/MPD%20Annual%20Report%202017_lowres.pdf

# %%
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(
    incidents_per_month_per_state[(incidents_per_month_per_state.year<=2020) & (incidents_per_month_per_state['state']!='DISTRICT OF COLUMBIA')].pivot(
        index='state',
        columns=['year', 'month_name'],
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


plt.xticks(rotation=90)
plt.tight_layout()

# %%
# # merge data about the winning party # TODO: prendere dati giusti...
# winning_party_per_state_copy = winning_party_per_state.copy()
# winning_party_per_state_copy['year'] = winning_party_per_state['year'] + 1
# winning_party_per_state = pd.concat([winning_party_per_state, winning_party_per_state_copy], ignore_index=True)
# incidents_df = incidents_df[incidents_df['year'].notna()].merge(winning_party_per_state[['state', 'year', 'majority_state_party']], on=['state', 'year'], how='left')

# %%
# incidents_per_state_2016 = incidents_df[(incidents_df['n_killed']>0)].groupby(['state', 'year', 'population_state_2010', 'povertyPercentage', 'majority_state_party']).size()
# incidents_per_state_2016 = incidents_per_state_2016.to_frame(name='incidents').reset_index()
# incidents_per_state_2016['incidents_per_100k_inhabitants'] = (incidents_per_state_2016['incidents'] / incidents_per_state_2016['population_state_2010'])*100000
# fig = px.scatter(
#     incidents_per_state_2016,
#     x='povertyPercentage',
#     y='incidents_per_100k_inhabitants',
#     color='majority_state_party',
#     hover_name='state',
#     hover_data={'povertyPercentage': True, 'incidents_per_100k_inhabitants': True},
#     title='Mortal gun incidents in the USA',
#     facet_col="year",
#     facet_col_wrap=3
# )
# pyo.plot(fig, filename='../html/scatter_poverty.html', auto_open=False)
# fig.show()

# %% [markdown]
# We re-order the columns and we save the cleaned dataset:

# %%
time_columns = ['date', 'date_original', 'year', 'month', 'day', 'day_of_week']

geo_columns = ['state', 'address', 'latitude', 'longitude',
               'county', 'city', 'location_importance', 'address_type',
               'congressional_district', 'state_house_district', 'state_senate_district',
               'px_code']

participants_columns = ['participant_age1', 'participant1_child',
       'participant1_teen', 'participant1_adult', 'participant1_male',
       'participant1_female', 'min_age_participants', 'avg_age_participants', 'max_age_participants',
       'n_participants_child', 'n_participants_teen', 'n_participants_adult',
       'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested',
       'n_unharmed', 'n_participants']

characteristic_columns = ['notes', 'incident_characteristics1', 'incident_characteristics2', 
    'firearm', 'air_gun', 'shots', 'aggression', 'suicide', 'injuries',
    'death', 'road', 'illegal_holding', 'house', 'school', 'children',
    'drugs', 'officers', 'organized', 'social_reasons', 'defensive',
    'workplace', 'abduction', 'unintentional'] #TODO: add tag_consistency

external_columns = ['povertyPercentage', 'party', 'candidatevotes', 'totalvotes', 'candidateperc', 'population_state_2010']
#TODO:  majority state party?

# %% [markdown]
# We re-order the columns and we save the cleaned dataset:

# %%
incidents_df = incidents_df[time_columns + geo_columns + participants_columns + characteristic_columns + external_columns]
incidents_df = incidents_df.rename(
    columns={
        'povertyPercentage': 'poverty_perc',
        'candidatevotes': 'candidate_votes',
        'totalvotes': 'total_votes',
        'candidateperc': 'candidate_perc'
    }
)

# %%
incidents_df.shape[0]

# %%
numerical_columns = incidents_df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(15, 12))
corr_matrix = incidents_df[numerical_columns].corr()
sns.heatmap(corr_matrix, mask=np.triu(corr_matrix))

# %%
incidents_df.to_csv('../data/incidents_cleaned.csv', index=False)
