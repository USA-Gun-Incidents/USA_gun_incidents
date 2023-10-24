# %% [markdown]
# # Task 1 Data Understanding and Preparation

# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.append(os.path.abspath('..')) # TODO: c'è un modo per farlo meglio?
from plot_utils import *
from sklearn.neighbors import KNeighborsClassifier
from geopy import distance as geopy_distance
import calendar
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from pyproj import Transformer

# %% [markdown]
# We define constants and settings for the notebook:

# %%
%matplotlib inline

DATA_FOLDER_PATH = '../data/'

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %% [markdown]
# ## Poverty Data

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
# ## Elections Data

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
# ## Incidents Data

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

# DATE
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

# %%
# TODO: copiare qui osservazioni sulla data

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

# %%
districts = incidents_data['congressional_district'].unique()
color_discrete_map={}
n_colors = len(px.colors.qualitative.Plotly)
for i, district in enumerate(districts):
    color_discrete_map[str(district)] = px.colors.qualitative.Plotly[i%n_colors]
color_discrete_map[str(np.nan)] = '#000000'

fig = px.scatter_mapbox(
        color=incidents_data['congressional_district'].astype(str),
        color_discrete_map=color_discrete_map,
        lat=incidents_data['latitude'], 
        lon=incidents_data['longitude'],
        zoom=2, 
        height=400,
        width=800,
        title="USA Congressional districts",
        text=incidents_data['congressional_district'].astype(str),
        category_orders={'color': sorted(incidents_data['congressional_district'].astype(str).unique())}
    )
fig.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0,"t":100,"l":0,"b":0},
    legend_title_text="Congressional district"
)
fig.show()

# %% [markdown]
# We cannot recover the missing values for the attribute `congressional_district` from the values of `state_house_district` either.
# 
# We can, instead, recover the missing values from the entries with "similar" `latitude` and `longitude`. We'll do this first for the state of Alabama, showing the results with some plots. Later we will do the same for all the other states.
# 
# As a first step, we plot on a map the incidents that happened in Alabama, coloring them according to the value of the attribute `congressional_district`:

# %%
plot_scattermap_plotly(
    incidents_data[incidents_data['state']=='ALABAMA'],
    attribute='congressional_district',
    width=500,
    height=600,
    zoom=5.5,
    title="Alabama incidents by Congressional Districts",
    legend_title="Congressional District"
)

# %% [markdown]
# Many points with missing values for the attribute `congressional_district` (those in light green) are very near to other points for which the congressional district is known. We could use KNN classifier to recover those values. To do so, we define a function to prepare the data for the classification task:

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
# By the way, missclassification can still occurr, depending on the position of the available examples w.r.t the position of the points to classify. Aware of this limitation, we proceed to apply this method to the other states:

# %%
for state in incidents_data['state'].unique():
    if state != "ALABAMA":
        print(f"{state} done.")
        X_train, X_test, y_train = build_X_y_for_district_inference(incidents_data[incidents_data['state']==state])
        if X_test.shape[0] == 0:
            continue
        knn_clf.fit(X_train, y_train)
        knn_pred = knn_clf.predict(X_test)
        incidents_data.loc[
            (incidents_data['state']==state) &
            (incidents_data['congressional_district'].isna()) &
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna()),
            'KNN_congressional_district'
        ] = knn_pred

# %% [markdown]
# We drop the original column with congressional districts and we replace it with the one with the one we just computed:

# %%
incidents_data.drop(columns=['congressional_district'], inplace=True)
incidents_data.rename(columns={'KNN_congressional_district':'congressional_district'}, inplace=True)

# %% [markdown]
# TAGS EXPLORATION:

# %%
fig = incidents_data['incident_characteristics1'].value_counts().sort_values().plot(kind='barh', figsize=(5, 15))
fig.set_xscale("log")
plt.title("Counts of 'incident_characteristics1'")
plt.xlabel('Count')
plt.ylabel('Incident characteristics')
plt.tight_layout()

# %%
fig = incidents_data['incident_characteristics2'].value_counts().sort_values().plot(kind='barh', figsize=(5, 18))
fig.set_xscale("log")
plt.title("Counts of 'incident_characteristics2'")
plt.xlabel('Count')
plt.ylabel('Incident characteristics')
plt.tight_layout()

# %%
characteristics_count_matrix = pd.crosstab(incidents_data['incident_characteristics1'], incidents_data['incident_characteristics2'])
fig, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(characteristics_count_matrix, cmap='coolwarm', ax=ax, xticklabels=True, yticklabels=True, linewidths=.5)
ax.set_xlabel('incident_characteristics2')
ax.set_ylabel('incident_characteristics1')  
ax.set_title('Counts of incident characteristics')
plt.tight_layout()

# %%
incidents_data[incidents_data['state']=='DISTRICT OF COLUMBIA'].size

# %% [markdown]
# We join the poverty data with the incidents data:

# %%
incidents_data['year'] = incidents_data['date'].dt.year
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
incidents_data[incidents_data['state']=='DISTRICT OF COLUMBIA'].groupby(['latitude', 'longitude']).size()[lambda x: x > 1]

# %%
incidents_data[(incidents_data['latitude']==38.8204) & (incidents_data['longitude']==-77.0076)]

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
plt.tight_layout()

# %%
incidents_per_state_2016 = incidents_data[incidents_data['year']==2016].groupby(['state', 'population', 'povertyPercentage', 'party']).size().to_frame(name='incidents').reset_index()
incidents_per_state_2016['incidents_per_100k_inhabitants'] = (incidents_per_state_2016['incidents'] / incidents_per_state_2016['population'])*100000
fig = px.scatter(
    incidents_per_state_2016,
    x='povertyPercentage',
    y='incidents_per_100k_inhabitants',
    color='party',
    hover_name='state',
    hover_data={'povertyPercentage': True, 'incidents_per_100k_inhabitants': True}
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
# TODO:
# importare qui i controlli su età e partecipanti
# importare qui considerazioni su tag per le caratteristiche
# fare plot sui dati puliti


