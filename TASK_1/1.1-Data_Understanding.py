# %% [markdown]
# # Task 1.1 Data Understanding

# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# %% [markdown]
# We define constants and settings for the notebook:

# %%
import os
import sys
sys.path.append(os.path.abspath('..')) # TODO: c'è un modo per farlo meglio?
from plot_utils import *

%matplotlib inline

DATA_FOLDER_PATH = '../data/'

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

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
# - `min_age_participants`, `avg_age_participants`, `max_age_participants`, `n_participants_child`, `n_participants_teen`, `n_participants_adult` are stored as `object` while should be `int64`
# - the presence of missing values within many attributes; the only attributes without missing values are the following: `date`, `state`, `city_or_county`, `n_killed`, `n_injured`, `n_participants`

# %% [markdown]
# We display descriptive statistics:

# %%
incidents_data.describe(include='all')

# %% [markdown]
# We notice that:
# - age min 0, max 311
# - ...

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
# We notice that the entry relative to 2010 is missing. Since the other entries are ordered by year, we correct this error setting the year of the row at index 571 to 2010.

# %%
poverty_data.at[571,'year']=2010

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
# TODO: commentare

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
# Missing values are probably due to the fact that District of Columbia is a non voting delegate district. We found in [Wikipedia](https://en.wikipedia.org/wiki/2020_United_States_House_of_Representatives_election_in_the_District_of_Columbia) that while the number of votes received by the winning party coincides, the number of totalvotes is different. TODO: capire perchè e leggere il csv con i dati di Wiki.

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
elections_data.at[10186, 'candidatevotes'] = 165136
elections_data.at[10186, 'totalvotes'] = 311278

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
# We plot again the distribution of `totalvotes` and `candidatevotes` after cleaning the data:

# %%
elections_data[
    (elections_data['year']>2012)
].boxplot(column='totalvotes', by='state', figsize=(20, 10), rot=90, xlabel='State', ylabel='Total votes')
plt.suptitle('Total votes from 2014')
plt.title('')
plt.tight_layout()

# %% [markdown]
# TODO: commentare

# %% [markdown]
# We get the unique names of the parties for the years of interest:

# %%
elections_data[
    (elections_data['year']>2012)
]['party'].unique()

# %% [markdown]
# The Democratic Farmer Labor is the affiliate of the Democratic Party in the U.S. state of Minnesota [[Wikipedia](https://en.wikipedia.org/wiki/Minnesota_Democratic–Farmer–Labor_Party)], hence we replace this party name with 'DEMOCRATIC' to ease later analysis.

# %%
elections_data['party'] = elections_data['party'].apply(
    lambda x: 'DEMOCRATIC' if x=='DEMOCRATIC-FARMER-LABOR' else x
)

# %% [markdown]
# TODO:
# plot popolazione totale dello stato dal 2010 al 2022
# plot di cose stacked: per ogni stato una colonna blu-rossa per 14, 16, 18? (aggregato per stato...)

# %% [markdown]
# We check if `candidatevotes` are always more than 50% of `totalvotes`:

# %%
elections_data[elections_data['candidatevotes'] <= 0.5 * elections_data['totalvotes']].size == elections_data.size


