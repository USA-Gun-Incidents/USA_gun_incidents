# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
# 
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
# 
# # Elections data understanding and preparation
# 
# We import the libraries:

# %%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as pyo

# %% [markdown]
# We define constants and settings for the notebook:

# %%
%matplotlib inline

import sys, os
sys.path.append(os.path.abspath('..'))
from plot_utils import *

DATA_FOLDER_PATH = '../data/'

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

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
# 
# We display a concise summary of the DataFrame:

# %%
elections_df.info()

# %% [markdown]
# We notice that:
# - the inferred types are correct
# - there are no missing values (however, we should still assess whether there are any missing rows for specific years, states, or congressional districts)
# 
# We display descriptive statistics:

# %%
elections_df.describe(include='all')

# %% [markdown]
# We notice that:
# - year spans from 2004 to 2020
# - there are 6 unique parties
# - the minimum of candidatevotes and totalvotes are negative numbers, meaning that there are actually missing values
# 
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
dc_elections_df = pd.read_csv('../data/external_data/district_of_columbia_house.csv')
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
# 
# Now we compute, for each year and state, the party with the highest percentage of votes, so to have a better understanding of the political orientation of each state:

# %%
# FIX: data l'osservazione sopra questo dato e questo plot non hanno più significato
usa_states_df = pd.read_csv(
    'https://www2.census.gov/geo/docs/reference/state.txt',
    sep='|',
    dtype={'STATE': str, 'STATE_NAME': str}
)
usa_name_alphcode = usa_states_df.set_index('STATE_NAME').to_dict()['STUSAB']
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
# We write the cleaned dataset to a csv file:

# %%
elections_df.to_csv('../data/year_state_district_house_cleaned.csv', index=False)


