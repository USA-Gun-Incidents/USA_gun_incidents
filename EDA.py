# %% [markdown]
# # Gun Incidents in the USA
# Dataset descriptions and explorative data analysis

# %%
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'
poverty_path = FOLDER + 'poverty_by_state_year.csv'
congressional_winner_path = FOLDER + 'year_state_district_house.csv'

# %%
# Load data from csv files
incidents_data = pd.read_csv(incidents_path)
poverty_data = pd.read_csv(poverty_path)
congressional_winner_data = pd.read_csv(congressional_winner_path)

# %% [markdown]
# ## Incidents data
# Contains information about gun incidents in the USA.

# %%
incidents_data.head()

# %%
incidents_data.columns

# %%
print('Number of entries: ', incidents_data.shape[0])
print('Numebr of columns: ', incidents_data.shape[1])

# %%
# Check for missing values
incidents_data.isnull().sum()

# %%
# Check for duplicate entries
incidents_data.duplicated().sum()

# %%
# Visualize how many incidents happened in each state
incidents_data['state'].value_counts().plot(kind='bar', figsize=(15, 5))
plt.title('Number of incidents in each state')
plt.xlabel('State')
plt.ylabel('Number of incidents')

# %%
incidents_data.describe()

# %%
incidents_data.describe(include='all')

# %%
incidents_data.info()

# %% [markdown]
# ### Date

# %%
# convert date column to datetime type
incidents_data['date'] = pd.to_datetime(incidents_data['date'])

# %%
# plot range data
incidents_data['date'].hist(figsize=(15, 5), bins=100)
print('Range data: ', incidents_data['date'].min(), ' - ', incidents_data['date'].max())

# %% [markdown]
# Check on Congressional District type

# %%
# check if distrect float values are actually integers
incidents_data['congressional_district'].apply(lambda x: x.is_integer()).value_counts()

# %%
incidents_data['congressional_district'].isna().sum()

# %%
# convert congressional district to integer if not null
incidents_data['congressional_district'] = incidents_data['congressional_district'].apply(lambda x: int(x) if not pd.isnull(x) else x)

# %%
incidents_data['congressional_district'] = incidents_data['congressional_district'].astype('Int64')

# %% [markdown]
# State House District

# %%
display(incidents_data['state_house_district'].apply(lambda x: x.is_integer()).value_counts())
print(incidents_data['state_house_district'].isna().sum())

# %%
incidents_data['state_house_district'] = incidents_data['state_house_district'].astype('Int64')

# %%
incidents_data['state'].value_counts().values

# %%
# display map of USA with number of incidents in each state
import plotly.express as px
import numpy as np
"""fig = px.choropleth(locations=incidents_data['state'].value_counts().index,
                    locationmode="USA-states",
                    color=incidents_data['state'].value_counts().values,
                    scope="usa",
                    color_continuous_scale="Viridis",
                    title='Number of incidents in each state')"""
color_scale = [(0, 'orange'), (1,'orange')]
fig = px.scatter_mapbox(color=np.ones(incidents_data.shape[0]), 
                        lat=incidents_data['latitude'], 
                        lon=incidents_data['longitude'],
                        color_continuous_scale=color_scale,
                        zoom=3, 
                        height=800,
                        width=800)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %% [markdown]
# ### State Senate District

# %%
display(incidents_data['state_senate_district'].apply(lambda x: x.is_integer()).value_counts())
print(incidents_data['state_senate_district'].isna().sum())

# %%
incidents_data['state_senate_district'] = incidents_data['state_senate_district'].astype('Int64')

# %%
incidents_data.info()

# %% [markdown]
# ### Descrive district

# %%
incidents_data['congressional_district'].describe()

# %%
# check if congressional  district with same number are in the same state
incidents_data.groupby(['state', 'congressional_district']).size().sort_values(ascending=False)

# %%
# state house district == 901
incidents_data[incidents_data['state_house_district'] == 901].state

# %%
# show all the unique state_house_district number in state = New Hampshire
inc = incidents_data[incidents_data['state'] == 'New Hampshire']['state_house_district'].unique()
# sort inc


# %%
incidents_data.loc[126777]

# %%


# %% [markdown]
# ## Poverty data
# Contains information about the poverty percentage for each USA state and year.

# %%
poverty_data.head()

# %%
print('Number of entries: ', poverty_data.shape[0])
print('Numebr of columns: ', poverty_data.shape[1])

# %%
# Check for missing values
poverty_data.isnull().sum()

# %%
# Check for duplicate entries
poverty_data.duplicated().sum()

# %%
print('Range of years: ', poverty_data['year'].min(), poverty_data['year'].max())
print('Number of states: ', poverty_data['state'].nunique())

# %%
# Visualize how poverty percentage changed over the years
poverty_data.groupby('year')['povertyPercentage'].mean().plot(kind='line', figsize=(15, 5))
plt.title('Poverty percentage over the years')
plt.xlabel('Year')
plt.ylabel('Poverty percentage (%)')

# %%
# Visualize how poverty percentage changed in each state
poverty_data.groupby('state')['povertyPercentage'].mean().plot(kind='bar', figsize=(15, 5))
plt.title('Poverty rate in each state')
plt.xlabel('State')
plt.ylabel('Poverty percentage (%)')

# %% [markdown]
# ## Congressional winner data
# Contains information about the winner of the congressional elections in the USA, for each year, state and congressional district.

# %%
congressional_winner_data.head()

# %%
print('Number of entries: ', congressional_winner_data.shape[0])
print('Numebr of columns: ', congressional_winner_data.shape[1])

# %%
# Check for missing values
congressional_winner_data.isnull().sum()

# %%
# Check for duplicate entries
congressional_winner_data.duplicated().sum()

# %%
print('Range of years: ', congressional_winner_data['year'].min(), congressional_winner_data['year'].max())
print('Number of states: ', congressional_winner_data['state'].nunique())
print('Number of parties: ', congressional_winner_data['party'].nunique())
print('Parties: ', congressional_winner_data['party'].unique())

# %%
print('Total number of total votes won by each party')
congressional_winner_data.groupby('party')['totalvotes'].sum()

# %%
# Visualize how many votes each party won in 2022
congressional_winner_data[congressional_winner_data['year'] == 2022].groupby('party')['totalvotes'].sum().plot(kind='bar', figsize=(15, 5))
plt.title('Total number of votes won by each party in 2022')
plt.xlabel('Party')
plt.ylabel('Total votes')

# %%
# Visualize how many votes democrats won in 2022 in each state
congressional_winner_data[congressional_winner_data['year'] == 2022][congressional_winner_data['party'] == 'DEMOCRAT'].groupby('state')['totalvotes'].sum().plot(kind='bar', figsize=(15, 5))
plt.title('Total number of votes won by democrats in 2022 in each state')
plt.xlabel('State')
plt.ylabel('Total votes')

# %%
# Visualize how many votes republicans won in 2022 in each state
congressional_winner_data[congressional_winner_data['year'] == 2022][congressional_winner_data['party'] == 'REPUBLICAN'].groupby('state')['totalvotes'].sum().plot(kind='bar', figsize=(15, 5))
plt.title('Total number of votes won by republicans in 2022 in each state')
plt.xlabel('State')
plt.ylabel('Total votes')

# %%
congressional_winner_data.info()


