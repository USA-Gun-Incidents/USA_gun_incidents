# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
# 
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa
# 
# # Poverty data understanding and preparation
# 
# We import the libraries:

# %%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyo

# %% [markdown]
# We define constants and settings for the notebook:

# %%
DATA_FOLDER_PATH = '../data/'
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

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
# We display a concise summary of the DataFrame:

# %%
poverty_df.info()

# %% [markdown]
# We notice that:
# - the inferred types of the attributes are correct
# - the presence of missing values within the attribute `povertyPercentage`
# 
# We display descriptive statistics:

# %%
poverty_df.describe(include='all')

# %% [markdown]
# We notice that:
# - the data are provided also for the United States as a whole
# - `year` spans from 2004 to 2020
# 
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

# %% [markdown]
# We write the cleaned dataset to a csv file:

# %%
poverty_df.to_csv('../data/poverty_by_state_year_cleaned.csv', index=False)


