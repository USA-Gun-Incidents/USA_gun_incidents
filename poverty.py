# -*- coding: utf-8 -*-
# %%
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'
poverty_path = FOLDER + 'poverty_by_state_year.csv'

# %%
incidents_data = pd.read_csv(incidents_path)
poverty_data = pd.read_csv(poverty_path)

# %%
poverty_data.head()

# %%
poverty_data.info()

# %%
poverty_data['state'].unique()

# %%
print(f"Range of years: [{poverty_data['year'].min()}, {poverty_data['year'].max()}]")
print(f"Number of states: {poverty_data['state'].nunique()}")

# %%
# check if state and year uniquely identify a row
poverty_data.groupby(['state', 'year']).size().max()==1

# %%
poverty_state_year = poverty_data.groupby(['state', 'year']).size()
poverty_state_year[poverty_state_year>1]

# %%
poverty_data[(poverty_data['state']=='Wyoming')]

# %%
# being the other rows sorted, it is probably an error
poverty_data.at[571,'year']=2010

# %%
poverty_data[(poverty_data['state']=='Wyoming')]

# %%
# check if for each state we have the expected number of rows (being <state, year> a key, it means there are no missing rows)
(poverty_data.groupby('state').size()==(poverty_data['year'].max()-poverty_data['year'].min()+1)).all()

# %%
# check statistics
poverty_data.describe()

# %%
# check if there are duplicated rows
poverty_data.duplicated().sum()

# %%
# check if there are null values
poverty_data.isnull().sum()

# %%
# check years with null values
poverty_data[poverty_data['povertyPercentage'].isnull()]['year'].unique()

# %% [markdown]
# Since no incidents happened in 2012, for now we will not care about these missing values

# %%
poverty_states = poverty_data['state'].unique()
poverty_states

# %%
poverty_states.size-1 # excluding United States

# %%
incidents_data['date'] = pd.to_datetime(incidents_data['date'], format="%Y/%m/%d")
incidents_data['year'] = incidents_data['date'].dt.year

# %%
# join incidents and poverty data
incidents_poverty_data = incidents_data.merge(poverty_data, on=['state', 'year'], how='left')

# %%
incidents_poverty_data.head()

# %%
incidents_poverty_data = incidents_poverty_data.drop(columns=['year'])

# %%
# check if the joining operation was successful
incidents_poverty_data[
    (incidents_poverty_data['povertyPercentage'].isnull()) &
    (incidents_poverty_data['date'].dt.year<=poverty_data['year'].max())
    ].size==0

# %%
# plot bar plot of povertyPercentage for each state sorting by povertyPercentage
poverty_data.groupby(['state'])['povertyPercentage'].mean().sort_values().plot(kind='bar', figsize=(15, 5))

# %% [markdown]
# Group line plot according to the average over the years

# %%
mean_poverty_per_state = poverty_data.groupby(['state'])['povertyPercentage'].mean()
thresholds = [ # chosen by looking at the bar plot
    mean_poverty_per_state['Vermont'],
    mean_poverty_per_state['Alaska'],
    mean_poverty_per_state['South Dakota'],
    mean_poverty_per_state['North Carolina'],
    mean_poverty_per_state['Arkansas']
    ]

# %%
fig, axs = plt.subplots(1, 5, sharey=True, figsize=(30, 15))
fig.suptitle('Poverty percentage for each state over the years', fontsize=16)
prev_th = -1
for i in range(len(thresholds)):
    axs[i].set_xlabel('Year')
    axs[i].set_ylabel('Poverty percentage (%)')
    states = list(mean_poverty_per_state[mean_poverty_per_state.between(prev_th, thresholds[i])].index)
    if i==3:
        markers = ['o' if i%2==0 else 'x' for i in range(20)]
        axs[i].set_prop_cycle(color=plt.cm.tab20.colors, marker=markers)
    poverty_data[poverty_data['state'].isin(states)].groupby(['year', 'state'])['povertyPercentage'].mean().unstack().plot(kind='line', ax=axs[i])
    prev_th = thresholds[i]

# %%
import geopandas
geo_usa = geopandas.read_file("./cb_2018_us_state_500k")
poverty_data_2018 = poverty_data[poverty_data['year']==2018].copy()
poverty_data_2018.rename(columns={'state':'NAME'},inplace=True)
geo_merge=geo_usa.merge(poverty_data_2018,on='NAME')
geo_merge.plot(column='povertyPercentage', figsize=(20, 10),legend=True,cmap='coolwarm')
plt.xlim(-130,-60)
plt.ylim(20,55)
for idx, row in geo_merge.iterrows():
    plt.annotate(text=row['NAME'], xy=row['geometry'].centroid.coords[0], ha='center', fontsize=10)
plt.title('Poverty percentage 2018',fontsize=25)
plt.show()

# %%
# potrebbe comunque essere interessante interpolare i valori mancanti se vogliamo studiare come è variata la povertà
# perchò non abbiamo i dati del 2012? c'è un motivo? cercando in rete sembra di no, li hanno tolti a caso


