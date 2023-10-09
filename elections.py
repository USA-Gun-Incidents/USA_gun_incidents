# %%
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'
elections_path = FOLDER + 'year_state_district_house.csv'

# %%
incidents_data = pd.read_csv(incidents_path)
elections_data = pd.read_csv(elections_path)

# %%
elections_data.info()

# %%
elections_data.describe()

# %%
# the triple (year, state and congressional district) uniquely identifies rows
elections_data.groupby(['year', 'state', 'congressional_district']).size().max() == 1

# %%
# candidatevotes are always less or equal than totalvotes
elections_data[elections_data['candidatevotes'] <= elections_data['totalvotes']].size == elections_data.size

# %%
# candidate votes are not always more than 50% of total votes...
elections_data[(elections_data['candidatevotes'] <= 0.5 * elections_data['totalvotes']) & (elections_data['year']>2012)]

# %%
elections_data[elections_data['candidatevotes'] == 0.5 * elections_data['totalvotes']]

# %%
elections_data['perc_winner'] = elections_data['candidatevotes'] / elections_data['totalvotes']

# %%
# plot perc_winner
elections_data['perc_winner'].plot.hist(bins=100, figsize=(10, 5), title='Percentage of winner votes')
plt.show()

# %%
elections_data[elections_data['perc_winner']<0.5]['perc_winner'].plot.hist(bins=100, figsize=(10, 5), title='Percentage of winner votes < 50%')
plt.show()

# %% [markdown]
# Maybe we should correct totalvotes according to the trend of totalvotes over the years

# %%
elections_data[elections_data['state']=='ALABAMA'].groupby(['year', 'state', 'congressional_district'])['totalvotes'].mean().unstack().plot(kind='line', figsize=(15, 5))
plt.show()

# %%
elections_data[elections_data['totalvotes']<10]

# %%
elections_data[elections_data['totalvotes']>600000]

# %%
elections_data['party'].unique()

# %%
elections_data[elections_data['party']=='FOGLIETTA (DEMOCRAT)']

# %%
elections_data[(elections_data['state']=='PENNSYLVANIA') & (elections_data['year']==1980)]

# %%
# replace 'FOGLIETTA (DEMOCRAT)' with 'DEMOCRAT'
elections_data['party'] = elections_data['party'].replace('FOGLIETTA (DEMOCRAT)', 'DEMOCRAT') # TODO: or independent?
# WIKIPEDIA: In the 1980 elections, Foglietta won in Pennsylvania's 1st Congressional District, running as an independent. 

# %%
elections_data[elections_data['party']=='INDEPENDENT']

# %%
elections_data[elections_data['party']=='INDEPENDENT-REPUBLICAN']

# %%
elections_data[elections_data['party']=='DEMOCRATIC-FARMER-LABOR']

# %%
elections_data[(elections_data['state']=='MINNESOTA') & (elections_data['year']>2012)]

# %% [markdown]
# The Minnesota Democratic–Farmer–Labor Party (DFL) is the affiliate of the Democratic Party in the U.S. state of Minnesota.

# %%
# replace DEMOCRATIC-FARMER-LABOR with DEMOCRAT
elections_data['party'] = elections_data['party'].replace('DEMOCRATIC-FARMER-LABOR', 'DEMOCRAT')

# %%
elections_data['year'].max()-elections_data['year'].min()+1

# %%
elections_data['state'].unique()

# %%
elections_data['state'].unique().size

# %%
elections_data.groupby('state').size().sort_values(ascending=False)

# %%
elections_data[elections_data['state']=='DISTRICT OF COLUMBIA']

# %%
states = elections_data['state'].unique()

# %%
states = states[states!='DISTRICT OF COLUMBIA']
states

# %%
years = [i for i in range(elections_data['year'].min(), elections_data['year'].max(), 2)]
years

# %%
for state, year in zip(states, years):
    if elections_data[(elections_data['state']==state) & (elections_data['year']==year)].size == 0:
        raise Exception('No data for state {} and year {}'.format(state, year))

# %%
# il numero di congressional distretti cambia, da wiki sembra 2003, 2013, 2023, qua anche 2012!!!!
elections_data[(elections_data['year']>=2010) & (elections_data['year']<=2020)].groupby(['year', 'state']).size().unstack()

# %%
# TODO: controlla 0-1, sequenzialità

# %%
year_state_votes = elections_data[elections_data['state']!='DISTRICT OF COLUMBIA'].groupby(['year', 'state'])['totalvotes'].mean().unstack()
for state in year_state_votes.columns:
    year_state_votes[state].plot.line(figsize=(15, 3), title=state)
    plt.show()

# %%
incidents_data.drop_duplicates(inplace=True)

# %%
incidents_data.info()
# ci mancano dei congressional districts

# %%
incidents_data.describe()

# %%
state_house_dist_values = incidents_data['state_house_district'].unique()
state_house_dist_values.sort()
state_house_dist_values

# %%
incidents_data[incidents_data['state_house_district']>=303]['state'].unique()

# %%
state_senate_dist_values = incidents_data['state_senate_district'].unique()
state_senate_dist_values.sort()
state_senate_dist_values

# %%
incidents_data[incidents_data['state_house_district']==1]['state'].unique()

# %%
incidents_data[incidents_data['state_senate_district']==1]['state'].unique()

# %%
# in a city or county there can be more than one congressional district
city_state_cong_dist = incidents_data.groupby(['state', 'city_or_county'])['congressional_district'].unique()
city_state_cong_dist

# %%
g = incidents_data.groupby(['latitude', 'longitude'])['state'].unique()
g[g.apply(lambda x: len(x)>1)]
# TODO: error to correct

# %%
g = incidents_data.groupby(['latitude', 'longitude'])['city_or_county'].unique()
g[g.apply(lambda x: len(x)>1)]
# trim whitespaces... make more uniform

# %%
g = incidents_data[incidents_data['congressional_district'].notnull()].groupby(['latitude', 'longitude'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]
# TODO: all errors to correct

# %%
g = incidents_data[incidents_data['state_house_district'].notnull()].groupby(['latitude', 'longitude'])['state_house_district'].unique()
g[g.apply(lambda x: len(x)>1)]
# TODO: errors to correct

# %%
g = incidents_data[incidents_data['state_senate_district'].notnull()].groupby(['latitude', 'longitude'])['state_senate_district'].unique()
g[g.apply(lambda x: len(x)>1)]
# TODO: errors to correct

# %%
g = incidents_data[incidents_data['congressional_district'].notnull()].groupby(['state', 'city_or_county', 'address', 'state_house_district', 'state_senate_district'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]
# TODO: errors to correct

# %%
g = incidents_data.groupby(['state', 'city_or_county', 'state_house_district'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]

# %%
g = incidents_data.groupby(['state', 'city_or_county', 'state_senate_district'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]

# %%
g = incidents_data.groupby(['state', 'city_or_county', 'state_house_district', 'state_senate_district'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]

# %% [markdown]
# # Info su distretti

# %% [markdown]
# Ciascun distretto congressuale è a sua volta diviso in distretti per la camera (House) e il senato.
# 
# I distretti congressuali vengono ridefiniti ogni 10 anni in base alla popolazione (su Wikipedia cifra unità = 3).
# 
# I distretti per la camera e il senato dovrebbero (?) essere ridefiniti ogni 10 anni in base alla popolazione (cifra unità = 0).
# 
# Tante informazioni su tali ridifinizione sono disponibli su [Ballotopedia](https://ballotpedia.org/State_Legislative_Districts) (ma l'API costa da 500$ al mese).
# 
# A noi servirebbe solo il distretto congressuale per poter fare il merge con i dati delle elezioni, ma alcuni valori mancano.
# Se assumessimo che latitudine e longitudine siano giuste, potremmo ricavare il distretto congressuale corretto usando librerie esterne ().
# Forse però la prof si aspetta da noi che inferiamo le cose dai dati che abbiamo.
# Probabilmente ci sono stati dati anche i distretti per la camera e il senato, proprio per questo.
# 
# Le contee forse sono un analogo delle nostre provincie, sono fisse e hanno dei "nomi".
# Quando l'incidente non è avvenuto nel centro di una città, probabilmente viene indicata la contea.
# 
# Il district of Columbia (dove c'è Washington DC (c'è anche lo stato del Washington ma non ci combina nulla)) forse funziona diversamente alle elezioni.
# 

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# I distretti per l'elezione dei senatori sono più ampi.

# %% [markdown]
# [Sito](https://sunlightlabs.github.io/congress/districts.html) per capire distretti da latitudine-longitudine?
# 
# [Dati pubblici](https://github.com/orgs/opencivicdata/repositories)?
# 
# Forse anche lo stesso plotly ha su Github dei dataset utili (ho visto nella gallery delle mappe con i distretti).

# %% [markdown]
# ## USA counties

# %% [markdown]
# ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Usa_counties_large.svg/2560px-Usa_counties_large.svg.png)

# %% [markdown]
# Ho usato un tool online per convertire in csv una tabella di Wikipedia con le contee.

# %%
counties = pd.read_csv('./data/counties.csv')

# %%
counties.info()

# %%
counties.head()

# %%
# the tuple <County or equivalent', 'State or equivalent'> uniquely identifies rows
counties.groupby(['County or equivalent', 'State or equivalent']).size().max()==1

# %% [markdown]
# La chiave primaria non è 'County or equivalent' perchè, ad esempio, la contea di Washington c'è in tanti stati diversi.


