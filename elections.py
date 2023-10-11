# -*- coding: utf-8 -*-
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
elections_data[elections_data['candidatevotes'] <= 0.5 * elections_data['totalvotes']]

# %% [markdown]
# # U.S. House special election, 2022: New York District 19
#
# Democratic hold

# %% [markdown]
# | Party             | Candidate       | Votes  | %    |
# | ----------------- | --------------- | ------ | ---- |
# | Democratic        | Pat Ryan        | 58,427 | 45.3 |
# | Working Families  | Pat Ryan        | 7,516  | 5.8  |
# | Total             | Pat Ryan        | 65,943 | 51.8 |
# | Republican        | Marc Molinaro   | 52,350 | 40.5 |
# | Conservative      | Marc Molinaro   | 10,602 | 8.2  |
# | Total             | Marc Molinaro   | 62,952 | 48.1 |
# | Write-in          |                 | 96     | 0.07 |
# | Total votes       |                 | 128,991| 100.0|

# %%
elections_data['perc_winner'] = elections_data['candidatevotes'] / elections_data['totalvotes']

# %%
# plot perc_winner
elections_data['perc_winner'].plot.hist(bins=100, figsize=(10, 5), title='Percentage of winner votes')
plt.show()

# %%
elections_data[elections_data['perc_winner']<0.5]['perc_winner'].plot.hist(bins=100, figsize=(10, 5), title='Percentage of winner votes < 50%')
plt.show()

# %%
elections_data.boxplot(column='totalvotes', by='state', figsize=(20, 10), rot=90)

# %%
elections_data[(elections_data['totalvotes']>2.51e6)]

# %%
# visualize the trend of totalvotes in Maine
elections_data[
    (elections_data['state']=='MAINE') & 
    (elections_data['congressional_district']==2) & 
    (elections_data['year']<2022)
].plot(x='year', y='totalvotes', figsize=(10, 5))

# %% [markdown]
# Maybe we should correct totalvotes according to the trend of totalvotes over the years

# %%
elections_data[(elections_data['totalvotes']<10) & (elections_data['year']>2012)]

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
# check if data is missing for some states
states = elections_data['state'].unique()
states = states[states!='DISTRICT OF COLUMBIA']
years = [i for i in range(elections_data['year'].min(), elections_data['year'].max(), 2)]
missing_data = False
for state, year in zip(states, years):
    if elections_data[(elections_data['state']==state) & (elections_data['year']==year)].size == 0:
        missing_data = True
        break
missing_data

# %%
# check if for every state and year congressional districts are sequential
sequential_districts = True
for state in states:
    for year in years:
        districts = elections_data[(elections_data['state']==state) & (elections_data['year']==year)]['congressional_district'].unique()
        districts.sort()
        # check if districts are sequential
        if districts.size > 1:
            if (districts != [i for i in range(1, districts.size+1)]).any():
                sequential_districts = False
                break
        elif districts[0] != 0:
            sequential_districts = False
            break
sequential_districts

# %%
num_dist_states = elections_data[(elections_data['year'].between(2013, 2018, inclusive="both"))].groupby(['year', 'state']).size().unstack()
num_dist_states

# %%
# in the period of interest the number of congressional districts is always the same, cannot use this feature to infer the missing year in incidents
distinct_counts = num_dist_states.nunique()
distinct_counts[distinct_counts != 1].index

# %%
year_state_votes = elections_data[elections_data['state']!='DISTRICT OF COLUMBIA'].groupby(['year', 'state'])['totalvotes'].mean().unstack()
for state in year_state_votes.columns:
    year_state_votes[state].plot.line(figsize=(15, 3), title=state)
    plt.show()

# %%
incidents_data.drop_duplicates(inplace=True)

# %%
incidents_data.info() # congressional_district is not always present

# %%
incidents_data.describe()

# %%
state_house_dist_values = incidents_data['state_house_district'].unique()
state_house_dist_values.sort()
state_house_dist_values

# %%
incidents_data[incidents_data['state_house_district']>=300]['state'].unique()

# %% [markdown]
# Wikipedia (after 2020, here we have data before): The state of New Hampshire has 228 legislative districts. The state Senate is made up of 24 senators elected from 24 districts. The House is composed of 400 members coming from 204 districts.

# %%
state_senate_dist_values = incidents_data['state_senate_district'].unique()
state_senate_dist_values.sort()
state_senate_dist_values

# %%
incidents_data[incidents_data['state_senate_district']==94]['state']

# %%
at_large_states = elections_data[
    (elections_data['year'].between(2013, 2018, inclusive="both")) & 
    (elections_data['congressional_district']==0)
    ]['state'].unique()
at_large_states

# %%
# capitalize incidents state
incidents_data['state'] = incidents_data['state'].str.upper()

# %% [markdown]
# From 2020 Illinois has 59 senate districts.

# %%
# check if states with a '0' as congressional district are the actual at large states
zero_congress_states_inc = incidents_data[incidents_data['congressional_district']==0]['state'].unique()
set(zero_congress_states_inc).issubset(set(at_large_states))

# %%
# check if states with a '1' as congressional district are not the actual at large states
one_congress_states_inc = incidents_data[incidents_data['congressional_district']==1]['state'].unique()
set(at_large_states).intersection(set(one_congress_states_inc))

# %%
# check if at large states are always numbered with 0
incidents_data[(incidents_data['state'] == at_large_states.any()) & (incidents_data['congressional_district']==1)].size==0

# %%
# set congressional district to 0 for at large states
incidents_data.loc[incidents_data['state'].isin(at_large_states), 'congressional_district'] = 0

# %%
# check if the same latitude and longitude are always associated with the same state
lat_long_state = incidents_data.groupby(['latitude', 'longitude'])['state'].unique()
lat_long_state[lat_long_state.apply(lambda x: len(x)>1)]

# %%
latitude = 38.8494
longitude = -76.9653
incidents_data[(incidents_data['latitude']==latitude) & (incidents_data['longitude']==longitude)]

# %%
# use geopy to get the correct state
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="DM_project")

# %%
location = geolocator.reverse(str(latitude) + ' ' + str(longitude)).raw
correct_state = location['address']['state'].upper()
correct_state

# %%
incidents_data.at[10232,'state'] = correct_state
incidents_data.at[110964,'state'] = correct_state

# %%
latitude = 38.9115
longitude = -76.9347
incidents_data[(incidents_data['latitude']==latitude) & (incidents_data['longitude']==longitude)]

# %%
location = geolocator.reverse(str(latitude) + ' ' + str(longitude)).raw
correct_state = location['address']['state'].upper()
correct_state

# %%
incidents_data.at[73841,'state'] = correct_state

# %%
latitude = 40.1053
longitude = -85.6803 
incidents_data[(incidents_data['latitude']==latitude) & (incidents_data['longitude']==longitude)]

# %%
location = geolocator.reverse(str(latitude) + ' ' + str(longitude)).raw
correct_state = location['address']['state'].upper()
correct_state

# %%
incidents_data.at[177763,'state'] = correct_state

# %%
# in a city or county there can be more than one congressional district
city_state_cong_dist = incidents_data.groupby(['state', 'city_or_county'])['congressional_district'].unique()
city_state_cong_dist

# %%
g = incidents_data.groupby(['latitude', 'longitude'])['city_or_county'].unique()
g[g.apply(lambda x: len(x)>1)]
# trim whitespaces... make more uniform

# %%
wrong_congr_states = elections_data.groupby('state')['congressional_district'].max()>=incidents_data.groupby('state')['congressional_district'].max()
wrong_congr_states[wrong_congr_states==False]

# %%
# TODO: città a statuto speciale, ricordarsi di non scartarla, gli incidenti possono essere interessanti
incidents_data[incidents_data['state']=='DISTRICT OF COLUMBIA']['congressional_district'].unique()

# %%
incidents_data.loc[incidents_data['state']=='DISTRICT OF COLUMBIA', 'congressional_district'] = 0

# %%
incidents_data[incidents_data['state']=='KENTUCKY']['congressional_district'].unique()

# %%
elections_data[(elections_data['state']=='KENTUCKY') & (elections_data['year']>2012)]['congressional_district'].unique()

# %%
incidents_data[(incidents_data['state']=='KENTUCKY') & (incidents_data['congressional_district']>6)]

# %%
location = geolocator.reverse(str(39.9186) + ' ' + str(-83.9392)).raw
location['address'] # it is in the 8th ohio district...

# %%
elections_data[(elections_data['state']=='OREGON') & (elections_data['year']>2012)]['congressional_district'].unique()

# %%
incidents_data[(incidents_data['state']=='OREGON') & (incidents_data['congressional_district']>6)]

# %%
location = geolocator.reverse(str(39.7573) + ' ' + str(-84.1818	)).raw
location['address']

# %%
# TODO: corregere indirizzo con dati sopra, è effettivamente il 10° distretto congressuale

# %%
location = geolocator.reverse(str(38.5856) + ' ' + str(-85.6436)).raw
location['address']

# %%
# TODO: corregere indirizzo con dati sopra, potrebbe essere il 9° distretto congressuale (in realtà sembra fuori dall'Indiana)

# %%
elections_data[(elections_data['state']=='WEST VIRGINIA') & (elections_data['year']>2012)]['congressional_district'].unique()

# %%
incidents_data[(incidents_data['state']=='WEST VIRGINIA') & (incidents_data['congressional_district']>3)]

# %%
location = geolocator.reverse(str(39.0006) + ' ' + str(-81.9786)).raw
location['address'] # it is in the 6th district of Ohio

# %%
# we should have
# latitute and longitude => congressional district
# latitute and longitude => state house district
# latitute and longitude => state senate district

# %%
g = incidents_data[incidents_data['congressional_district'].notnull()].groupby(['latitude', 'longitude'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]

# %% [markdown]
# Sembra che la prima entry sopra appartenga a due distretti diversi in quanto al confine tra i due...

# %%
incidents_data[(incidents_data['latitude']==25.7829) & (incidents_data['longitude']==-80.1312)]

# %% [markdown]
# ![image](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Florida_Congressional_Districts%2C_113th_Congress.tif/lossless-page1-1920px-Florida_Congressional_Districts%2C_113th_Congress.tif.png)

# %%
corrected_congr_districts = incidents_data[incidents_data['congressional_district'].notnull()].groupby(['latitude', 'longitude'])['congressional_district'].agg(lambda x: x.value_counts().index[0])

# %%
incidents_data = incidents_data.merge(corrected_congr_districts, on=['latitude', 'longitude'], how='left')
incidents_data['congressional_district_y'].fillna(incidents_data['congressional_district_x'], inplace=True)
incidents_data.rename(columns={'congressional_district_y':'congressional_district'}, inplace=True)
incidents_data.drop(columns=['congressional_district_x'], inplace=True)
incidents_data

# %%
incidents_data[incidents_data['congressional_district'].isnull()].shape[0] # 450 valori non più null

# %%
g = incidents_data[incidents_data['state_house_district'].notnull()].groupby(['latitude', 'longitude'])['state_house_district'].unique()
g[g.apply(lambda x: len(x)>1)]
# TODO: errors to correct

# %%
g = incidents_data[incidents_data['state_senate_district'].notnull()].groupby(['latitude', 'longitude'])['state_senate_district'].unique()
g[g.apply(lambda x: len(x)>1)]
# TODO: errors to correct

# %%
# we cannot conclude that 'state', 'city_or_county', 'state_senate_district' => 'congressional_district'
g = incidents_data[incidents_data['congressional_district'].notnull()].groupby(['state', 'city_or_county', 'state_senate_district'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]

# %%
# we cannot conclude that 'state', 'city_or_county', 'state_house_district' => 'congressional_district'
g = incidents_data[incidents_data['congressional_district'].notnull()].groupby(['state', 'city_or_county', 'state_house_district'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]

# %%
# we cannot conclude that 'state', 'city_or_county', 'state_house_district', 'state_senate_district' => 'congressional_district'
g = incidents_data[incidents_data['congressional_district'].notnull()].groupby(['state', 'city_or_county', 'state_house_district', 'state_senate_district'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]

# %%
# DAI DATI RISULTA CHE:
# lat, long, state house district => congressional district
# lat, long, state senate district => congressional district

# TODO: ridurre nan entries come fatto per congressional district e riapplicare

# UNA VOLTA SISTEMATI GLI INDIRIZZI VALUTARE:
# state, city, senate => cong
# state, city, house => cong

# %%
incidents_data[(incidents_data['congressional_district'].isna()) & (incidents_data['state_senate_district'].notna())]

# %%
incidents_data[(incidents_data['congressional_district'].isna()) & (incidents_data['state_house_district'].notna())]

# %%
g = incidents_data[incidents_data['congressional_district'].notnull()].groupby(['longitude', 'latitude', 'state_senate_district'])['congressional_district'].unique()
g[g.apply(lambda x: len(x)>1)]

# %% [markdown]
# # Info su distretti

# %% [markdown]
# Congressional districts are the boundaries of a district of the Representatives in the US Congress. A few states have only one congressman — ND, SD, WY, MT, DE, VT, and AK. Their congressional districts are the whole state. Every congressional district has one representative. There are 435 districts across the USA.
#
# State legislative districts are for the state representatives — they meet in the state capital city and decide state laws, state budget, etc. The number of state reps varies by state — NH has 400, for instance. Legislative districts are usually much smaller than congressional districts, because a state may have 5 congressmen (so congressional districts) but 80 legislative districts.
#
# I distretti per l'elezione dei senatori sono più ampi.
#
# Tante informazioni su tali ridifinizione sono disponibli su [Ballotopedia](https://ballotpedia.org/State_Legislative_Districts) (ma l'API costa da 500$ al mese).
#
# Le contee forse sono un analogo delle nostre provincie, sono fisse e hanno dei "nomi".
# Quando l'incidente non è avvenuto nel centro di una città, probabilmente viene indicata la contea.
#
# Il district of Columbia (dove c'è Washington DC (c'è anche lo stato del Washington ma non ci combina nulla)) forse funziona diversamente alle elezioni.
#
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


