# %%
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %%
# downloaded from https://mail.statefirearmlaws.org/state-state-firearm-law-data
laws_df = pd.read_csv('../data/external_data/laws_by_state_year.csv', sep=';')
laws_df.head()

# %%
laws_df['state'] = laws_df['state'].str.upper()
laws_df.rename(columns={'lawtotal': 'number_of_laws'}, inplace=True)
laws_df[['state', 'year', 'number_of_laws']].info()

# %%
(laws_df['year'].max()-laws_df['year'].min()+1)*laws_df['state'].nunique()

# %%
laws_df = laws_df[(laws_df['year']>=2013) & (laws_df['year']<=2020)]#[['state', 'year', 'number_of_laws']]
laws_df

# %%
laws_df[laws_df['state']=='CALIFORNIA'][['mentalhealth']]

# %%
laws_df.sort_values(by=['number_of_laws', 'state', 'year'], ascending=[False, True, False])[['state', 'year', 'mentalhealth', 'number_of_laws']][:50]

# %%
dc_number_of_laws = 100
for year in range(2013, 2021):
    laws_df = pd.concat([laws_df, pd.DataFrame({'state': 'DISTRICT OF COLUMBIA', 'year': year, 'number_of_laws': dc_number_of_laws}, index=[0])], ignore_index=True)
laws_df

# %%
laws_df = laws_df.sort_values(by=['state', 'year', 'number_of_laws', ], ignore_index=True)[['state', 'year', 'number_of_laws']]
laws_df

# %%
laws_df.to_csv('../data/external_data/number_laws_by_state_year.csv', sep=';', index=False)

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=0,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

# %%
incidents_per_state = incidents_df[(incidents_df['year']<=2020) & (incidents_df['death'])].groupby(['state', 'year', 'population_state_2010']).size()
incidents_per_state = ((incidents_per_state / incidents_per_state.index.get_level_values('population_state_2010'))*100000).to_frame(name='num_incidents_per_100k_inhabitants')
incidents_per_state.reset_index(inplace=True)
incidents_per_state

# %%
# merge incidents_per_state with laws_df on state and year
incidents_laws_df = incidents_per_state.merge(laws_df, on=['state', 'year'], how='left')
incidents_laws_df

# %%
incidents_laws_df.info()

# %%
incidents_laws_df[incidents_laws_df['year']==2013][['num_incidents_per_100k_inhabitants', 'number_of_laws']].corr()

# %%
incidents_laws_df[incidents_laws_df['year']==2014][['num_incidents_per_100k_inhabitants', 'number_of_laws']].corr()

# %%
incidents_laws_df[incidents_laws_df['year']==2015][['num_incidents_per_100k_inhabitants', 'number_of_laws']].corr()

# %%
incidents_laws_df[incidents_laws_df['year']==2016][['num_incidents_per_100k_inhabitants', 'number_of_laws']].corr()

# %%
incidents_laws_df[incidents_laws_df['year']==2017][['num_incidents_per_100k_inhabitants', 'number_of_laws']].corr()

# %%
incidents_laws_df[incidents_laws_df['year']==2018][['num_incidents_per_100k_inhabitants', 'number_of_laws']].corr()

# %%
fig = px.scatter(
    incidents_laws_df,
    x='num_incidents_per_100k_inhabitants',
    y='number_of_laws',
    hover_name='state',
    hover_data={'num_incidents_per_100k_inhabitants': True, 'number_of_laws': True},
    title='Mortal gun incidents in the USA',
    facet_col="year",
    facet_col_wrap=3,
    height=800
)
pyo.plot(fig, filename='../html/scatter_poverty.html', auto_open=False)
fig.show()

# %% [markdown]
# # Gun Laws

# %% [markdown]
# Every year, the [Giffords Law Center to Prevent Gun Violence](https://giffords.org/lawcenter/resources/scorecard/) assesses the gun regulations of all 50 states in the United States, assigning each state a letter grade reflective of the strength of its gun laws. The grades are determined by the efficacy and robustness of the state's firearm regulations, with the center asserting that states with stringent gun laws have fewer gun deaths.
# 
# We manually collected the data from the 2013 pdf report in the file 'giffords_scorecard_2013.csv'. In the following we will download the data from 2014 to 2018 (the years in which the incident data is available) and merge it into one dataframe.
# 
# Giffords Law Center to Prevent Gun Violence does not grade the District of Columbia. Based on a research of the gun laws in the District of Columbia (see e.g. [here](https://giffords.org/lawcenter/gun-laws/washington-dc/)) we will give it an A.
# 
# We will map the letter grades to numerical values representing the rank of the state's gun laws. We will use the following mapping:

# %%
replace_map = {
    'A+': 1,
    'A': 2,
    'A-' : 3,
    'B+' : 4,
    'B' : 5,
    'B-': 6,
    'C+': 7,
    'C': 8,
    'C-': 9,
    'D+': 10,
    'D': 11,
    'D-': 12,
    'F': 13
}

# %% [markdown]
# ## Gun Laws in 2013

# %%
gun_grades_2013 = pd.read_csv('../data/external_data/giffords_scorecard_2013.csv')
gun_grades_2013

# %%
gun_grades_2013 = gun_grades_2013.replace({"2013 Grade": replace_map})
gun_grades_2013 = gun_grades_2013[['State', '2013 Grade']]
gun_grades_2013['year'] = 2013
gun_grades_2013.rename(columns={'State': 'state', '2013 Grade': 'gun_law_rank'}, inplace=True)
gun_grades_2013 = pd.concat([gun_grades_2013, pd.DataFrame({'state': 'District of Columbia', 'gun_law_rank': 2, 'year': 2013}, index=[0])], ignore_index=True)
gun_grades_2013

# %% [markdown]
# ## Gun Laws in 2014

# %%
gun_grades_2014 = pd.read_html('https://giffords.org/scorecard2014/')
gun_grades_2014

# %%
gun_grades_2014 = gun_grades_2014[0]
gun_grades_2014

# %%
gun_grades_2014 = gun_grades_2014.replace({"2014 Grade": replace_map})
gun_grades_2014 = gun_grades_2014[['State', '2014 Grade']]
gun_grades_2014['year'] = 2014
gun_grades_2014.rename(columns={'State': 'state', '2014 Grade': 'gun_law_rank'}, inplace=True)
gun_grades_2014 = pd.concat([gun_grades_2014, pd.DataFrame({'state': 'District of Columbia', 'gun_law_rank': 2, 'year': 2014}, index=[0])], ignore_index=True)
gun_grades_2014

# %% [markdown]
# ## Gun Laws in 2015

# %%
gun_grades_2015 = pd.read_html('https://giffords.org/scorecard2015/')
gun_grades_2015

# %%
gun_grades_2015 = gun_grades_2015[0]
gun_grades_2015

# %%
gun_grades_2015 = gun_grades_2015.replace({"2015 Grade": replace_map})
gun_grades_2015 = gun_grades_2015[['State', '2015 Grade']]
gun_grades_2015['year'] = 2015
gun_grades_2015.rename(columns={'State': 'state', '2015 Grade': 'gun_law_rank'}, inplace=True)
gun_grades_2015 = pd.concat([gun_grades_2015, pd.DataFrame({'state': 'District of Columbia', 'gun_law_rank': 2, 'year': 2015}, index=[0])], ignore_index=True)
gun_grades_2015

# %% [markdown]
# ## Gun Laws in 2016

# %%
gun_grades_2016 = pd.read_html('https://giffords.org/scorecard2016/')
gun_grades_2016

# %%
gun_grades_2016 = gun_grades_2016[0]
gun_grades_2016

# %%
gun_grades_2016 = gun_grades_2016.replace({"2016 Grade": replace_map})
gun_grades_2016 = gun_grades_2016[['State', '2016 Grade']]
gun_grades_2016['year'] = 2016
gun_grades_2016.rename(columns={'State': 'state', '2016 Grade': 'gun_law_rank'}, inplace=True)
gun_grades_2016 = pd.concat([gun_grades_2016, pd.DataFrame({'state': 'District of Columbia', 'gun_law_rank': 2, 'year': 2016}, index=[0])], ignore_index=True)
gun_grades_2016

# %% [markdown]
# ## Gun Laws in 2017

# %%
gun_grades_2017 = pd.read_html('https://giffords.org/lawcenter/resources/scorecard2017/')
gun_grades_2017

# %%
gun_grades_2017 = gun_grades_2017[0]
gun_grades_2017

# %%
gun_grades_2017 = gun_grades_2017.replace({"2017 Grade": replace_map})
gun_grades_2017 = gun_grades_2017[['State', '2017 Grade']]
gun_grades_2017['year'] = 2017
gun_grades_2017.rename(columns={'State': 'state', '2017 Grade': 'gun_law_rank'}, inplace=True)
gun_grades_2017 = pd.concat([gun_grades_2017, pd.DataFrame({'state': 'District of Columbia', 'gun_law_rank': 2, 'year': 2017}, index=[0])], ignore_index=True)
gun_grades_2017

# %% [markdown]
# ## Gun Laws in 2018

# %%
gun_grades_2018 = pd.read_html('https://giffords.org/lawcenter/resources/scorecard2018/')
gun_grades_2018

# %%
gun_grades_2018 = gun_grades_2018[0]
gun_grades_2018

# %%
gun_grades_2018 = gun_grades_2018.replace({"2018 Grade": replace_map})
gun_grades_2018 = gun_grades_2018[['State', '2018 Grade']]
gun_grades_2018['year'] = 2018
gun_grades_2018.rename(columns={'State': 'state', '2018 Grade': 'gun_law_rank'}, inplace=True)
gun_grades_2018 = pd.concat([gun_grades_2018, pd.DataFrame({'state': 'District of Columbia', 'gun_law_rank': 2, 'year': 2018}, index=[0])], ignore_index=True)
gun_grades_2018

# %%
gun_grades = pd.concat([
    gun_grades_2013,
    gun_grades_2014,
    gun_grades_2015,
    gun_grades_2016,
    gun_grades_2017,
    gun_grades_2018
])
gun_grades

# %%
gun_grades['state'] = gun_grades['state'].str.upper()
gun_grades.to_csv('../data/external_data/gun_law_rank.csv', index=False)

# %% [markdown]
# ## Correlation between Gun Laws and Gun Deaths

# %% [markdown]
# We load the incident data:

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=0,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

# %% [markdown]
# We compute the number of mortal incidents per state:

# %%
incidents_per_state = incidents_df[(incidents_df['year']<=2020) & (incidents_df['death'])].groupby(['state', 'year', 'population_state_2010']).size()
incidents_per_state = ((incidents_per_state / incidents_per_state.index.get_level_values('population_state_2010'))*100000).to_frame(name='num_incidents_per_100k_inhabitants')
incidents_per_state.reset_index(inplace=True)
incidents_per_state

# %% [markdown]
# We merge the dataframes:

# %%
incidents_laws_df = incidents_per_state.merge(gun_grades, on=['state', 'year'], how='left')
incidents_laws_df

# %% [markdown]
# We plot the number of mortal incidents per state against the rank of the state's gun laws:

# %%
fig = px.scatter(
    incidents_laws_df,
    x='num_incidents_per_100k_inhabitants',
    y='gun_law_rank',
    hover_name='state',
    hover_data={'num_incidents_per_100k_inhabitants': True, 'gun_law_rank': True},
    title='Mortal gun incidents in the USA',
    facet_col="year",
    facet_col_wrap=3,
    height=800
)
fig.show()

# %% [markdown]
# No clear correlation can be seen.
# 
# We also compute the pearson correlation coefficient for each year:

# %%
for year in [2013, 2014, 2015, 2016, 2017, 2018]:
    print(year, incidents_laws_df[incidents_laws_df['year']==year][['num_incidents_per_100k_inhabitants', 'gun_law_rank']].corr().iloc[0][1])

# %% [markdown]
# Correlations are not significant.


