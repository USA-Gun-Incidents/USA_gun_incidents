# %% [markdown]
# # Population data download and preparation
# 
# In this notebook we download U.S. population data from [US Census Bureau](https://www.census.gov) using the API to the American Community Survey (ACS) - 5 Year dataset.
# 
# More information about the data can be found at the following URL: [ACS 5 year data](https://www.census.gov/data/developers/data-sets/acs-5year.html).

# %%
# import libraries
import requests
import pandas as pd

# %% [markdown]
# In this dataset population is grouped in the following age categories:

# %%
ages = [
    'under 5 years',
    '5 to 9 years',
    '10 to 14 years',
    '15 to 17 years',
    '18 and 19 years',
    '20 years',
    '21 years',
    '22 to 24 years',
    '25 to 29 years',
    '30 to 34 years',
    '35 to 39 years',
    '40 to 44 years',
    '45 to 49 years',
    '50 to 54 years',
    '55 to 59 years',
    '60 and 61 years',
    '62 to 64 years',
    '65 and 66 years',
    '67 to 69 years',
    '70 to 74 years',
    '75 to 79 years',
    '80 to 84 years',
    '85 years and over'
]

# %% [markdown]
# The age group '10 to 14 years' is overlapped with the age groups 'Child 0-11' and 'Teen 12-17' provided in the gun incidents dataset. To use the same age groups, we will assume an even distribution of the population within this range across each individual year of age, assigning 2/5 of the population to the 'Child 0-11' group and 3/5 to the 'Teen 12-17' group.
# 
# In the code below we define the variables to query the API and download the data:

# %%
host = "https://api.census.gov/data"
dataset = "acs/acs5"

vars_to_retrieve = {}
for i, age in enumerate(ages):
    if i+3 < 10:
        males_suf = "00" + str(i+3) + "E"
    else:
        males_suf = "0" + str(i+3) + "E"
    females_suf = "0" + str(i+27) + "E"
    
    vars_to_retrieve['B01001_'+males_suf] = "Males " + age
    vars_to_retrieve['B01001_'+females_suf] = "Females " + age

predicates = {}
predicates["get"] = ",".join(vars_to_retrieve.keys())
predicates["for"] = "congressional district:*"

# %% [markdown]
# As an example, we now make a query for the year 2016:

# %%
base_url = "/".join([host, "2016", dataset])
req = requests.get(base_url, params=predicates)

population_df = pd.DataFrame(
    columns=req.json()[0],
    data=req.json()[1:]
)
population_df.columns = population_df.columns.map(lambda x: vars_to_retrieve[x] if x in vars_to_retrieve else x)
population_df.head()

# %%
population_df.info()

# %% [markdown]
# We cast the numerical columns:

# %%
columns_to_cast = [x for x in population_df.columns if x not in ["state", "congressional district"]]
population_df[columns_to_cast] = population_df[columns_to_cast].astype('UInt64')

# %% [markdown]
# We retrieve the state name from the code, using the official mapping provided by the Census Bureau:

# %%
usa_states = pd.read_csv(
    'https://www2.census.gov/geo/docs/reference/state.txt',
    sep='|',
    dtype={'STATE': str, 'STATE_NAME': str}
)
usa_num_name = usa_states.set_index('STATE').to_dict()['STATE_NAME']
population_df['state_name'] = population_df['state'].map(lambda x: usa_num_name[x])
population_df.head()

# %% [markdown]
# And now that we tested the API we can download the data for all the years:

# %%
years = ["20"+str(i) for i in range(13, 21)]
population_df = pd.DataFrame()

for year in years:
    base_url = "/".join([host, year, dataset])
    req = requests.get(base_url, params=predicates)
    
    population_year_df = pd.DataFrame(
        columns=req.json()[0],
        data=req.json()[1:]
    )
    population_year_df.columns = population_year_df.columns.map(lambda x: vars_to_retrieve[x] if x in vars_to_retrieve else x)
    columns_to_cast = [x for x in population_year_df.columns if x not in ["state", "congressional district"]]
    population_year_df[columns_to_cast] = population_year_df[columns_to_cast].astype('UInt64')
    
    population_year_df['year'] = year
    population_year_df['state_name'] = population_year_df['state'].map(lambda x: usa_num_name[x])
    population_df = pd.concat([population_df, population_year_df])

population_df.head()

# %% [markdown]
# We now group the population by age as stated above:

# %%
child_ages = [ages[i] for i in range(2)]
teen_ages = [ages[i] for i in range(3, 5)]
adult_ages = [ages[i] for i in range(5, len(ages))]

population_df['male_child'] = population_df[['Males '+age for age in child_ages]].sum(axis=1)
population_df['male_child'] += ((2/5)*population_df['Males '+ages[2]]).astype('UInt64')
population_df['male_teen'] = population_df[['Males '+age for age in teen_ages]].sum(axis=1)
population_df['male_teen'] += ((3/5)*population_df['Males '+ages[2]]).astype('UInt64')
population_df['male_adult'] = population_df[['Males '+age for age in adult_ages]].sum(axis=1)

population_df['female_child'] = population_df[['Females '+age for age in child_ages]].sum(axis=1)
population_df['female_child'] += ((2/5)*population_df['Females '+ages[2]]).astype('UInt64')
population_df['female_teen'] = population_df[['Females '+age for age in teen_ages]].sum(axis=1)
population_df['female_teen'] += ((3/5)*population_df['Females '+ages[2]]).astype('UInt64')
population_df['female_adult'] = population_df[['Females '+age for age in adult_ages]].sum(axis=1)

population_df.head()

# %% [markdown]
# We rename and reorder the columns and we sort the rows:

# %%
population_df.rename(
    columns={
        'state': 'state_code',
        'state_name': 'state',
        'congressional district': 'congressional_district'
    },
    inplace=True
)
cols = [
    'state_code',
    'state',
    'congressional_district',
    'year',
    'male_child',
    'male_teen',
    'male_adult',
    'female_child',
    'female_teen',
    'female_adult'
    ] + \
    ['Males '+age for age in ages] + \
    ['Females '+age for age in ages]
population_df = population_df[cols]
population_df.sort_values(
    by=['year', 'state_code', 'congressional_district'],
    inplace=True
)

# %%
population_df['congressional_district'].unique()

# %%
population_df[population_df['congressional_district']=='ZZ']

# %% [markdown]
# We drop obsolete congressional districts:

# %%
population_df = population_df[population_df['congressional_district'] != 'ZZ']

# %%
population_df[population_df['congressional_district']=='98']['state'].unique()

# %% [markdown]
# We set to 0 the congressional districts for District of Columbia to use the same notation as in the gun incidents dataset:

# %%
population_df.loc[population_df['state'] == 'District of Columbia', 'congressional_district'] = 0

# %% [markdown]
# We convert in uppercase the state names:

# %%
population_df['state'] = population_df['state'].str.upper()

# %% [markdown]
# We assess if there are any missing values:

# %%
population_df.info()

# %% [markdown]
# We save the data to a CSV file:

# %%
population_df.to_csv('./data/population.csv', index=False)


