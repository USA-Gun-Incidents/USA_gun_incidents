import requests
import pandas as pd

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
child_ages = [ages[i] for i in range(2)]
teen_ages = [ages[i] for i in range(3, 5)]
adult_ages = [ages[i] for i in range(5, len(ages))]

usa_states = pd.read_csv(
    'https://www2.census.gov/geo/docs/reference/state.txt',
    sep='|',
    dtype={'STATE': str, 'STATE_NAME': str}
)
usa_num_name = usa_states.set_index('STATE').to_dict()['STATE_NAME']

vars_to_retrieve = {}
for i, age in enumerate(ages):
    if i+3 < 10:
        males_suf = "00" + str(i+3) + "E"
    else:
        males_suf = "0" + str(i+3) + "E"
    females_suf = "0" + str(i+27) + "E"
    
    vars_to_retrieve['B01001_'+males_suf] = "Males " + age
    vars_to_retrieve['B01001_'+females_suf] = "Females " + age

host = "https://api.census.gov/data"
dataset = "acs/acs5"
predicates = {}
predicates["get"] = ",".join(vars_to_retrieve.keys())
predicates["for"] = "congressional district:*"

years = ["20"+str(i) for i in range(13, 21)]
all_data = pd.DataFrame()

for year in years:
    base_url = "/".join([host, year, dataset])
    req = requests.get(base_url, params=predicates)
    
    data = pd.DataFrame(
        columns=req.json()[0],
        data=req.json()[1:]
    )
    data.columns = data.columns.map(lambda x: vars_to_retrieve[x] if x in vars_to_retrieve else x)
    columns_to_cast = [x for x in data.columns if x not in ["state", "congressional district"]]
    data[columns_to_cast] = data[columns_to_cast].astype('UInt64')
    
    data['male_child'] = data[['Males '+age for age in child_ages]].sum(axis=1)
    data['male_child'] += ((2/5)*data['Males '+ages[2]]).astype('UInt64')
    data['male_teen'] = data[['Males '+age for age in teen_ages]].sum(axis=1)
    data['male_teen'] += ((3/5)*data['Males '+ages[2]]).astype('UInt64')
    data['male_adult'] = data[['Males '+age for age in adult_ages]].sum(axis=1)
    
    data['female_child'] = data[['Females '+age for age in child_ages]].sum(axis=1)
    data['female_child'] += ((2/5)*data['Females '+ages[2]]).astype('UInt64')
    data['female_teen'] = data[['Females '+age for age in teen_ages]].sum(axis=1)
    data['female_teen'] += ((3/5)*data['Females '+ages[2]]).astype('UInt64')
    data['female_adult'] = data[['Females '+age for age in adult_ages]].sum(axis=1)
    
    data['year'] = year
    data['state_name'] = data['state'].map(lambda x: usa_num_name[x])
    data.drop(
        columns=["Males " + age for age in ages]+["Females " + age for age in ages],
        inplace=True
    )
    all_data = pd.concat([all_data, data])

all_data.rename(
    columns={
        'state': 'state_code',
        'state_name': 'state',
        'congressional district': 'congressional_district'
    },
    inplace=True
)
all_data = all_data[[
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
]]
all_data.sort_values(
    by=['year', 'state_code', 'congressional_district'],
    inplace=True
)
all_data.to_csv('./data/census_bureau.csv', index=False)