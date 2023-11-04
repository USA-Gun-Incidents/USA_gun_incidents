# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))

DATA_FOLDER_PATH = 'data/'

# %%
geopy_path = os.path.join(DATA_FOLDER_PATH, 'geopy/geopy.csv')
geopy_df = pd.read_csv(geopy_path, index_col=['index'], low_memory=False, dtype={})
geopy_df.head(n=2)

# %%
counties_path = os.path.join(DATA_FOLDER_PATH, 'wikipedia/counties.csv')
counties_df = pd.read_csv(counties_path)
counties_df.head()

# %%
geopy_df['state'].unique()

# %%
for c in counties_df['County or equivalent'].unique():
    print("'"+c+"'")

# %%
counties_df['State or equivalent'].unique()

# %%
state_map = { # wikipedia: geopy
    'Alabama': 'Alabama',
    'Alaska': 'Alaska',
    'American Samoa': np.nan, # diverso 
    'Arizona': 'Arizona',
    'Arkansas': 'Arkansas',
    'California': 'California',
    'Colorado': 'Colorado',
    'Connecticut': 'Connecticut',
    'Delaware': 'Delaware',
    'District of Columbia': 'District of Columbia',
    'Florida': 'Florida', 
    'Georgia': 'Georgia',
    'Guam': np.nan, # diverso
    'Hawaiʻi': 'Hawaii', #diverso
    'Idaho': 'Idaho',
    'Illinois': 'Illinois',
    'Indiana':'Indiana',
    'Iowa': 'Iowa', 
    'Kansas': 'Kansas', 
    'Kentucky': 'Kentucky',
    'Louisiana': 'Louisiana', 
    'Maine': 'Maine',
    'Maryland': 'Maryland', 
    'Massachusetts': 'Massachusetts',
    'Michigan': 'Michigan',
    'Minnesota': 'Minnesota', 
    'Mississippi': 'Mississippi', 
    'Missouri': 'Missouri',
    'Montana': 'Montana', 
    'Nebraska': 'Nebraska',
    'Nevada': 'Nevada',
    'New Hampshire': 'New Hampshire', 
    'New Jersey': 'New Jersey', 
    'New Mexico': 'New Mexico', 
    'New York': 'New York',
    'North Carolina': 'North Carolina',
    'North Dakota': 'North Dakota',
    'Northern Mariana Islands': np.nan, # diverso
    'Ohio': 'Ohio',
    'Oklahoma': 'Oklahoma', 
    'Oregon': 'Oregon', 
    'Pennsylvania': 'Pennsylvania',
    'Puerto Rico': np.nan, # diverso
    'Rhode Island': 'Rhode Island',
    'South Carolina': 'South Carolina', 
    'South Dakota': 'South Dakota', 
    'Tennessee': 'Tennessee',
    'Texas': 'Texas', 
    'U.S. Minor Outlying Islands': np.nan, # diverso
    'Utah': 'Utah',
    'Vermont': 'Vermont',
    'Virgin Islands (U.S.)': np.nan, # diverso 
    'Virginia': 'Virginia',
    'Washington': 'Washington',
    'West Virginia': 'West Virginia',
    'Wisconsin': 'Wisconsin', 
    'Wyoming': 'Wyoming'
}

# %%
# create a dictionary with counties_df['County or equivalent'].unique() as keys and geopy_df['county'].unique() or  geopy_df['city'].unique() as values
county_map = {}
for c in counties_df['County or equivalent'].unique():
    cont = True
    for gc in str(geopy_df['county'].unique()):
        if c in gc:
            county_map[c] = gc
            cont = False
            break
    if cont:
        for gc in str(geopy_df['city'].unique()):
            if c in gc:
                county_map[c] = gc
                cont = False
                break
    if cont:
        county_map[c] = np.nan

# %%
county_map

# %%
# save the dictionary in a json file
import json
with open('county_map.json', 'w') as fp:
    json.dump(county_map, fp)
    


# %%
incidents_path = DATA_FOLDER_PATH + 'incidents.csv'
incidents_df = pd.read_csv(incidents_path, low_memory=False)


# %%
final_path = DATA_FOLDER_PATH + 'checkpoints\checkpoint_geo.csv'
final_df = pd.read_csv(final_path, low_memory=False)

# %%
for i in final_df['county'].unique():
    if '(city)' in str(i):
        print(i)

# %%
final_df[final_df['county'] == 'Virginia Beach (city)']

# %%
incidents_df[incidents_df['city_or_county'] == 'Suffolk']

# %%
for i in geopy_df['county'].unique():
    if '(city)' in str(i):
        print(i)

# %%
final_df[final_df['city'] == 'New York']

# %%
for i in final_df['city'].unique(): # 
    if (#('City of' in str(i)) or 
        #('Town of' in str(i)) or 
        #('Village of' in str(i)) or 
        ('/' in str(i)) or 
        (',' in str(i))):
        print(i)

# %%
final_df[final_df['city'] == 'City of Rome']

# %%
geopy_df[geopy_df['city'] == 'Garden Lakes, City of Rome']

# %%
final_df[final_df['city'] == 'Red Rock']

# %%
# sostituire 'Village/Town' o 'Town/Village' con 'Town'
def susb(s): #TODO: lo volgiamo?
    if 'Village/Town' in s:
        return s.replace('Village/Town', 'Town')
    elif 'Town/Village' in s:
        return s.replace('Town/Village', 'Town')
    else:
        return s


# %% [markdown]
# 'Garden Lakes, City of Rome': 'City of Rome'
# 'Cairo, Georgia': 'Cairo'
# 
# Village/Town of Mount Kisco
# Town/Village of Harrison
# Town/Village of East Rochester
# 
# Westampton, New Jersey 
# Anthony, TX
# Village of Allegany / ohi꞉yoʾ 
# San Felipe Pueblo / Katishtya
# Redby / Madaabiimog
# Nambe Pueblo / Nambé Oweengé
# Santo Domingo Pueblo / Kewa
# New Odanah / Oshki-Oodena
# Lynchburg, Moore County
# Komatke / Komaḍk
# Edgefield City/Town Limits
# Oneida / ukwehuwé·ne
# Pala / Páala
# Anthony, NM
# Little Rock / Gaa-Asiniinsikaag
# Hockessin, Delaware
# City of Salamanca / Onë:dagö:h
# Ganienkeh / Kanièn:ke
# Town of Allegany / ohi꞉yoʾ
# San Ildefonso Pueblo / Pʼohwhogeh Ówîngeh
# Becenti / Tłʼóoʼditsin
# Red Lake / Ogaakaaning
# Chinle / Chʼínílį́
# Red Rock / Tsé Łichííʼ Dah Azkání

# %%
city_map = {
    'Garden Lakes, City of Rome': 'City of Rome',
    'Cairo, Georgia': 'Cairo'
    }

# %%
city_map['Garden Lakes, City of Rome']

# %%
def check(city):
    if city in city_map:
        return city_map[city]
    else:
        return city

# %%
check('Cairo, Georgia')

# %%
check('NY')

# %%
def c(city):
    print(city_map[city] if city in city_map.keys() else city)

c('Cairo, Georgia')
c('NY')


