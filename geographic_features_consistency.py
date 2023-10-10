# -*- coding: utf-8 -*-
# %% [markdown]
# # Geographic features

# %% [markdown]
# ## Import data

# %%
import pandas as pd

# %%
# read data
FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'

incidents_data = pd.read_csv(incidents_path)

# %%
# drop duplicates rows
incidents_data.drop_duplicates(inplace=True)

# %%
# select only relevant columns from incidents_data
geo_data = incidents_data[['date', 'state', 'city_or_county', 'address', 'latitude', 'longitude',
       'congressional_district', 'state_house_district', 'state_senate_district']]

# %%
geo_data

# %% [markdown]
# ## GeoPy Data Description

# %%
# geopy data esample
geopy_sample = {
    "place_id": 327684232, 
    "licence": "Data \u00a9 OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright", 
    "osm_type": "way", 
    "osm_id": 437510561, 
    "lat": "39.832221333801186", 
    "lon": "-86.24921127905256", 
    "class": "highway", 
    "type": "secondary", 
    "place_rank": 26, 
    "importance": 0.10000999999999993, 
    "addresstype": "road", 
    "name": "Pike Plaza Road", 
    "display_name": "Pike Plaza Road, Indianapolis, Marion County, Indiana, 46254, United States", 
    "address": {"road": "Pike Plaza Road", 
                "city": "Indianapolis", 
                "county": "Marion County", 
                "state": "Indiana", 
                "ISO3166-2-lvl4": "US-IN", 
                "postcode": "46254", 
                "country": "United States", 
                "country_code": "us"}, 
    "boundingbox": ["39.8322034", "39.8324807", "-86.2492452", "-86.2487207"]}

# %%
geopy_sample.keys()

# %% [markdown]
# ### GeoPy Keys:
#
# - place_id: identificatore numerico univoco del luogo.
#
# - licence: licenza per uso dei dati geografici.
#
# - osm_type: tipo di oggetto OpenStreetMap (OSM) al quale appartiene la posizione ("node" per un punto, "way" per una strada o "relation" per una relazione tra elementi).
#
# - osm_id: identificatore univoco assegnato all'oggetto OSM.
#
# - lat + lon: latitudine e longitudine della posizione.
#
# - class: classificazione della posizione (es. "place").
#
# - type: classificazione della posizione (es. "city").
#
# - place_rank: Rango o la priorità del luogo nella gerarchia geografica (quanto una posizione è significativa).
#
# - importance: Valore numerico, indica l'importanza della posizione rispetto ad altre posizioni.
#
# - addresstype: tipo di indirizzo (es. "house", "street", "postcode")
#
# - name: nome del luogo (es.nome di una città o di una strada).
#
# - display_name: rappresentazione leggibile per l'utente della posizione, spesso formattata come un indirizzo completo.
#
# - address: indirizzo dettagliato.
#
# - boundingbox: elenco di quattro coordinate (latitudine e longitudine) che definiscono un rettangolo che racchiude la posizione (è un'approx dell'area coperta dalla posizione).

# %% [markdown]
# Usefull additional features from GeoPy:
# - importance and/or rank
# - address: to check with our dataset
#     - "road"
#     - "city"
#     - "county"
#     - "state"
# - class and/or type: to classify incident's place
# - adresstype: to classify incident's place
#
# Se si vuole fare check per luoghi che corrispondono tra loro:
# - osm_id
# - boundingbox
#

# %% [markdown]
# ## Import counties data from Wikipedia 

# %%
counties_path = FOLDER + 'wikipedia/counties.csv'

counties_data = pd.read_csv(counties_path)
counties_data.head()

# %% [markdown]
# ## 'state', 'city_or_county', 'address', 'latitude', 'longitude' consistency

# %%
from clean_data_utils import check_address_consistency, state_county_consistency

# %%
data_check_consistency = pd.DataFrame(columns=['state', 'city_or_county', 'address', 'latitude', 'longitude', 
    'road_geopy', 'city_geopy', 'county_geopy', 'state_geopy'])



# %%
data_check_consistency[['state', 'city_or_county', 'address', 'latitude', 'longitude']] = incidents_data[[
    'state', 'city_or_county', 'address', 'latitude', 'longitude']]

# %%
# TODO: read data from geopy file and file and fill data_check_consistency dataframe matching latitude and longitude

# %%
clean_geo_data = pd.DataFrame(columns=['state', 'city', 'county', 'road', 'latitude', 'longitude', 'adresstype', 'importance'])
clean_geo_data.index(data_check_consistency.index)

# %%
incomplete_data_count = 0

for index, row in data_check_consistency.iterrows():
    state_consistency, county_consistency, city_consistency = check_address_consistency(row)
    if (state_consistency + county_consistency + city_consistency) >= 2:
        # set geopy data
        clean_geo_data.loc[index, 'state'] = row['state_geopy']
        clean_geo_data.loc[index, 'county'] = row['county_geopy']
        clean_geo_data.loc[index, 'city'] = row['city_geopy']
        clean_geo_data.loc[index, 'road'] = row['road_geopy']
        clean_geo_data.loc[index, 'latitude'] = row['latitude']
        clean_geo_data.loc[index, 'longitude'] = row['longitude']
        clean_geo_data.loc[index, 'adresstype'] = row['adresstype']
        clean_geo_data.loc[index, 'importance'] = row['importance']
    else:
        incomplete_data_count += 1
        state, county = state_county_consistency(row['state'], row['city_or_county'], counties_data)

        if state_consistency:
            clean_geo_data.loc[index, 'state'] = row['state_geopy']
        elif state is not None:
            clean_geo_data.loc[index, 'state'] = state 
        
        if county_consistency:
            clean_geo_data.loc[index, 'county'] = row['county_geopy']
        elif county is not None:
            clean_geo_data.loc[index, 'county'] = county


# %%
print('Number of incomplete entries: ', incomplete_data_count)

# %%
clean_geo_data.head(10)

# %%
clean_geo_data.to_csv(FOLDER + 'post-proc/new_columns_geo.csv', index=False)

# %% [markdown]
# ## 'congressional_district', 'state_house_district', 'state_senate_district' consistency

# %%
#TODO: find dataset to match congressional_district, state_house_district, state_senate_district
# check if state with similar latitude and longitude are in the same 
# congressional_district, state_house_district, state_senate_district


