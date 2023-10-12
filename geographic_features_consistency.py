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

additional_data = pd.read_csv(counties_path)
additional_data.head()

# %%
additional_data.dtypes

# %% [markdown]
# ## 'state', 'city_or_county', 'address', 'latitude', 'longitude' consistency

# %%
data_check_consistency = pd.DataFrame(columns=['state', 'city_or_county', 'address', 'latitude', 'longitude', 
    'road_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy'])

# %%
data_check_consistency[['state', 'city_or_county', 'address', 'latitude', 'longitude']] = incidents_data[[
    'state', 'city_or_county', 'address', 'latitude', 'longitude']]

# %%
geopy_path = FOLDER + 'geopy/geopy.csv'
geopy_data = pd.read_csv(geopy_path)
geopy_data.head()

# %%
# display geopy_data columns names
geopy_data.columns # country, state, city, road

# %%
print('Number of rows in which state is null: ', geopy_data[geopy_data['state'].isnull()].shape[0])
print('Number of rows in which county is null: ', geopy_data[geopy_data['county'].isnull()].shape[0])
print('Number of rows in which city is null: ', geopy_data[geopy_data['city'].isnull()].shape[0])
print('Number of rows in which town is null: ', geopy_data[geopy_data['town'].isnull()].shape[0])
print('Number of rows in which road is null: ', geopy_data[geopy_data['road'].isnull()].shape[0])
print('Number of rows in which addresstype is null: ', geopy_data[geopy_data['addresstype'].isnull()].shape[0])
print('Number of rows in which importance is null: ', geopy_data[geopy_data['importance'].isnull()].shape[0])

# %%
print('Number of rows in which city is null and town is not null: ', 
    geopy_data[(geopy_data['city'].isnull()) & (geopy_data['town'].notnull())].shape[0])

# %%
data_check_consistency[['road_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy']] = geopy_data[[
    'road', 'town', 'city', 'county', 'state']]

# %%
data_check_consistency.head()

# %%
# convert latitude and longitude to float
data_check_consistency['latitude'] = data_check_consistency['latitude'].astype(float)
data_check_consistency['longitude'] = data_check_consistency['longitude'].astype(float)

# %%
clean_geo_data = pd.DataFrame(columns=['state', 'city', 'county', 'road', 'latitude', 'longitude'])
clean_geo_data = clean_geo_data.reindex(incidents_data.index)

# %%
from clean_data_utils import check_geographical_data_consistenc
clean_geo_data = data_check_consistency.apply(lambda row: 
    check_geographical_data_consistenc(row, additional_data=additional_data), axis=1)

# %%
print('Number of rows with all null values: ', clean_geo_data.isnull().all(axis=1).sum())
print('Number of rows with null value for state: ', clean_geo_data['state'].isnull().sum())
print('Number of rows with null value for county: ', clean_geo_data['county'].isnull().sum())
print('Number of rows with null value for city: ', clean_geo_data['city'].isnull().sum())
print('Number of rows with null value for road: ', clean_geo_data['road'].isnull().sum())
print('Number of rows with null value for latitude: ', clean_geo_data['latitude'].isnull().sum())
print('Number of rows with null value for longitude: ', clean_geo_data['longitude'].isnull().sum())

# %%
clean_geo_data.head(10)

# %%
clean_geo_data.to_csv(FOLDER + 'post_proc/new_columns_geo.csv', index=False)

# %% [markdown]
# ## 'congressional_district', 'state_house_district', 'state_senate_district' consistency

# %%
#TODO: find dataset to match congressional_district, state_house_district, state_senate_district
# check if state with similar latitude and longitude are in the same 
# congressional_district, state_house_district, state_senate_district


