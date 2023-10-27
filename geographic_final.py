# -*- coding: utf-8 -*-
# %% [markdown]
# # Geographic features

# %% [markdown]
# ## Import data

# %%
import utils
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plot_utils
import sys
sys.path.append(os.path.abspath('..\\')) # TODO: c'è un modo per farlo meglio?


# %%
# read data
dirname = os.path.dirname(' ')
FOLDER = os.path.join(dirname, 'data')
incidents_path = os.path.join(FOLDER, 'incidents.csv')
incidents_data = pd.read_csv(incidents_path, low_memory=False)

# %%
# drop duplicates rows
incidents_data.drop_duplicates(inplace=True)
incidents_data.info()

# %%
LOAD_DATA_FROM_CHECKPOINT = True # boolean: True if you want to load data, False if you want to compute it
CHECKPOINT_FOLDER_PATH = 'data/checkpoints/'

def checkpoint(df, checkpoint_name):
    df.to_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv')

def load_checkpoint(checkpoint_name, casting={}):
    #d_p = pd.datetools.to_datetime
    if casting:
        return pd.read_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv', low_memory=False, index_col=0, parse_dates=['date'], dtype=casting)
    else: #TODO: sistemare il casting quando ci sono tutte le colonne 
        return pd.read_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv', low_memory=False, index_col=0)#, parse_dates=['date'])


# %% [markdown]
# ## Da qua

# %%
# select only relevant columns from incidents_data
geo_data = incidents_data[['date', 'state', 'city_or_county', 'address', 'latitude', 'longitude',
       'congressional_district', 'state_house_district', 'state_senate_district']]
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
counties_path = os.path.join(FOLDER, 'wikipedia/counties.csv')

additional_data = pd.read_csv(counties_path)
additional_data.head()

# %%
additional_data.dtypes

# %% [markdown]
# ## 'state', 'city_or_county', 'address', 'latitude', 'longitude' consistency

# %%
geopy_path = os.path.join(FOLDER, 'geopy/geopy.csv')
geopy_data = pd.read_csv(geopy_path, index_col=['index'], low_memory=False, dtype={})
geopy_data.info()

# %%
geopy_data.loc[geopy_data['suburb'].isna()]

# %%
for col in geopy_data:
    dummy = geopy_data[col].unique()
    print( [ col, dummy, len(dummy)] )

# %%
print('Number of rows without coordinates: ', geopy_data['coord_presence'].value_counts())
print('Number of rows without importance: ', geopy_data['importance'].isnull().value_counts())

# %%
print('Number of rows in which city is null and town is not null: ', 
    geopy_data[(geopy_data['city'].isnull()) & (geopy_data['town'].notnull())].shape[0])

# %%
geopy_data['addresstype'].unique()

# %%
print('Number of rows in which class is null: ', geopy_data[geopy_data['class'].isnull()].shape[0])
print('Number of rows in which addresstype is null: ', geopy_data[geopy_data['addresstype'].isnull()].shape[0])

# %%
print('Number of rows in which class is null: ', geopy_data[geopy_data['class'].isnull()].shape[0])
print('Number of rows in which addresstype is null: ', geopy_data[geopy_data['addresstype'].isnull()].shape[0])

# %%
data_check_consistency = pd.DataFrame(columns=['state', 'city_or_county', 'address', 'latitude', 'longitude', 'display_name', 
    'village_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy', 'importance_geopy', 'addresstype_geopy', 
    'coord_presence', 'suburb_geopy'])
data_check_consistency[['state', 'city_or_county', 'address', 'latitude', 'longitude']] = incidents_data[[
    'state', 'city_or_county', 'address', 'latitude', 'longitude']]
    
data_check_consistency[['address_geopy', 'village_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy', 
    'importance_geopy', 'addresstype_geopy', 'coord_presence', 'suburb_geopy']] = geopy_data.loc[incidents_data.index][['display_name', 'village', 'town', 'city', 
    'county', 'state', 'importance', 'addresstype', 'coord_presence', 'suburb']]



# %%
data_check_consistency.head(2)

# %%
# convert latitude and longitude to float
data_check_consistency['latitude'] = data_check_consistency['latitude'].astype(float)
data_check_consistency['longitude'] = data_check_consistency['longitude'].astype(float)

# %%
data_check_consistency[(data_check_consistency['coord_presence'] == True) & (data_check_consistency['importance_geopy'].isnull())]

# %%
data_check_consistency['town_geopy'].loc[0]

# %%
pd.isnull(data_check_consistency['town_geopy'].loc[0])

# %%
from utils import check_geographical_data_consistency_2

if LOAD_DATA_FROM_CHECKPOINT: # load data
    clean_geo_data = load_checkpoint('checkpoint_age_temporary')
else: # compute data
    clean_geo_data = data_check_consistency.apply(lambda row: check_geographical_data_consistency_2(row), axis=1)
    checkpoint(clean_geo_data, 'checkpoint_geo_temporary') # save data

# %%
print('Number of rows with all null values: ', clean_geo_data.isnull().all(axis=1).sum())
print('Number of rows with null value for state: ', clean_geo_data['state'].isnull().sum())
print('Number of rows with null value for county: ', clean_geo_data['county'].isnull().sum())
print('Number of rows with null value for city: ', clean_geo_data['city'].isnull().sum())
print('Number of rows with null value for latitude: ', clean_geo_data['latitude'].isnull().sum())
print('Number of rows with null value for longitude: ', clean_geo_data['longitude'].isnull().sum())

# %%
clean_geo_data.head(3)

# %%

clean_geo_data.groupby(['state_consistency',	'county_consistency','address_consistency']).count().sort_index(ascending=False)

# %%
dummy = {}
stats_columns = ['null_val', 'not_null', 'value_count']
for col in clean_geo_data.columns:
    dummy[col] = []
    dummy[col].append(clean_geo_data[col].isna().sum())
    dummy[col].append(len(clean_geo_data[col]) - clean_geo_data[col].isna().sum())
    dummy[col].append(len(clean_geo_data[col].value_counts()))
    
print(dummy)
clean_geo_stat_stats = pd.DataFrame(dummy, index=stats_columns).transpose()
clean_geo_stat_stats

# %%
a = len(clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].notna())])
b = len(clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].isna())])
c = len(clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())])
d = len(clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].isna())])
e = len(clean_geo_data.loc[(clean_geo_data['latitude'].isna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].notna())])
f = len(clean_geo_data.loc[(clean_geo_data['latitude'].isna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].isna())])
g = len(clean_geo_data.loc[(clean_geo_data['latitude'].isna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())])
h = len(clean_geo_data.loc[(clean_geo_data['latitude'].isna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].isna())])

print('LAT/LONG --- COUNTY --- CITY')
print( ' 0 --- 0 --- 0\t', a)
print( ' 0 --- 0 --- 1\t', b)
print( ' 0 --- 1 --- 0\t', c)
print( ' 0 --- 1 --- 1\t', d)
print( ' 1 --- 0 --- 0\t', e)
print( ' 1 --- 0 --- 1\t', f)
print( ' 1 --- 1 --- 0\t', g)
print( ' 1 --- 1 --- 1\t', h)
print( ' ---- TOT ----\t', a+b+c+d+e+f+g+h)
print( ' ---- GOOD ---\t', a+b+c+d)
print( ' ---- BAD ----\t', e+f+g+h)

# %%
dummy_data = clean_geo_data[clean_geo_data['latitude'].notna()]
print(len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %%
dummy_data = clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())]
print(len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %%
clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())].groupby('city').count()

# %%
clean_geo_data.loc[(clean_geo_data['latitude'].isna())].groupby('city').count()

# %%
missing_county={'Missouri':'Saint Louis County', 'Denver':'Denver County', 'Juneau': 'Juneau County', 'San Francisco': 'San Francisco County' }

# %%
dummy_data = clean_geo_data[(clean_geo_data['latitude'].notna()) & (clean_geo_data['city'].isna()) & (clean_geo_data['county'].isna())]
print(len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %%
dummy_data = clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].isna())]
print(len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %%
print(clean_geo_data.columns)

# %%

#clean_geo_data = clean_geo_data.astype({"a": int, "b": complex})

#clean_geo_data.to_csv(FOLDER + 'post_proc/new_columns_geo.csv', index=False)

# %% [markdown]
# # FINAL EVALUATIONS:
# We divided the dataset into several groups depending on what information we were able to demonstrate consistent between the latitude, longitude, state, county, and city fields. And we did this by also making use of the address field, which, however, we decided not to use further because it is not very identifying of the line and is too variable. Finally, we defined strategies to be applied on these groups to fill in the missing data (considered erroneous or missing from the original dataset) in a more or less effective way according to the row information.
#
# We now report the division into disjointed groups in which we indicate the size
#
# ---------- GOOD GROUPS ----------
# * 174796 = The completely consistent and final rows of the dataset.
# * 26635 = The rows in which only the city is missing that can be inferred easily from the location (k-nn)
# * 15000 = The rows in which only the county is missing that can be inferred easily from the location (k-nn)
# * 33 = The rows where city and county are missing, also in this group the missing information can be inferred from the location (All clustered close to Baltimore)
#
# ---------- BAD GROUPS ----------
# * 3116 = The rows where latitude and longitude and city are missing, they can be inferred (not faithfully) from the pair county-state
# * 19844 = The rows in which only the state field is present. difficult to retrieve
#
# missing combinations are not present in the dataset
#
# # Final considerations
# as many as 216464 lines are either definitive or can be derived with a good degree of fidelity. the remainder must be handled carefully\
#
# CAUTION: EVALUATE THE CHOSEN TRESHOULDS


# %%
incidents_data.groupby(['state', 'county', 'city']).count()

# %%
incidents_data['latitude'].isna().sum()

# %%
for f in incidents_data['latitude']:
    a = []
    a.append(np.isnan(f))
sum(a)

# %%
for i in incidents_data.loc[incidents_data['city'].isna() & np.isnan(incidents_data['latitude'])].index:
    print(incidents_data.loc[i]['latitude'])

# %%
a = len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].notna()) & (incidents_data['city'].notna())])
b = len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].notna()) & (incidents_data['city'].isna())])
c = len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].isna()) & (incidents_data['city'].notna())])
d = len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].isna()) & (incidents_data['city'].isna())])
e = len(incidents_data.loc[(incidents_data['latitude'].isna()) & (incidents_data['county'].notna()) & (incidents_data['city'].notna())])
f = len(incidents_data.loc[(incidents_data['latitude'].isna()) & (incidents_data['county'].notna()) & (incidents_data['city'].isna())])
g = len(incidents_data.loc[(incidents_data['latitude'].isna()) & (incidents_data['county'].isna()) & (incidents_data['city'].notna())])
h = len(incidents_data.loc[(incidents_data['latitude'].isna()) & (incidents_data['county'].isna()) & (incidents_data['city'].isna())])

print('LAT/LONG --- COUNTY --- CITY')
print( ' 0 --- 0 --- 0\t', a)
print( ' 0 --- 0 --- 1\t', b)
print( ' 0 --- 1 --- 0\t', c)
print( ' 0 --- 1 --- 1\t', d)
print( ' 1 --- 0 --- 0\t', e)
print( ' 1 --- 0 --- 1\t', f)
print( ' 1 --- 1 --- 0\t', g)
print( ' 1 --- 1 --- 1\t', h)
print( ' ---- TOT ----\t', a+b+c+d+e+f+g+h)
print( ' ---- GOOD ---\t', a+b+c+d)
print( ' ---- BAD ----\t', e+f+g+h)

# %%
centroids = incidents_data.loc[incidents_data['latitude'].notna() & incidents_data['city'].notna()][['latitude', 'longitude', 'city', 'state', 'county']].groupby(['state', 'county', 'city']).mean()
centroids.head(10)

# %%
print(centroids.index.to_list())

# %%
centroids.sample()

# %%
info_city = pd.DataFrame(columns=['5', '15', '25', '35', '45', '55', '65', '75', '85', '95', 'tot_points', 'min', 'max', 'avg', 'centroid_lat', 'centroid_lon'], index=centroids.index)
info_city.info()

# %%
for state, county, city in centroids.index:
    dummy = []
    for lat, long in zip(incidents_data.loc[(incidents_data['city'] == city) & (incidents_data['state'] == state) & (incidents_data['county'] == county) & incidents_data['latitude'].notna()]['latitude'], 
                         incidents_data.loc[(incidents_data['city'] == city) & (incidents_data['state'] == state) & (incidents_data['county'] == county) & incidents_data['longitude'].notna()]['longitude']):
        dummy.append(geopy.distance.geodesic([lat, long], centroids.loc[state, county, city]).km)
    dummy = sorted(dummy)
    pc = np.quantile(dummy, np.arange(0,1, 0.05))
    for i in range(len(info_city.columns) - 6):
        info_city.loc[state, county, city][i] = pc[i*2 + 1]
    info_city.loc[state, county, city][len(info_city.columns) - 6] = len(dummy)
    info_city.loc[state, county, city][len(info_city.columns) - 5] = min(dummy)
    info_city.loc[state, county, city][len(info_city.columns) - 4] = max(dummy)
    info_city.loc[state, county, city][len(info_city.columns) - 3] = sum(dummy)/len(dummy)
    info_city.loc[state, county, city][len(info_city.columns) - 2] = centroids.loc[state, county, city]['latitude']
    info_city.loc[state, county, city][len(info_city.columns) - 1] = centroids.loc[state, county, city]['longitude']




# %%
info_city

# %%
info_city.loc[info_city['tot_points'] > 1].info()

# %%
plot_utils.plot_scattermap_plotly(info_city, 'tot_points', x_column='centroid_lat', y_column='centroid_lon', hover_name=False)

# %%
for i in [  5955,  19567,  22995,  23433,  35631,  39938,  45163,  55557,  55868,
        60596,  65016,  69992,  70730,  73290,  73949,  78689, 104390, 116673,
       133043, 150273, 153933, 160492, 162559, 178887, 192938, 196820, 206125,
       225494, 227231, 227287, 230283]:
       print(incidents_data.iloc[i][['latitude', 'longitude']])
print(incidents_data.iloc[i])

# %%
incidents_data.sample()

# %%
def substitute_city(row, info_city):
    if pd.isna(row['city']) and not np.isnan(row['latitude']):
        for state, county, city in info_city.index:
            if row['state'] == state and row['county'] == county:
                if info_city.loc[state, county, city]['tot_points'] > 1:
                    max_radius = info_city.loc[state, county, city]['75'] #0.75 esimo quantile
                    centroid_coord = [info_city.loc[state, county, city]['centroid_lat'], info_city.loc[state, county, city]['centroid_lon']]
                    if geopy.distance.geodesic([row['latitude'], row['longitude']], centroid_coord).km <= max_radius:
                            row['city'] = city
                            break
                    
    return row


final_geo_data = incidents_data.apply(lambda row: substitute_city(row, info_city), axis=1)

# %%
checkpoint(final_geo_data, 'checkpoint_geo') # save data

# %%
a = len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].notna())])
b = len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].isna())])
c = len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].isna()) & (final_geo_data['city'].notna())])
d = len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].isna()) & (final_geo_data['city'].isna())])
e = len(final_geo_data.loc[(final_geo_data['latitude'].isna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].notna())])
f = len(final_geo_data.loc[(final_geo_data['latitude'].isna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].isna())])
g = len(final_geo_data.loc[(final_geo_data['latitude'].isna()) & (final_geo_data['county'].isna()) & (final_geo_data['city'].notna())])
h = len(final_geo_data.loc[(final_geo_data['latitude'].isna()) & (final_geo_data['county'].isna()) & (final_geo_data['city'].isna())])

print('LAT/LONG --- COUNTY --- CITY')
print( ' 0 --- 0 --- 0\t', a)
print( ' 0 --- 0 --- 1\t', b)
print( ' 0 --- 1 --- 0\t', c)
print( ' 0 --- 1 --- 1\t', d)
print( ' 1 --- 0 --- 0\t', e)
print( ' 1 --- 0 --- 1\t', f)
print( ' 1 --- 1 --- 0\t', g)
print( ' 1 --- 1 --- 1\t', h)
print( ' ---- TOT ----\t', a+b+c+d+e+f+g+h)
print( ' ---- GOOD ---\t', a+b+c+d)
print( ' ---- BAD ----\t', e+f+g+h)

# %%
final_geo_data.to_csv(os.path.join(dirname, 'data/post_proc/final_incidents_city_inf.csv'))
info_city.to_csv(os.path.join(dirname, 'data/post_proc/info_city.csv'))


# %%
a = len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].notna()) & (incidents_data['city'].notna())])
b = len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].notna()) & (incidents_data['city'].isna())])
c = len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].isna()) & (incidents_data['city'].notna())])
d = len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].isna()) & (incidents_data['city'].isna())])
e = len(incidents_data.loc[(incidents_data['latitude'].isna()) & (incidents_data['county'].notna()) & (incidents_data['city'].notna())])
f = len(incidents_data.loc[(incidents_data['latitude'].isna()) & (incidents_data['county'].notna()) & (incidents_data['city'].isna())])
g = len(incidents_data.loc[(incidents_data['latitude'].isna()) & (incidents_data['county'].isna()) & (incidents_data['city'].notna())])
h = len(incidents_data.loc[(incidents_data['latitude'].isna()) & (incidents_data['county'].isna()) & (incidents_data['city'].isna())])

print('LAT/LONG --- COUNTY --- CITY')
print( ' 0 --- 0 --- 0\t', a)
print( ' 0 --- 0 --- 1\t', b)
print( ' 0 --- 1 --- 0\t', c)
print( ' 0 --- 1 --- 1\t', d)
print( ' 1 --- 0 --- 0\t', e)
print( ' 1 --- 0 --- 1\t', f)
print( ' 1 --- 1 --- 0\t', g)
print( ' 1 --- 1 --- 1\t', h)
print( ' ---- TOT ----\t', a+b+c+d+e+f+g+h)
print( ' ---- GOOD ---\t', a+b+c+d)
print( ' ---- BAD ----\t', e+f+g+h)

# %%
plot_utils.plot_scattermap_plotly(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['county'].notna()) & (incidents_data['city'].isna())], 'state')

# %%
plot_utils.plot_scattermap_plotly(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['state'] == 'Missouri') & (incidents_data['county'] == 'Platte County') & (incidents_data['city'] == 'Kansas City')], 'latitude')
len(incidents_data.loc[(incidents_data['latitude'].notna()) & (incidents_data['state'] == 'Missouri') & (incidents_data['county'] == 'Platte County') & (incidents_data['city'] == 'Kansas City')])

# %%
#TODO: plottare le città che ha inserto e i centroidi??


# %%
final_geo_data

# %%
incidents_data.head(3)

# %%
geopy_data.head(3)

# %%
dummy = incidents_data.loc[final_geo_data.loc[final_geo_data['latitude'].isna()].index]
interesting_index = dummy.loc[dummy['latitude'].notna()].index
incidents_data.loc[interesting_index][['state', 'city_or_county', 'address', 'latitude', 'longitude']]

# %%
import random
new_ii = random.sample(interesting_index.to_list(), 15)

# %%
incidents_data.loc[new_ii][['state', 'city_or_county', 'address', 'latitude', 'longitude']]

# %%
geopy_data.loc[new_ii][['class', 'type', 'addresstype', 'display_name', 'state', 'county', 'city', 'town', 'suburb', 'neighbourhood']]

# %%
col_value_count = []
for col in geopy_data.columns:

    col_value_count.append([col, geopy_data.loc[(geopy_data['county'].isna()) & (geopy_data['lat'].notna())][col].count()])

col_value_count.sort(key=lambda x: x[1], reverse=True)

for c in col_value_count:
    print(c[0], c[1])

print(col_value_count[9][1] > col_value_count[8][1])

# %%
geopy_data.loc[(geopy_data['county'].isna()) & (geopy_data['lat'].notna())].groupby('suburb').count().sort_values('place_id', ascending=False).head(20)

# %%
geopy_data.loc[229463]['display_name']

# %%
for i in geopy_data.loc[215070].index:
    print(i, geopy_data.loc[215070][i])

# %%
incidents_data.loc[(incidents_data['latitude'] == '39.7591')]

# %%
utils.check_address(incidents_data.loc[239662]['address'], geopy_data.loc[239662]['display_name'])

# %%
incidents_data.loc[108203]['address']

# %%
geopy_data.loc[108203]

# %%
dummy_data = final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].isna())]
print(len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %%
dummy_data = final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['city'].isna())]
print(len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')
