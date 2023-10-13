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
incidents_data.info()

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
    'display_name', 'village_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy', 'coord_presence'])

# %%
data_check_consistency[['state', 'city_or_county', 'address', 'latitude', 'longitude']] = incidents_data[[
    'state', 'city_or_county', 'address', 'latitude', 'longitude']]

# %%
geopy_path = FOLDER + 'geopy/geopy.csv'
geopy_data = pd.read_csv(geopy_path, index_col=['index'])
geopy_data.info()

# %%
for col in geopy_data:
    dummy = geopy_data[col].unique()
    print( [ col, dummy, len(dummy)] )

# %%
'''print('Number of rows in which state is null: ', geopy_data[geopy_data['state'].isnull()].shape[0])
print('Number of rows in which county is null: ', geopy_data[geopy_data['county'].isnull()].shape[0])
print('Number of rows in which city is null: ', geopy_data[geopy_data['city'].isnull()].shape[0])
print('Number of rows in which town is null: ', geopy_data[geopy_data['town'].isnull()].shape[0])
print('Number of rows in which road is null: ', geopy_data[geopy_data['road'].isnull()].shape[0])
print('Number of rows in which addresstype is null: ', geopy_data[geopy_data['addresstype'].isnull()].shape[0])
print('Number of rows in which importance is null: ', geopy_data[geopy_data['importance'].isnull()].shape[0])'''

print('Number of rows without coordinates: ', geopy_data['coord_presence'].value_counts())
print('Number of rows without importance: ', geopy_data['importance'].isnull().value_counts())

# %%
geopy_data.groupby(['class']).count()

# %%
data_check_consistency[['address_geopy', 'village_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy', 'coord_presence']] = geopy_data[[
    'display_name', 'village', 'town', 'city', 'county', 'state', 'coord_presence']]

# %%
data_check_consistency.head()

# %%
# convert latitude and longitude to float
data_check_consistency['latitude'] = data_check_consistency['latitude'].astype(float)
data_check_consistency['longitude'] = data_check_consistency['longitude'].astype(float)

# %%
from utils import check_geographical_data_consistency

clean_geo_data = data_check_consistency.apply(lambda row: 
    check_geographical_data_consistency(row, additional_data=additional_data), axis=1)

# %%
print('Number of rows with all null values: ', clean_geo_data.isnull().all(axis=1).sum())
print('Number of rows with null value for state: ', clean_geo_data['state'].isnull().sum())
print('Number of rows with null value for county: ', clean_geo_data['county'].isnull().sum())
print('Number of rows with null value for city: ', clean_geo_data['city'].isnull().sum())
print('Number of rows with null value for latitude: ', clean_geo_data['latitude'].isnull().sum())
print('Number of rows with null value for longitude: ', clean_geo_data['longitude'].isnull().sum())

# %%
clean_geo_data.head(10)

# %%
print('Number of rows divided by state_consistency: ', clean_geo_data['state_consistency'].value_counts())
print('Number of rows divided by county_consistency: ', clean_geo_data['county_consistency'].value_counts())
print('Number of rows divided by address_consistency: ', clean_geo_data['address_consistency'].value_counts())

clean_geo_data.info()

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


# %%
import plotly.express as px
import numpy as np
"""fig = px.choropleth(locations=incidents_data['state'].value_counts().index,
                    locationmode="USA-states",
                    color=incidents_data['state'].value_counts().values,
                    scope="usa",
                    color_continuous_scale="Viridis",
                    title='Number of incidents in each state')"""

color_scale = [(0, 'orange'), (1,'blue')]
fig = px.scatter_mapbox(color=np.ones(clean_geo_data.shape[0]), 
                        lat=clean_geo_data['latitude'], 
                        lon=clean_geo_data['longitude'],
                        color_continuous_scale=color_scale,
                        zoom=3, 
                        height=800,
                        width=800)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %%
only_nan_coord = clean_geo_data[clean_geo_data['latitude'].isnull()]

for col in only_nan_coord:
    dummy = only_nan_coord[col].unique()
    print( [ col, dummy, len(dummy)] )

# %%
print('Number of rows divided state:\n', only_nan_coord['state'].value_counts())

# %%
print('Number of nan: ', only_nan_coord['county'].isna().sum())
print('Number of not nan: ', len(only_nan_coord['county']) - only_nan_coord['county'].isna().sum())

# %%
#TODO: aggiungere tutte le colonne di 'incidenti' significative e mancanti, e aggiungere colonne aggiuntive geopy solo nelle righe sensatte
#TODO: per farlo aggiungere booleano della utils per capire se i dati salvati sono o no di geopy

clean_geo_data.to_csv(FOLDER + 'post_proc/new_columns_geo.csv', index=False)

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
# * 33 = The rows where city and county are missing, also in this group the missing information can be inferred from the location (k-nn)
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

# %% [markdown]
# ## 'congressional_district', 'state_house_district', 'state_senate_district' consistency

# %%
#TODO: find dataset to match congressional_district, state_house_district, state_senate_district
# check if state with similar latitude and longitude are in the same 
# congressional_district, state_house_district, state_senate_district


