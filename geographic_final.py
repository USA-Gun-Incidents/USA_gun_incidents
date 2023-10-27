# %% [markdown]
# # Geographic features

# %% [markdown]
# ## Import data

# %%
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plot_utils
import plotly.express as px
import sys
sys.path.append(os.path.abspath('..\\')) # TODO: c'è un modo per farlo meglio?

# %%
# read data
dirname = os.path.dirname(' ')
FOLDER = os.path.join(dirname, 'data')
incidents_path = os.path.join(FOLDER, 'incidents.csv')
incidents_data = pd.read_csv(incidents_path, low_memory=False)

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
# ## Geographic data

# %% [markdown]
# Columns of the dataset are considered in order to verify the correctness and consistency of data related to geographical features:
# - *state*
# - *city_or_county*
# - *address*
# - *latitude*
# - *longitude*

# %%
# select only relevant columns from incidents_data
incidents_data[['state', 'city_or_county', 'address', 'latitude', 'longitude',
       'congressional_district', 'state_house_district', 'state_senate_district']].head(5)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
incidents_data[['state', 'city_or_county', 'address', 'latitude', 'longitude']].info()

# %%
print('Number of rows with missing latitude: ', incidents_data['latitude'].isnull().sum())
print('Number of rows with missing longitude: ', incidents_data['longitude'].isnull().sum())

# %% [markdown]
# Plot incidents' location on a map:

# %%
fig = px.scatter_mapbox(
    lat=incidents_data['latitude'],
    lon=incidents_data['longitude'],
    zoom=0, 
    height=500,
    width=800
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %% [markdown]
# We note that, excluding 7923 entries in the dataset where latitude and longitude data are missing and 4 entries outside the borders of the USA, the remaining dataset entries have latitude and longitude values. The *state* field is non-null for all entries.
# 
# Therefore, we decided to use external sources to verify the data consistency in the dataset. For the fields where latitude and longitude were present within the United States, we utilized GeoPy data, while for the remaining data and those that did not have a positive match with GeoPy's results, we used data from Wikipedia.
# 
# In the following sections of the notebook, we provide a summary of the additional data used and the methodologies for verifying data consistency.

# %% [markdown]
# ### Additional dataset: GeoPy

# %% [markdown]
# In order to check the data consistency of our dataset, we use [GeoPy](https://geopy.readthedocs.io/en/stable/). We have previously saved the necessary data for performing these checks using the latitude and longitude data from our dataset.

# %% [markdown]
# We load GeoPy data from the CSV file where it is stored.

# %%
geopy_path = os.path.join(FOLDER, 'geopy/geopy.csv')
geopy_data = pd.read_csv(geopy_path, index_col=['index'], low_memory=False, dtype={})

# %% [markdown]
# geopy_data is a data frame containing one row for each row in our dataset, with matching indices.

# %% [markdown]
# **GeoPy Keys** we saved and used:
# 
# - *lat* and *lon*: Latitude and longitude of the location.
# 
# - *importance*: Numerical value $\in [0,1]$, indicates the importance of the location relative to other locations.
# 
# - *addresstype*: Address type (e.g., "house," "street," "postcode") to classify the incident's place.
# 
# - *address*: A dictionary containing detailed address information. \
#     Dictionary keys included in the data frame are: *state*, *county*, *suburb*, *city*, *town*, *village*.
# 
# - *display_name*: User-friendly representation of the location, often formatted as a complete address. Used by us to cross-reference with the address in case we are unable to find a match between our data and the GeoPy data set using other information from the address.
# 
# Additional columns we added to the dataframe:
# 
# - *coord_presence*: Boolean, false if the corresponding row in the original dataset did not have latitude and longitude values, making it impossible to query GeoPy.

# %%
print('Number of rows without surburbs: ', geopy_data.loc[geopy_data['suburb'].isna()].shape[0])
display(geopy_data.loc[geopy_data['suburb'].isna()].head(2))

# %%
print('Number of rows without coordinates: \n', geopy_data['coord_presence'].value_counts())
print('\nNumber of rows without importance: \n', geopy_data['importance'].isnull().value_counts())

# %%
print('Number of rows in which city is null and town is not null: ', 
    geopy_data[(geopy_data['city'].isnull()) & (geopy_data['town'].notnull())].shape[0])

# %%
print(geopy_data['addresstype'].unique())
print('Number of rows in which addresstype is null: ', geopy_data[geopy_data['addresstype'].isnull()].shape[0])

# %% [markdown]
# ### Additional dataset: Wikipedia

# %% [markdown]
# We also downloaded data from [Wikipedia](https://en.wikipedia.org/wiki/County_(United_States)) containing the states and counties (or the equivalent) for each state in the USA. This data was used in cases where no consistency was found with GeoPy data.
# 
# This dataset made it possible to verify the data consistency for the *state* and *county* fields without the need for *latitude* and *longitude* values.

# %%
counties_path = os.path.join(FOLDER, 'wikipedia/counties.csv')

additional_data = pd.read_csv(counties_path)
additional_data.head()

# %%
additional_data.dtypes

# %% [markdown]
# ### Studying Data Consistency

# %% [markdown]
# We concatenate geographic data from our dataset and GeoPy data into a single DataFrame.

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

# %% [markdown]
# Convert latitude and longitude to float type.

# %%
# convert latitude and longitude to float
data_check_consistency['latitude'] = data_check_consistency['latitude'].astype(float)
data_check_consistency['longitude'] = data_check_consistency['longitude'].astype(float)

# %% [markdown]
# Print the first 10 occurrences of city_or_county values where parentheses are present in the string:

# %%
[c for c in data_check_consistency['city_or_county'].unique() if '(' in c][:10]

# %%
len([c for c in data_check_consistency['city_or_county'].unique() if '(' in c and 'county' not in c])

# %% [markdown]
# We can note that in 1782 entries, both city and county values are present in city_or_county, so we need to take this into consideration.

# %% [markdown]
# We created some functions to check the consistency of geographic data using the external sources we mentioned earlier. We also replaced the values for State, County, and City with a string in title case, without punctuation or numbers, to obtain a dataset of clean and consistent data.
# 
# Below, we provide a brief summary of all the functions used to check data consistency and replace values when necessary.

# %% [markdown]
# **String Preprocessing**
# - Initially, we convert the values for state, county, and city to lowercase for both our dataset and external dataset.
# - For cases where the *city_or_county* field contained values for both city and county, we split the string into two and used both new strings to match with both county and city.
# - We removed the words 'city of' and 'county' from the strings in the *city_or_county* field.
# - We removed punctuation and numerical values from the string, if present, in the *state* and *city_or_county* fields.
# 
# **If We Had GeoPy Data**
# - We attempted direct comparisons with GeoPy data:
#     - Our data's *state* was compared with *state* from GeoPy.
#     - To assign the value of *county*, we compared with *county_geopy* and with *suburb_geopy*.
#     - The *city* was compared with *city_geopy*, *town_geopy* and *village_geopy*.
# - If direct comparisons were not positive:
#     - We checked for potential typos in the string using the Damerau-Levenshtein distance (definition below).
#     - Thresholds to decide the maximum distance for two strings to be considered equal were set after several preliminary tests. We decided to use different thresholds for state and city/county.
# - In cases where previous comparisons were not sufficient, we also used the *address* field from our dataset, comparing it with GeoPy's *display_name*, from which commonly occurring words throughout the column were removed (e.g., "Street," "Avenue," "Boulevard"). Again, we used the Damerau-Levenshtein distance with an appropriate threshold to verify address consistency.
# 
# In cases where we were able to evaluate data consistency through these comparisons, we set the values for the fields *state*, *county*, *city*, *latitude*, *longitude*, *importance*, *address_type* using GeoPy values. Additionally, we also saved values reflecting the consistency with the fields evaluated earlier in: *state_consistency*, *county_consistency*, *address_consistency* (0 if not consistent, 1 if consistent, -1 if null values are presents)
# 
# If the fields in our dataset were not consistent through the previously described checks or could not be verified due to the absence of latitude and longitude values, we attempted to assess consistency using Wikipedia data, with similar checks as before. In this case, we could only retrieve the *state* and *county* fields.

# %% [markdown]
# General formula for calculating the **Damerau-Levenshtein distance** between two strings $s$ and $t$ \
# $D(i, j) = \min
# \begin{cases}
# D(i-1, j) + 1 \\
# D(i, j-1) + 1 \\
# D(i-1, j-1) + \delta \\
# D(i-2, j-2) + \delta & \text{if } s[i] = t[j] \text{ and } s[i-1] = t[j-1]
# \end{cases}$
# 
# where:
# - $D(i, j)$ is the Damerau-Levenshtein distance between the first $i$ letters of a string $s$ and the first $j$ letters of a string $t$.
# - $\delta$ is 0 if the current letters $s[i]$ and $t[j]$ are equal, otherwise, it is 1.
# - $D(i-2, j-2) + \delta$ represents transposition (swapping two adjacent letters) if the current letters $s[i]$ and $t[j]$ are equal, and the preceding letters $s[i-1]$ and $t[j-1]$ are also equal.
# 
# 

# %%
from data_preparation_utils import check_geographical_data_consistency

if LOAD_DATA_FROM_CHECKPOINT: # load data
    clean_geo_data = load_checkpoint('checkpoint_geo_temporary')
else: # compute data
    clean_geo_data = data_check_consistency.apply(lambda row: check_geographical_data_consistency(row, 
        additional_data=additional_data), axis=1)
    checkpoint(clean_geo_data, 'checkpoint_geo_temporary') # save data

# %% [markdown]
# ### Visualize Consistent Geographical Data

# %%
clean_geo_data.head(2)

# %%
print('Number of rows with all null values: ', clean_geo_data.isnull().all(axis=1).sum())
print('Number of rows with null value for state: ', clean_geo_data['state'].isnull().sum())
print('Number of rows with null value for county: ', clean_geo_data['county'].isnull().sum())
print('Number of rows with null value for city: ', clean_geo_data['city'].isnull().sum())
print('Number of rows with null value for latitude: ', clean_geo_data['latitude'].isnull().sum())
print('Number of rows with null value for longitude: ', clean_geo_data['longitude'].isnull().sum())

# %%
clean_geo_data['state'].unique().shape[0]

# %% [markdown]
# After this check, all the entries in the dataset have at least the state value not null and consistent. Only 12,796 data points, which account for 4.76% of the dataset, were found to have inconsistent latitude and longitude values.

# %% [markdown]
# Below, we have included some plots to visualize the inconsistency values in the dataset.

# %%
clean_geo_data.groupby(['state_consistency','county_consistency','address_consistency']).count().sort_index(ascending=False)

# %%
dummy = {}
stats_columns = ['#null_val', '#not_null', '#value_count']
for col in clean_geo_data.columns:
    dummy[col] = []
    dummy[col].append(clean_geo_data[col].isna().sum())
    dummy[col].append(len(clean_geo_data[col]) - clean_geo_data[col].isna().sum())
    dummy[col].append(len(clean_geo_data[col].value_counts()))
    
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

print('LAT/LONG     COUNTY     CITY             \t#samples')
print( 'not null    not null   not null         \t', a)
print( 'not null    not null   null             \t', b)
print( 'not null    null       not null         \t', c)
print( 'not null    null       null             \t', d)
print( 'null        not null   not null         \t', e)
print( 'null        not null   null             \t', f)
print( 'null        null       null             \t', g)
print( 'null        null       null             \t', h)
print('\n')
print( 'TOT samples                             \t', a+b+c+d+e+f+g+h)
print( 'Samples with not null values for lat/lon\t', a+b+c+d)
print( 'Samples with null values for lat/lon    \t', e+f+g+h)

# %%
dummy_data = clean_geo_data[clean_geo_data['latitude'].notna()]
print('Number of entries with not null values for latitude and longitude: ', len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %%
dummy_data = clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())]
print('Number of entries with not null values for latitude, longitude, county and city: ', len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %% [markdown]
# Visualize the number of entries for each city where we have the *city* value but not the *county*

# %%
clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].isna()) & (clean_geo_data['city'].notna())].groupby('city').count()

# %%
missing_county={'Missouri':'Saint Louis County', 'Denver':'Denver County', 'Juneau': 'Juneau County', 'San Francisco': 'San Francisco County' }

# %%
clean_geo_data[(clean_geo_data['latitude'].notna()) & (clean_geo_data['city'].isna()) & (clean_geo_data['county'].isna())]

# %%
dummy_data = clean_geo_data.loc[(clean_geo_data['latitude'].notna()) & (clean_geo_data['county'].notna()) & (clean_geo_data['city'].isna())]
print('Number of rows with null values for city, but not for lat/lon and county: ', len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %% [markdown]
# **Final evaluation**:
# We segmented the dataset into distinct groups based on the consistency we could establish among the latitude, longitude, state, county, and city fields. We also considered the address field in our analysis, but its variability and limited identifiability led us to set it aside for further use. In the end, we formulated strategies to address the missing data in these groups, considering the quality of available information.
# 
# Below, we provide a breakdown of these groups, along with their sizes:
# 
# ---------- GOOD GROUPS ----------
# * 174,796 entries: These are the fully consistent and finalized rows in the dataset.
# * 26,635 entries: Rows where only the city is missing, but it can be easily inferred from the location (k-nn).
# * 15,000 entries: Rows where only the county is missing, but it can be easily inferred from the location (k-nn).
# * 33 entries: Rows where both the city and county are missing. Even in this group, the missing information can be inferred from the location, as they are all closely clustered around Baltimore.
# 
# ---------- BAD GROUPS ----------
# * 3,116 entries: Rows where latitude, longitude, and city are missing. They can be inferred (though not perfectly) from the county-state pair.
# * 19,844 entries: Rows where only the state field is present, making it challenging to retrieve missing information.
# 
# The dataset does not contain any missing combinations beyond those mentioned.
# Out of the total of 216,464 lines, either the information is definitive or can be derived with a high degree of accuracy.

# %% [markdown]
# ### Infer Missing City Values

# %%
clean_geo_data.groupby(['state', 'county', 'city']).count()

# %% [markdown]
# Compute the centroid for each city and visualize the first 10 centroids in alphabetical order.

# %%
centroids = clean_geo_data.loc[clean_geo_data['latitude'].notna() & clean_geo_data['city'].notna()][[
    'latitude', 'longitude', 'city', 'state', 'county']].groupby(['state', 'county', 'city']).mean()
centroids.head(10)

# %%
print('Number of distinct cities:', len(centroids.index.to_list()))

# %%
centroids.sample()

# %%
info_city = pd.DataFrame(columns=['5', '15', '25', '35', '45', '55', '65', '75', '85', '95', 'tot_points', 'min', 'max', 'avg', 'centroid_lat', 'centroid_lon'], index=centroids.index)
info_city.info()

# %%
if LOAD_DATA_FROM_CHECKPOINT: # load data
    info_city = load_checkpoint('checkpoint_geo_temporary2')
else: # compute data
    for state, county, city in centroids.index:
        dummy = []
        for lat, long in zip(clean_geo_data.loc[(clean_geo_data['city'] == city) & (clean_geo_data['state'] == state) & (clean_geo_data['county'] == county) & clean_geo_data['latitude'].notna()]['latitude'], 
                            clean_geo_data.loc[(clean_geo_data['city'] == city) & (clean_geo_data['state'] == state) & (clean_geo_data['county'] == county) & clean_geo_data['longitude'].notna()]['longitude']):
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
    checkpoint(info_city, 'checkpoint_geo_temporary2') # save data 

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
       print(clean_geo_data.iloc[i][['latitude', 'longitude']])
print(clean_geo_data.iloc[i])

# %%
clean_geo_data.sample()

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

# %%
if LOAD_DATA_FROM_CHECKPOINT: # load data
    final_geo_data = load_checkpoint('checkpoint_geo_temporary3')
else: # compute data
    final_geo_data = clean_geo_data.apply(lambda row: substitute_city(row, info_city), axis=1)
    checkpoint(final_geo_data, 'checkpoint_geo_temporary3') # save data

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
plot_utils.plot_scattermap_plotly(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['county'].notna()) & (final_geo_data['city'].isna())], 'state')

# %%
plot_utils.plot_scattermap_plotly(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['state'] == 'Missouri') & (final_geo_data['county'] == 'Platte County') & (final_geo_data['city'] == 'Kansas City')], 'latitude')
len(final_geo_data.loc[(final_geo_data['latitude'].notna()) & (final_geo_data['state'] == 'Missouri') & (final_geo_data['county'] == 'Platte County') & (final_geo_data['city'] == 'Kansas City')])

# %%
#TODO: plottare le città che ha inserto e i centroidi??

# %%
final_geo_data.head(3)

# %%
geopy_data.head(3)

# %%
dummy = incidents_data.loc[incidents_data.loc[incidents_data['latitude'].isna()].index]
interesting_index = dummy.loc[dummy['latitude'].notna()].index
incidents_data.loc[interesting_index][['state', 'city_or_county', 'address', 'latitude', 'longitude']]

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
final_geo_data.loc[(final_geo_data['latitude'] == '39.7591')]

# %%
final_geo_data.loc[108203]['address']

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

# %%
checkpoint(final_geo_data, 'checkpoint_geo') # save data


