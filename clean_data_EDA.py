# %% [markdown]
# # Age and numerical attributes explorative data analysis
# Based on clean data

# %%
import pandas as pd

# %%
# read data
FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'
post_proc_path = FOLDER + 'post_proc/new_columns_incidents.csv'

incidents_data = pd.read_csv(incidents_path)
class_data = pd.read_csv(post_proc_path)

# %%
# drop duplicates rows
incidents_data.drop_duplicates(inplace=True)

# select only useful columns from incidents_data
geo_data = incidents_data[['date', 'state', 'city_or_county', 'address', 'latitude', 'longitude',
       'congressional_district', 'state_house_district', 'state_senate_district']]

# %%
# drop duplicates rows
incidents_data.drop_duplicates(inplace=True)

# select only useful columns from incidents_data
geo_data = incidents_data[['date', 'state', 'city_or_county', 'address', 'latitude', 'longitude',
       'congressional_district', 'state_house_district', 'state_senate_district']]

# %%
clean_data = pd.concat([geo_data, class_data], axis=1)

# %%
clean_data.head()

# %%
len(clean_data)

# %% [markdown]
# ## E.D.A.

# %%
clean_data.columns

# %%
# number of rows with NaN values and inconsistent values 
print('Number of raw with NaN values: ', clean_data[(clean_data['NaN_values'] == True)].shape[0])
print('Number of raw with inconsistent values: ', clean_data[(clean_data['inconsistent'] == True)].shape[0])


