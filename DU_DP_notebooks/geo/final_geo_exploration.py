# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as plt
import geopy.distance
import plot_utils
import utils 

dirname = os.path.dirname(' ')
f_data = pd.read_csv(os.path.join(dirname, 'data/post_proc/final_incidents_city_inf.csv'), index_col=0, low_memory=False)
orig_data = pd.read_csv(os.path.join(dirname, 'data/incidents.csv'), low_memory=False).drop_duplicates()
geopy_data = pd.read_csv(os.path.join(dirname, 'data/geopy/geopy.csv'), index_col=0, low_memory=False)

# %%
f_data

# %%
orig_data.head(3)

# %%
geopy_data.head(3)

# %%
dummy = orig_data.loc[f_data.loc[f_data['latitude'].isna()].index]
interesting_index = dummy.loc[dummy['latitude'].notna()].index
orig_data.loc[interesting_index][['state', 'city_or_county', 'address', 'latitude', 'longitude']]

# %%
import random
new_ii = random.sample(interesting_index.to_list(), 15)

# %%
orig_data.loc[new_ii][['state', 'city_or_county', 'address', 'latitude', 'longitude']]

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
orig_data.loc[(orig_data['latitude'] == '39.7591')]

# %%
utils.check_address(orig_data.loc[239662]['address'], geopy_data.loc[239662]['display_name'])

# %%
orig_data.loc[108203]['address']

# %%
geopy_data.loc[108203]

# %%
dummy_data = f_data.loc[(f_data['latitude'].notna()) & (f_data['county'].isna())]
print(len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %%
dummy_data = f_data.loc[(f_data['latitude'].notna()) & (f_data['city'].isna())]
print(len(dummy_data))
plot_utils.plot_scattermap_plotly(dummy_data, 'state')

# %%
date_data = pd.read_csv(os.path.join(dirname, 'data/checkpoints/checkpoint_date.csv'), index_col=0, low_memory=False)
to_save = pd.concat([date_data, f_data], join="inner", axis=1)

# %%
to_save

# %%
FOLDER = os.path.join(dirname, 'data')
#to_save.to_csv(os.path.join(FOLDER,'post_proc/incidents_data_geo.csv'))


