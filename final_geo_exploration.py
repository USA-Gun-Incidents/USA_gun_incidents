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
f_data = pd.read_csv(os.path.join(dirname, 'data/post_proc/final_incidents.csv'), index_col=0, low_memory=False)
orig_data = pd.read_csv(os.path.join(dirname, 'data/incidents.csv'), low_memory=False).drop_duplicates()
geopy_data = pd.read_csv(os.path.join(dirname, 'data/geopy/geopy.csv'), index_col=0, low_memory=False)

# %%
f_data

# %%
orig_data

# %%
geopy_data

# %%
dummy = orig_data.loc[f_data.loc[f_data['latitude'].isna()].index]
interesting_index = dummy.loc[dummy['latitude'].notna()].index

orig_data.loc[interesting_index][['state', 'city_or_county', 'address', 'latitude', 'longitude']]

# %%
geopy_data.loc[interesting_index][['class', 'type', 'addresstype', 'display_name', 'state', 'county', 'city', 'town']]

# %%
for i in geopy_data.loc[239591].index:
    print(i, geopy_data.loc[239591][i])

#TODO: SUBURB!!!!!


# %%
orig_data.loc[(orig_data['latitude'] == '39.7591')]

# %%
utils.check_address(orig_data.loc[239662]['address'], geopy_data.loc[239662]['display_name'])

# %%
orig_data.loc[239662]['address']

# %%
geopy_data.loc[239662]['display_name']

# %%
sub = {'Kane':'McKean County' 
       }

#TODO: City of in wikipedia
#TODO: 1 0 1....


