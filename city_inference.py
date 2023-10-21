# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import math

# %%
dirname = os.path.dirname(' ')
data = pd.read_csv(os.path.join(dirname, 'data/post_proc/final_incidents_KNN.csv'))

# %%
city_data_struct = {}
print(data.loc[data['city'] == 'Detroit']['latitude'].mean())


for city in data['city'].unique():
    city_data_struct[city] = {}
    city_data_struct[city]['centroid_latitude'] = data.loc[data['city'] == city]['latitude'].mean()
    city_data_struct[city]['centroid_longitude'] = data.loc[data['city'] == city]['longitude'].mean()
    city_data_struct[city]['distance_array'] = []
    for lat, long in zip(data.loc[data['city'] == city]['latitude'], data.loc[data['city'] == city]['longitude']):
        city_data_struct[city]['distance_array'].append(math.dist([lat, long], city_data_struct[city]['centroid_latitude'], city_data_struct[city]['centroid_longitude']))
    city_data_struct[city]['distance_array'] = sorted(city_data_struct[city]['distance_array'])

print(city_data_struct)


