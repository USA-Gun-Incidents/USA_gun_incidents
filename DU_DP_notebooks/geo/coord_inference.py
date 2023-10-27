# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as plt
import geopy.distance
import plot_utils
import data_preparation_utils

dirname = os.path.dirname(' ')
f_data = pd.read_csv(os.path.join(dirname, 'data/post_proc/incidents_date_geo.csv'), index_col=0, low_memory=False)
orig_data = pd.read_csv(os.path.join(dirname, 'data/incidents.csv'), low_memory=False).drop_duplicates()
info_city = pd.read_csv(os.path.join(dirname, 'data/post_proc/info_city.csv'), index_col=0, low_memory=False)

# %%
datas = {}

for i in orig_data.loc[f_data['latitude'].isna()].index:
    s = orig_data.loc[i]['state']
    cc = orig_data.loc[i]['city_or_county']
    datas[(s,cc)] = orig_data.loc[(orig_data['state'] == s) & (orig_data['city_or_county'] == cc)][['latitude', 'longitude']]

# %%
for i,b in datas:
    print(len(datas[i,b]))

# %%
# BEH SI PUç FARE!!!!!
