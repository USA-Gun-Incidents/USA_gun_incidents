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

for city in data['city'].unique():
    city_data_struct[city] = {'centroid':0, 'count':0, 'max_dist':-math.inf, 'min_dist':math.inf}
    for i, row in data.loc[['city' == city]].iterrows():
        city_data_struct[]

