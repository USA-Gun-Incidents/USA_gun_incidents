# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as plt
import geopy.distance
import plot_utils

dirname = os.path.dirname(' ')
data = pd.read_csv(os.path.join(dirname, 'data/post_proc/final_incidents.csv'), index_col=0, dtype={'latitude':float, 'logitude':float})
