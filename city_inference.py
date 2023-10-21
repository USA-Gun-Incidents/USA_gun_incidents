import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data/post_proc/final_incidents_KNN.csv'))