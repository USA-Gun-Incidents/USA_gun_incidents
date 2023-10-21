import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data/post_proc/final_incidents.csv'))

X_train = np.concatenate((
    data[
        (data['latitude'].notna()) &
        (data['longitude'].notna()) & 
        (data['county'].notna())
        ]['longitude'].values.reshape(-1, 1),
    data[
        (data['latitude'].notna()) &
        (data['longitude'].notna()) & 
        (data['county'].notna())
        ]['latitude'].values.reshape(-1, 1)),
    axis=1
)
X_test = np.concatenate((
    data[
        (data['county'].isna()) & 
        (data['latitude'].notna()) & 
        (data['longitude'].notna())
        ]['longitude'].values.reshape(-1, 1),
    data[
        (data['county'].isna()) &
        (data['latitude'].notna()) & 
        (data['longitude'].notna())
        ]['latitude'].values.reshape(-1, 1)),
    axis=1
)
y_train = data[
    (data['county'].notna()) & 
    (data['latitude'].notna()) & 
    (data['longitude'].notna())
]['county'].values

K = 3
knn_clf = KNeighborsClassifier(n_neighbors=K)
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)

data['KNN_county'] = data['county']
data.loc[
    (data['county'].isna()) &
    (data['latitude'].notna()) & 
    (data['longitude'].notna()),
    'KNN_county'
] = knn_pred

data = pd.read_csv(os.path.join(dirname, 'data/post_proc/final_incidents_KNN.csv'))