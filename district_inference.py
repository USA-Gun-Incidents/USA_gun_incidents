# %%
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import NearestCentroid

# %%
incidents_data = pd.read_csv('./data/incidents.csv')
incidents_data.drop_duplicates()
incidents_data['state'] = incidents_data['state'].str.upper()

# %%
def map_plotly_plot(incidents_data, attribute):
    fig = px.scatter_mapbox(
        color=incidents_data[attribute].astype(str),
        lat=incidents_data['latitude'], 
        lon=incidents_data['longitude'],
        zoom=6, 
        height=800,
        width=800,
        text=incidents_data[attribute].astype(str)
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def map_matplotlib_plot(incidents_data, color_map, colors):
    plt.scatter(
        y=incidents_data['latitude'],
        x=incidents_data['longitude'],
        c=colors
    )
    plt.axis('scaled')
    for key, value in color_map.items():
        plt.scatter([], [], c=value, label=key)
    plt.legend(bbox_to_anchor=(-0.5, 1), loc='upper left')
    plt.yticks([])
    plt.xticks([])

def build_X_y_for_district_inference(incidents_data):
    X_train = np.concatenate((
        incidents_data[
            (incidents_data['congressional_district'].notna()) &
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna())
            ]['longitude'].values.reshape(-1, 1),
        incidents_data[
            (incidents_data['congressional_district'].notna()) & 
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna())
            ]['latitude'].values.reshape(-1, 1)),
        axis=1
    )
    X_test = np.concatenate((
        incidents_data[
            (incidents_data['congressional_district'].isna()) & 
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna())
            ]['longitude'].values.reshape(-1, 1),
        incidents_data[
            (incidents_data['congressional_district'].isna()) &
            (incidents_data['latitude'].notna()) & 
            (incidents_data['longitude'].notna())
            ]['latitude'].values.reshape(-1, 1)),
        axis=1
    )
    y_train = incidents_data[
        (incidents_data['congressional_district'].notna()) & 
        (incidents_data['latitude'].notna()) & 
        (incidents_data['longitude'].notna())
        ]['congressional_district'].values
    return X_train, X_test, y_train

def plot_clf_decision_boundary(clf, X_train, y_train, color_map):
    colors = []
    for c in y_train.astype(int):
        colors.append(color_map[c])

    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_train,
        response_method="predict",
        colors="lime",
        plot_method="contour" # "pcolormesh", "pcolormesh"
    )

    disp.ax_.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=colors,
        edgecolor="k"
    )
    plt.axis('scaled')
    plt.yticks([])
    plt.xticks([])
    plt.show()

# %% [markdown]
# ![image](https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif/lossless-page1-1256px-United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif.png)

# %%
alabama_color_map = {
    1:'red',
    2:'orange',
    3:'yellow',
    4:'green',
    5:'lightblue',
    6:'blue',
    7:'purple'
}

colors = []
for c in incidents_data[(incidents_data['state']=="ALABAMA")]['congressional_district']:
    if math.isnan(c): 
        colors.append('black')
    else:
        colors.append(alabama_color_map[int(c)])

# %%
map_plotly_plot(incidents_data[incidents_data['state']=='ALABAMA'], 'congressional_district')

# %%
map_matplotlib_plot(incidents_data[incidents_data['state']=='ALABAMA'], alabama_color_map, colors)

# %%
X_train, X_test, y_train = build_X_y_for_district_inference(incidents_data[incidents_data['state']=="ALABAMA"])

# %%
K = 3
knn_clf = KNeighborsClassifier(n_neighbors=K)
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)

incidents_data['KNN_congressional_district'] = incidents_data['congressional_district']
incidents_data.loc[
    (incidents_data['state']=="ALABAMA") &
    (incidents_data['congressional_district'].isna()) &
    (incidents_data['latitude'].notna()) & 
    (incidents_data['longitude'].notna()),
    'KNN_congressional_district'
    ] = knn_pred

# %%
map_plotly_plot(incidents_data[incidents_data['state']=='ALABAMA'], 'KNN_congressional_district')

# %%
plot_clf_decision_boundary(knn_clf, X_train, y_train, alabama_color_map)

# %%
nc_clf = NearestCentroid()
nc_clf.fit(X_train, y_train)
nc_pred = nc_clf.predict(X_test)

incidents_data['NC_congressional_district'] = incidents_data['congressional_district']
incidents_data.loc[
    (incidents_data['state']=="ALABAMA") &
    (incidents_data['congressional_district'].isna()) &
    (incidents_data['latitude'].notna()) & 
    (incidents_data['longitude'].notna()),
    'NC_congressional_district'
    ] = nc_pred

# %%
map_plotly_plot(incidents_data[incidents_data['state']=='ALABAMA'], 'NC_congressional_district')

# %%
plot_clf_decision_boundary(nc_clf, X_train, y_train, alabama_color_map)


