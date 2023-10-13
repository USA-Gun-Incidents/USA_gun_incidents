import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.inspection import DecisionBoundaryDisplay


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