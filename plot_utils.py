import geopandas
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.inspection import DecisionBoundaryDisplay

usa_code = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

def plot_usa_map(
    df,
    col_to_plot,
    ax,
    state_col='state',
    vmin=None,
    vmax=None,
    title=None,
    cbar_title=None,
    cmap='coolwarm',
    borders_path="./cb_2018_us_state_500k"
    ):
    geo_usa = geopandas.read_file(borders_path)
    geo_merge=geo_usa.merge(df, left_on='NAME', right_on=state_col)
    
    #fig, continental_ax = plt.subplots(figsize=(20, 10))
    alaska_ax = ax.inset_axes([-128,22,16,8], transform=ax.transData)
    hawaii_ax = ax.inset_axes([-110,22.8,8,5], transform=ax.transData)

    ax.set_xlim(-130, -64)
    ax.set_ylim(22, 53)

    alaska_ax.set_ylim(51, 72)
    alaska_ax.set_xlim(-180, -127)

    hawaii_ax.set_ylim(18.8, 22.5)
    hawaii_ax.set_xlim(-160, -154.6)

    if vmin==None or vmax==None:
        vmin, vmax = df[col_to_plot].agg(['min', 'max']) # share the same colorbar
    geo_merge.plot(column=col_to_plot, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                   legend=True, legend_kwds={"label": cbar_title, "shrink": 0.5})
    geo_merge.plot(column=col_to_plot, ax=alaska_ax, vmin=vmin, vmax=vmax, cmap=cmap)
    geo_merge.plot(column=col_to_plot, ax=hawaii_ax, vmin=vmin, vmax=vmax, cmap=cmap)

    for _, row in geo_merge.iterrows():
        x = row['geometry'].centroid.coords[0][0]
        y = row['geometry'].centroid.coords[0][1]
        x_displacement = 0
        y_displacement = 0
        xytext = None
        arrows = None
        if row['NAME']=="Alaska":
            x = -150
            y = 65
            xytext=(x,y)
        elif row['NAME']=="Hawaii":
            x = -157
            y = 20.5
            xytext=(x,y)
        elif row['NAME']=="Maryland":
            xytext = (x+4.5, y+0.5)
            arrows = dict(arrowstyle="-")
        elif row['NAME']=="District of Columbia":
            xytext = (x+4.5, y-1)
            arrows = dict(arrowstyle="-")
        elif row['NAME']=="Delaware":
            xytext =  (x+4.5, y+0.05)
            arrows = dict(arrowstyle="-")
        elif row['NAME']=="Rhode Island":
            xytext =  (x+5, y-0.1)
            arrows = dict(arrowstyle="-")
        elif row['NAME']=="Connecticut":
            xytext =  (x+4, y-1.5)
            arrows = dict(arrowstyle="-")
        elif row['NAME'] in ['Mississippi', 'West Virginia', 'New Hampshire']:
            y_displacement = -0.35

        alaska_ax.annotate(
            text=row['NAME'],
            xy=(x+x_displacement, y+y_displacement),
            xytext=xytext,
            arrowprops=arrows,
            ha='center',
            fontsize=8
        )
        hawaii_ax.annotate(
            text=row['NAME'],
            xy=(x+x_displacement, y+y_displacement),
            xytext=xytext,
            arrowprops=arrows,
            ha='center',
            fontsize=8
        )
        ax.annotate(
            text=row['NAME'],
            xy=(x+x_displacement, y+y_displacement),
            xytext=xytext,
            arrowprops=arrows,
            ha='center',
            fontsize=8
        )
    
    ax.set_title(title, fontsize=16)

    for ax in [ax, alaska_ax, hawaii_ax]:
        ax.set_yticks([])
        ax.set_xticks([])
    
    #return ax
    #fig.show()
    #fig.tight_layout()


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