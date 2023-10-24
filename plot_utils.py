import geopandas
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.dates as mdates
import pandas as pd

# TODO: sostiture questo dizionario con:
# usa_states = pd.read_csv(
# 'https://www2.census.gov/geo/docs/reference/state.txt',
# sep='|',
# dtype={'STATE': str, 'STATE_NAME': str}
# )
# usa_name_alphcode = usa_states.set_index('STATE_NAME').to_dict()['STUSAB']
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

def plot_scattermap_plotly(data, attribute, zoom=6, height=800, width=800, title=None, legend_title=None, x_column='latitude', y_column='longitude', hover_name=True):
    if hover_name:
        fig = px.scatter_mapbox(
            hover_name=data.index,
            color=data[attribute].astype(str),
            lat=data[x_column], 
            lon=data[y_column],
            zoom=zoom, 
            height=height,
            width=width,
            title=title,
            text=data[attribute].astype(str),
            category_orders={'color': sorted(data[attribute].astype(str).unique())}
        )
    else:
        fig = px.scatter_mapbox(
            color=data[attribute].astype(str),
            lat=data[x_column], 
            lon=data[y_column],
            zoom=zoom, 
            height=height,
            width=width,
            title=title,
            text=data[attribute].astype(str),
            category_orders={'color': sorted(data[attribute].astype(str).unique())}
        )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":100,"l":0,"b":0},
        legend_title_text=legend_title
    )
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

def plot_clf_decision_boundary(clf, X_train, y_train, color_map, title=None):
    colors = []
    for c in y_train.astype(int):
        colors.append(color_map[c])

    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_train,
        response_method="predict",
        colors="lime",
        plot_method="contour" # ‘contourf’, ‘pcolormesh’
    )
    
    disp.ax_.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=colors,
        edgecolor="k"
    )
    plt.axis('scaled')
    plt.axis('scaled')
    plt.title(title)
    plt.yticks([])
    plt.xticks([])
    plt.show()

def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = mdates.num2date(bp['whiskers'][i*2].get_ydata()[1])
        dict1['lower_quartile'] = mdates.num2date(bp['boxes'][i].get_ydata()[1])
        dict1['median'] = mdates.num2date(bp['medians'][i].get_ydata()[1])
        dict1['upper_quartile'] = mdates.num2date(bp['boxes'][i].get_ydata()[2])
        dict1['upper_whisker'] = mdates.num2date(bp['whiskers'][(i*2)+1].get_ydata()[1])
        
        dict1['fliers'] = len(bp['fliers'][i].get_ydata())
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)