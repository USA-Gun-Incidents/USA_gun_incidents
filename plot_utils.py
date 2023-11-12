import geopandas
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.dates as mdates
import pandas as pd

def hist_box_plot(
    df,
    col,
    title,
    xlabel=None,
    ylabel=None,
    bins=50,
    figsize=(10, 5),
    kde=True,
    bw_method='scott'
    ):
    '''
    This function plots an histogram and a boxplot of the given column of the given dataframe.
    
    :param df: dataframe
    :param col: column to plot
    :param title: title of the plot
    :param xlabel: label of the x axis
    :param ylabel: label of the y axis
    :param bins: number of bins for the histogram
    :param figsize: size of the figure
    :param kde: toggle to the overlapping kde rapresentation
    :param bw_method: kde parameter
    :param bw_adjust: kde parameter
    '''
    _, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    df[col].plot.hist(bins=bins, figsize=figsize, ax=ax_hist)
    if kde: df[col].plot.kde(bw_method=bw_method, ax=ax_hist, secondary_y=True)
    df.boxplot(ax=ax_box, column=col, vert=False, grid=False)  
    ax_box.set(yticks=[])
    plt.suptitle(title +' (#NotNanVal/#TotVal: ' + str(len(df[col].dropna())) + '/' + str(len(df[col])) + ')')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

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
    '''
    This function plots a map of the USA colouring states according to the value of the given column of the given dataframe.

    :param df: dataframe
    :param col_to_plot: column to plot
    :param ax: axis where to plot the map
    :param state_col: name of the column containing the states
    :param vmin: minimum value for the colorbar
    :param vmax: maximum value for the colorbar
    :param title: title of the plot
    :param cbar_title: title of the colorbar
    :param cmap: colormap to use
    :param borders_path: path to the shapefile containing the borders of the states
    '''
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

def plot_scattermap_plotly(
    data,
    attribute=None,
    zoom=6,
    height=800,
    width=800,
    title=None,
    legend_title=None,
    x_column='latitude',
    y_column='longitude',
    hover_name=True,
    black_nan=True
    ):
    '''
    This function plots a scattermap using plotly.

    :param data: dataframe
    :param attribute: attribute to use for coloring the points
    :param zoom: zoom of the map
    :param height: height of the map
    :param width: width of the map
    :param title: title of the map
    :param legend_title: title of the legend
    :param x_column: name of the column containing the x coordinates
    :param y_column: name of the column containing the y coordinates
    :param hover_name: if True, the hover name is the index of the dataframe # TODO: ?
    :param black_nan: if True, the nan values are colored in black
    '''
    color_map = None
    if black_nan and attribute is not None:
        color_map = {}
        n_colors = len(px.colors.qualitative.Plotly)
        for i, category in enumerate(data[attribute].unique()):
            color_map[str(category)] = px.colors.qualitative.Plotly[i%n_colors]
        color_map[str(np.nan)] = '#000000'

    fig = px.scatter_mapbox(
        hover_name=data.index if hover_name else None,
        color=data[attribute].astype(str) if attribute is not None else None,
        color_discrete_map=color_map,
        lat=data[x_column], 
        lon=data[y_column],
        zoom=zoom, 
        height=height,
        width=width,
        title=title,
        text=data[attribute].astype(str) if attribute is not None else None,
        category_orders={'color': sorted(data[attribute].astype(str).unique())} if attribute is not None else None
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":100,"l":0,"b":0},
        legend_title_text=legend_title
    )
    fig.show()

def plot_clf_decision_boundary(clf, X_train, y_train, color_map, title=None):
    '''
    This function plots the decision boundary of the given classifier.

    :param clf: classifier
    :param X_train: training set
    :param y_train: labels of the training set
    :param color_map: dictionary mapping each class to a color
    :param title: title of the plot
    '''
    colors = []
    for c in y_train.astype(int):
        colors.append(color_map[c])

    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_train,
        response_method="predict",
        colors="lime",
        plot_method="contour"
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

"""def get_box_plot_data(labels, bp): # TODO: si puà rimuovere?
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

    return pd.DataFrame(rows_list)"""