import geopandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.offline as pyo

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
    black_nan=True,
    size=None,
    color_sequence=None,
    showlegend=True
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
        color_discrete_sequence=color_sequence,
        color_discrete_map=color_map,
        lat=data[x_column], 
        lon=data[y_column],
        zoom=zoom, 
        height=height,
        width=width,
        title=title,
        text=data[attribute].astype(str) if attribute is not None else None,
        category_orders={'color': sorted(data[attribute].astype(str).unique())} if attribute is not None else None,
        size= data[size] if size is not None else None

    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":100,"l":0,"b":0},
        showlegend=showlegend,
        legend_title_text=legend_title
    )
    #fig.show()
    return fig

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

def plot_not_nan_entries_by_state(df, attribute_list, labels, n_rows=2, n_columns=4):
    rows = n_rows
    cols = n_columns
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'choropleth'} for c in range(cols)] for r in range(rows)],
        subplot_titles=labels,
        vertical_spacing=0.3,
        horizontal_spacing=0.01,
    )

    for i, attribute in enumerate(attribute_list):
        frame = px.choropleth(
            df,
            color=attribute,
            locations='px_code',
            locationmode="USA-states",
            hover_name='state',
            hover_data={
                'px_code': False,
                'not_nan_entries': True,
            },
        )

        choropleth_trace = frame['data'][0]
        fig.add_trace(choropleth_trace, 
            row=(i)//cols+1, 
            col=(i) % cols+1
        )
        fig.update_layout(
            title_text="Ratio of NaN entries by state for different attributes",
            showlegend=False,
        )
        fig.update_geos( 
            scope = 'usa',
            visible=False)

    fig.show()

def plot_missing_values_for_state(df, attribute):
    
    fig, ax = plt.subplots(figsize=(20, 2))
    ax.bar(df.groupby('state')['state'].count().index, df.groupby('state')['state'].count().values, 
        label='#Total', edgecolor='black', linewidth=0.8, alpha=0.5)
    ax.bar(df[df[attribute].isna()].groupby('state')['state'].count().index, df[df[attribute].isna()
        ].groupby('state')['state'].count().values, label=f'#Missing {attribute}', edgecolor='black', linewidth=0.8)
    ax.set_xlabel('State')
    ax.set_yscale('log')
    ax.set_ylabel('Number of incidents')
    ax.legend()
    ax.set_title(f'Percentage of missing values for {attribute} values by state')
    ax.xaxis.set_tick_params(rotation=90)
    for state in df['state'].unique():
        try:
            plt.text(
                x=state, 
                y=df[df[attribute].isna()].groupby('state')['state'].count()[state], 
                s=str(round(100*df[df[attribute].isna()].groupby('state')['state'].count()[state] / 
                df.groupby('state')['state'].count()[state]))+'%', 
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=8)
        except:
            pass
    plt.show()

def discrete_attribute_distribuition_plot(df, attribute, state):    
    plt.figure(figsize=(20, 2))
    plt.bar(df.groupby(attribute)[attribute].count().index,
        df.groupby(attribute)[attribute].count().values, 
        label='Whole dataset', edgecolor='black', linewidth=0.8, alpha=0.5)
    plt.bar(df[df['state']==state].groupby(attribute)[attribute].count().index, 
        df[df['state']==state].groupby(attribute)[attribute].count().values, 
        label=state, edgecolor='black', linewidth=0.8, alpha=0.8)
    plt.xlabel(f'Number of {attribute}')
    plt.ylabel('Number of incidents')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Number of {attribute} per incident')
    plt.show()

def continuous_attribute_distribuition_plot(df, attribute, state):
    plt.figure(figsize=(20, 2))
    plt.hist(df[attribute], bins=100, label='Whole dataset', edgecolor='black', linewidth=0.8, alpha=0.5)
    plt.hist(df[df['state']==state][attribute], bins=100, label=state, edgecolor='black', linewidth=0.8, alpha=0.8)
    plt.xlabel(f'{attribute}')
    plt.ylabel('Number of incidents')
    plt.legend()
    plt.yscale('log')
    plt.title(f'{attribute} distribuition')
    plt.show()

def scatter_pairs(df, features, figsize=(35,100), ncols=3):
    '''
    
    '''
    ncols = ncols
    nplots = len(features)*(len(features)-1)/2
    nrows = int(nplots / ncols)
    if nplots % ncols != 0:
        nrows += 1

    f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    id = 0
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            x, y = df[features].columns[i], df[features].columns[j]
            axs[int(id/ncols)][id%ncols].scatter(df[x], df[y], s=20, edgecolor="k")
            axs[int(id/ncols)][id%ncols].set_xlabel(x)
            axs[int(id/ncols)][id%ncols].set_ylabel(y)
            id += 1
    if nrows > 1:
        for ax in axs[nrows-1, id%ncols:]:
            ax.remove()
    plt.tight_layout()

def sankey_plot(
        labels,
        labels_titles=None,
        title=None,
        color_palette=sns.color_palette()
    ):
    '''
    This function plots a Sankey diagram of the sets of labels passed as arguments.

    :param labels1: list of labels list
    :param labels2: lables titles
    :param title: title of the plot
    :param color_palette: color palette to use
    '''

    n_clusters = [len(set(label_list)) for label_list in labels]

    plot_labels = []
    for i in range(len(labels)):
        for j in range(n_clusters[i]):
            plot_labels.append(str(j))

    source = []
    target = []
    value = []
    for i in range(len(labels)-1):
        confusion_matrix = pd.crosstab(labels[i], labels[i+1])
        curr_source = []
        curr_target = []
        curr_value = []

        source_add = 0
        for j in range(0, i):
            source_add += n_clusters[j]
        target_add = source_add + n_clusters[i]

        for j in range(n_clusters[i]):
            for k in range(n_clusters[i+1]):
                if confusion_matrix.iloc[j, k] != 0:
                    curr_source.append(j+source_add)
                    curr_target.append(k+target_add)
                    curr_value.append(confusion_matrix.iloc[j, k])

        source += curr_source
        target += curr_target
        value += curr_value

    colors = []
    for i in range(len(labels)):
        colors += color_palette.as_hex()[:n_clusters[i]]

    fig = go.Figure(
        data=[
            go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = "black", width = 0.5),
                    label = plot_labels,
                    color = colors
                ),
                link = dict(
                    source = source,
                    target = target,
                    value = value
                )
            )
        ]
    )

    for x_coordinate, column_name in enumerate(labels_titles):
        fig.add_annotation(
            x=x_coordinate,
            y=1.05,
            xref="x",
            yref="paper",
            text=column_name,
            showarrow=False
        )
    fig.update_layout(
        title_text=title, 
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        plot_bgcolor='rgba(0,0,0,0)',
        font_size=10
    )

    file_name = f'../html/sankey'
    if title is not None:
        camel_title = title.replace(' ', '_')
        file_name += f'_{camel_title}'
    file_name += '.html'
    pyo.plot(fig, filename=file_name, auto_open=False)
    fig.show()