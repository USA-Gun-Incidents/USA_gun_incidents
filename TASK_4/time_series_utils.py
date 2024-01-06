import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

def create_cities_df(incidents_df):
    # create a dataframe with as index city and state
    cities_df = incidents_df.groupby(['city', 'state'])['population_state_2010'].mean() # population_state_2010
    cities_df = pd.DataFrame(cities_df)

    # quantile of population_state_2010
    cities_df['population_quantile'] = pd.qcut(cities_df['population_state_2010'], 4, labels=False)

    # n_incidents
    cities_df['n_incidents_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['week'].count()
    cities_df['n_incidents_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['week'].count()
    cities_df['n_incidents_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['week'].count()
    cities_df['n_incidents_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['week'].count()
    cities_df['n_incidents'] = incidents_df.groupby(['city', 'state'])['week'].count()
    cities_df['n_incidents_quantile'] = pd.qcut(cities_df['n_incidents'], 4, labels=False)

    # n_weeks_with_incidents
    cities_df['n_weeks_with_incidents_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['week'].nunique()
    cities_df['n_weeks_with_incidents_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['week'].nunique()
    cities_df['n_weeks_with_incidents_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['week'].nunique()
    cities_df['n_weeks_with_incidents_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['week'].nunique()
    cities_df['n_weeks_with_incidents'] = incidents_df.groupby(['city', 'state'])['week'].nunique()
    cities_df['n_weeks_quantile'] = pd.qcut(cities_df['n_weeks_with_incidents'], 4, labels=False)

    # n_participants
    cities_df['n_participants_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['n_participants'].sum()
    cities_df['n_participants_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['n_participants'].sum()
    cities_df['n_participants_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['n_participants'].sum()
    cities_df['n_participants_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['n_participants'].sum()
    cities_df['n_participants'] = incidents_df.groupby(['city', 'state'])['n_participants'].sum()
    cities_df['n_participants_quantile'] = pd.qcut(cities_df['n_participants'], 4, labels=False)

    # n_participants_avg
    cities_df['n_participants_avg_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['n_participants'].mean()
    cities_df['n_participants_avg_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['n_participants'].mean()
    cities_df['n_participants_avg_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['n_participants'].mean()
    cities_df['n_participants_avg_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['n_participants'].mean()
    cities_df['n_participants_avg'] = incidents_df.groupby(['city', 'state'])['n_participants'].mean()

    # n_killed
    cities_df['n_killed_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['n_killed'].sum()
    cities_df['n_killed_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['n_killed'].sum()
    cities_df['n_killed_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['n_killed'].sum()
    cities_df['n_killed_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['n_killed'].sum()
    cities_df['n_killed'] = incidents_df.groupby(['city', 'state'])['n_killed'].sum()
    cities_df['n_killed_quantile'] = pd.qcut(cities_df['n_killed'], 4, labels=False)

    # n_killed_avg
    cities_df['n_killed_avg_2014'] = incidents_df[incidents_df['year']==2014].groupby(['city', 'state'])['n_killed'].mean()
    cities_df['n_killed_avg_2015'] = incidents_df[incidents_df['year']==2015].groupby(['city', 'state'])['n_killed'].mean()
    cities_df['n_killed_avg_2016'] = incidents_df[incidents_df['year']==2016].groupby(['city', 'state'])['n_killed'].mean()
    cities_df['n_killed_avg_2017'] = incidents_df[incidents_df['year']==2017].groupby(['city', 'state'])['n_killed'].mean()
    cities_df['n_killed_avg'] = incidents_df.groupby(['city', 'state'])['n_killed'].mean()

    # n_females, n_males, n_injured, n_arrested
    cities_df['n_females'] = incidents_df.groupby(['city', 'state'])['n_females'].sum()
    cities_df['n_females_quantile'] = pd.qcut(cities_df['n_females'], 4, labels=False)
    cities_df['n_males'] = incidents_df.groupby(['city', 'state'])['n_males'].sum()
    cities_df['n_males_quantile'] = pd.qcut(cities_df['n_males'], 4, labels=False)
    cities_df['n_injured'] = incidents_df.groupby(['city', 'state'])['n_injured'].sum()
    cities_df['n_injured_quantile'] = pd.qcut(cities_df['n_injured'], 4, labels=False)
    cities_df['n_arrested'] = incidents_df.groupby(['city', 'state'])['n_arrested'].sum()
    cities_df['n_arrested_quantile'] = pd.qcut(cities_df['n_arrested'], 4, labels=False)

    # fatal_incidents
    cities_df['fatal_incidents'] = incidents_df[incidents_df['n_killed'] > 0].groupby(['city', 'state'])['week'].count()
    cities_df['fatal_incidents_quantile'] = pd.qcut(cities_df['fatal_incidents'], 4, labels=False)
    cities_df['fatal_incidents_ratio'] = cities_df['fatal_incidents']/cities_df['n_incidents']

    return cities_df

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

def plot_timeseries_per_cluster(X, labels):
    n_clusters = len(np.unique(labels))
    fig, ax = plt.subplots(math.ceil(n_clusters/2), 2, figsize=(20, 20))
    print(np.shape(ax))
    colors = plt.rcParams["axes.prop_cycle"]()

    max_num_samples_per_cluster = 5
    for c in range(n_clusters):
        ax_c = ax[c//2, c%2]
        ax_c.set_title(f'Cluster {c}', fontsize=8)
        ax_c.set_axisbelow(True)
        for i, idx in enumerate(np.where(labels == c)[0]):
            if i >= max_num_samples_per_cluster:
                break
            ax_c.plot(X[idx], '.--', color=next(colors)['color'])
    fig.tight_layout();

def plot_bars_by_cluster(
        df,
        feature,
        cluster_column,
        figsize=(15, 5),
        log_scale=False
    ):
    '''
    This function plots a bar chart of the given categorical features in the given dataframe, both in the whole dataset
    and in each cluster.

    :param df: dataframe containing the data
    :param feature: categorical feature to plot
    :param cluster_column: name of the dataframe column containing the cluster labels
    :param figsize: size of the figure
    '''

    _, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 2, 1]}, sharey=True)
    df[feature].value_counts().sort_index().plot(kind='bar', ax=axs[0], color=sns.color_palette('hls').as_hex())
    axs[0].set_title(f'{feature} distribution in the whole dataset')
    axs[0].set_xlabel(feature)
    axs[0].set_ylabel('Number of incidents')
    day_xt = pd.crosstab(cluster_column, df[feature])
    day_xt.plot(
        kind='bar',
        stacked=False,
        figsize=(15, 7),
        ax=axs[1],
        color=sns.color_palette('hls').as_hex()
        )
    axs[1].set_title(f'{feature} distribution in each cluster')
    axs[1].set_xlabel('Cluster')
    axs[1].set_ylabel('Number of incidents')
    
    plot_clusters_size(cluster_column, ax=axs[2], title='Clusters size', color_palette=sns.color_palette('tab10'))
    if log_scale:
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')
    plt.show()

def plot_clusters_size(
        clusters,
        ax,
        color_palette=sns.color_palette(),
        title=None
    ):
    '''
    This function plots a bar chart of the number of points in each cluster.

    :param clusters: cluster labels
    :param color_palette: color palette to use
    '''

    counts = np.bincount(clusters)
    ax.bar(range(len(counts)), counts, color=color_palette)
    ax.set_xticks(range(len(counts)))
    ax.set_ylabel('Number of points')
    ax.set_xlabel('Cluster')
    ax.set_title(title)

def kmeans_evaluation(X, kmeans, cluster_centers):
    results = {}
    results['model'] = kmeans
    results['SSE'] = kmeans.inertia_
    results['BSS'] = compute_bss_per_cluster(X=X, clusters=kmeans.labels_, centroids=cluster_centers, weighted=True).sum()
    results['davies_bouldin_score'] = davies_bouldin_score(X=X, labels=kmeans.labels_)
    results['calinski_harabasz_score'] = calinski_harabasz_score(X=X, labels=kmeans.labels_)
    results['silhouette_score'] = silhouette_score(X=X, labels=kmeans.labels_) 
    results['n_iter'] = kmeans.n_iter_
    return results

def hierarchical_evaluation(X, hierarchical):
    results = {}
    results['model'] = hierarchical
    results['SSE'] = np.nan
    results['BSS'] = np.nan
    results['davies_bouldin_score'] = np.nan
    results['calinski_harabasz_score'] = np.nan
    results['silhouette_score'] = silhouette_score(X=X, labels=hierarchical.labels_) 
    results['n_iter'] = np.nan
    return results

def compute_bss_per_cluster(X, clusters, centroids, weighted=True):
    '''
    This function computes the between-cluster sum of squares for each cluster.

    :param X: matrix of data points
    :param clusters: cluster labels
    :param centroids: cluster centroids
    :param weighted: if True, the between-cluster sum of squares is weighted by the number of points in each cluster
    :return: between-cluster sum of squares for each cluster
    '''

    centroid = X.mean(axis=0)
    sizes = np.ones(centroids.shape[0])
    if weighted:
        sizes = np.bincount(clusters)
    return np.sum(np.square((centroids - centroid)), axis=1)*sizes