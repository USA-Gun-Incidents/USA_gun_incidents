import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.offline as pyo

def compute_bss_per_cluster(X, clusters, centroids, weighted=True): # TODO: capire se è corretto chiamarla bss se relativa a ciascun cluster
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

def compute_se_per_point(X, clusters, centroids):
    '''
    This function computes the squared error for each point.

    :param X: matrix of data points
    :param clusters: cluster labels
    :param centroids: cluster centroids
    :return: squared error for each point
    '''

    return np.sum(np.square((X - centroids[clusters])), axis=(1 if X.ndim > 1 else 0))

def plot_bars_by_cluster(
        df,
        feature,
        cluster_column,
        figsize=(15, 5)
    ):
    '''
    This function plots a bar chart of the given categorical features in the given dataframe, both in the whole dataset
    and in each cluster.

    :param df: dataframe containing the data
    :param feature: categorical feature to plot
    :param cluster_column: name of the dataframe column containing the cluster labels
    :param figsize: size of the figure
    '''

    _, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2]})
    df[feature].value_counts().sort_index().plot(kind='bar', ax=axs[0], color='gray')
    axs[0].set_title(f'{feature} distribution in the whole dataset')
    axs[0].set_xlabel(feature)
    axs[0].set_ylabel('Number of incidents')
    day_xt = pd.crosstab(df[cluster_column], df[feature])
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
    plt.show()

def scatter_by_cluster(
        df,
        features,
        cluster_column,
        centroids=None,
        ncols=3,
        figsize=(35, 60),
        color_palette=sns.color_palette() # default palette assumes having max 6 cluster
    ):
    '''
    This function plots a scatter plot of each pair of the given features in the given dataframe,
    coloring the points according to the cluster they belong to.

    :param df: dataframe containing the data
    :param features: list of features to plot
    :param cluster_column: name of the dataframe column containing the cluster labels
    :param centroids: list of cluster centroids
    :param ncols: number of columns of the plot
    :param figsize: size of the figure
    :param color_palette: color palette to use
    '''
    
    ncols = ncols
    nplots = len(features)*(len(features)-1)/2
    nrows = int(nplots / ncols)
    if nplots % ncols != 0:
        nrows += 1

    colors = [color_palette[c] for c in df[cluster_column]]
    f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    id = 0
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            x, y = df[features].columns[i], df[features].columns[j]
            axs[int(id/ncols)][id%ncols].scatter(df[x], df[y], s=20, c=colors, edgecolor="k")
            if centroids is not None:
                for c in range(len(centroids)):
                    axs[int(id/ncols)][id%ncols].scatter(
                        centroids[c][df[features].columns.get_loc(x)],
                        centroids[c][df[features].columns.get_loc(y)],
                        marker='o', c='white', alpha=1, s=200, edgecolor='k')
                    axs[int(id/ncols)][id%ncols].scatter(
                        centroids[c][df[features].columns.get_loc(x)],
                        centroids[c][df[features].columns.get_loc(y)],
                        marker='$%d$' % c, c='black', alpha=1, s=50, edgecolor='k')
            axs[int(id/ncols)][id%ncols].set_xlabel(x)
            axs[int(id/ncols)][id%ncols].set_ylabel(y)
            id += 1
    for ax in axs[nrows-1, id%ncols:]:
        ax.remove()

    legend_elements = []
    clusters_ids = df['cluster'].unique()
    for c in sorted(clusters_ids):
        legend_elements.append(Line2D(
            [0], [0], marker='o', color='w', label=f'Cluster {c}', markerfacecolor=color_palette[c]))
    f.legend(handles=legend_elements, loc='lower center', ncols=len(clusters_ids))

    plt.suptitle(("Clusters in different feature spaces"), fontsize=20)
    plt.show()

def scatter_pca_features_by_cluster(
        X_pca,
        n_components,
        clusters,
        palette,
        hue_order=None,
        title=None
    ):
    '''
    This function plots a scatter plot of the first n_components of the X_pca matrix, coloring the points according to
    labels.

    :param X_pca: PCA matrix
    :param n_components: number of components to plot
    :param clusters: cluster labels
    :param palette: color palette to use
    :param hue_order: order of the hue levels
    :param title: title of the plot
    '''

    pca_data = {}
    for i in range(n_components):
        pca_data[f'Component {i+1}'] = X_pca[:,i]
    pca_data['Cluster'] = clusters
    df_pca = pd.DataFrame(data=pca_data)
    f = sns.pairplot(
        df_pca,
        hue='Cluster',
        hue_order=hue_order,
        plot_kws=dict(edgecolor="k"),
        palette=palette,
        corner=True
    )
    f.fig.suptitle(title)

def plot_boxes_by_cluster(
        df,
        features,
        cluster_column,
        ncols=3,
        figsize=(36, 36),
        title=None
    ):
    '''
    This function plots a box plot of the given features in the given dataframe, grouped by cluster.

    :param df: dataframe containing the data
    :param features: list of features to plot
    :param cluster_column: name of the dataframe column containing the cluster labels
    :param ncols: number of columns of the plot
    :param figsize: size of the figure
    '''

    nplots = len(features)
    nrows = int(nplots/ncols)
    if nplots % ncols != 0:
        nrows += 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    id = 0
    for feature in features:
        df.boxplot(column=feature, by=cluster_column, ax=axs[int(id/ncols)][id%ncols])
        id += 1
    for ax in axs[nrows-1, id%ncols:]:
        ax.remove()
    fig.suptitle(title, fontweight='bold')

def plot_violin_by_cluster(
        df,
        features,
        cluster_column,
        ncols=3,
        figsize=(36, 36),
        title=None
    ):
    '''
    This function plots a violin plot of the given features in the given dataframe, grouped by cluster.

    :param df: dataframe containing the data
    :param features: list of features to plot
    :param cluster_column: name of the dataframe column containing the cluster labels
    :param ncols: number of columns of the plot
    :param figsize: size of the figure
    '''

    nplots = len(features)
    nrows = int(nplots/ncols)
    if nplots % ncols != 0:
        nrows += 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    id = 0
    for feature in features:
        sns.violinplot(x=cluster_column, y=feature, data=df, ax=axs[int(id/ncols)][id%ncols])
        id += 1
    for ax in axs[nrows-1, id%ncols:]:
        ax.remove()
    fig.suptitle(title)

def plot_hists_by_cluster2( # TODO: colorare per cluster
        df,
        feature,
        cluster_column,
        bins=20,
        figsize=(20, 5)
    ):
    '''
    This function plots a histogram of the given feature in the given dataframe, grouped by cluster.

    :param df: dataframe containing the data
    :param feature: feature to plot
    :param cluster_column: name of the dataframe column containing the cluster labels
    :param bins: number of bins for the histogram
    :param figsize: size of the figure
    :param color_palette: color palette to use
    '''

    n_clusters = df['cluster'].unique().shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=n_clusters, figsize=figsize)
    df[feature].hist(
        by=df[cluster_column],
        bins=bins,
        ax=axes
    )
    for i, ax in enumerate(axes):
        ax.set_title(f'Cluster {i}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Number of incidents')
    fig.suptitle(f'Histogram of {feature} by cluster', fontsize=16, fontweight='bold')

def plot_clusters_size(
        clusters,
        color_palette=sns.color_palette()
    ):
    '''
    This function plots a bar chart of the number of points in each cluster.

    :param clusters: cluster labels
    :param color_palette: color palette to use
    '''

    counts = np.bincount(clusters)
    plt.bar(range(len(counts)), counts, color=color_palette)
    plt.xticks(range(len(counts)))
    plt.ylabel('Number of points')
    plt.xlabel('Cluster')
    plt.title('Number of points per cluster')

def plot_scores_per_point(score_per_point, clusters, score_name):
    '''
    This function plots the clustering score for each point, grouped by cluster.

    :param score_per_point: clustering score for each point
    :param clusters: cluster labels
    :param score_name: name of the clustering score
    '''

    n_clusters = len(np.unique(clusters))
    y_lower = 0
    for i in range(n_clusters):
        ith_cluster_sse = score_per_point[np.where(clusters == i)[0]]
        ith_cluster_sse.sort()
        size_cluster_i = ith_cluster_sse.shape[0]
        y_upper = y_lower + size_cluster_i
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sse,
            facecolor=sns.color_palette()[i],
            edgecolor=sns.color_palette()[i],
            alpha=0.7,
        )
        plt.text(-0.07, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper

    plt.axvline(x=score_per_point.mean(), color="k", linestyle="--", label='Average')
    plt.title(f"{score_name} for each point in each cluster")
    plt.xlabel(score_name)
    plt.ylabel("Cluster label")
    plt.legend(loc='best')
    plt.yticks([])

def scatter_pca_features_by_score(
    X_pca,
    clusters,
    x_component,
    y_component,
    score_per_point,
    score_name,
    cmaps=['Blues', 'Greens', 'Reds', 'PuRd', 'YIOrBr', 'GnBu'],
    figsize=(25,5)
    ):
    '''
    This function plots the clusters in the PCA space, coloring the points by the score given in input.

    :param X_pca: PCA matrix
    :param clusters: cluster labels
    :param x_component: x component of the PCA matrix to plot
    :param y_component: y component of the PCA matrix to plot
    :param score_per_point: clustering score for each point
    :param score_name: name of the score
    :param cmaps: list of colormaps to use for each cluster
    :param figsize: size of the figure
    '''
    
    n_clusters = len(np.unique(clusters))
    fig, axs = plt.subplots(nrows=1, ncols=n_clusters, figsize=figsize)
    for i in range(n_clusters):
        ith_cluster_indices = np.where(clusters == i)[0]
        plot = axs[i].scatter(
            X_pca[ith_cluster_indices,x_component],
            X_pca[ith_cluster_indices,y_component],
            c=score_per_point[ith_cluster_indices],
            cmap=cmaps[i]
        )
        cbar = fig.colorbar(plot, ax=axs[i])
        cbar.set_label(score_name)
        axs[i].set_title(f'Cluster {i}')
        axs[i].set_xlabel(f'PC {x_component}')
        axs[i].set_ylabel(f'PC {y_component}')
    fig.suptitle(f'Clusters in PCA space colored by {score_name}', fontweight='bold')

def sankey_plot(
        labels1,
        labels2,
        title=None,
        color_palette=sns.color_palette()
    ):
    '''
    This function plots a Sankey diagram of the two sets of labels passed as arguments.

    :param labels1: first list of labels
    :param labels2: second list of labels
    :param title: title of the plot
    '''

    n_clusters1 = len(set(labels1))
    n_clusters2 = len(set(labels2))

    plot_labels = []
    for i in range(n_clusters1):
        plot_labels.append(str(i))
    for i in range(n_clusters2):
        plot_labels.append(str(i))

    confusion_matrix = pd.crosstab(labels1, labels2)
    source = []
    target = []
    value = []
    for i in range(n_clusters1):
        for j in range(n_clusters2):
            if confusion_matrix.iloc[i, j] != 0:
                source.append(i)
                target.append(n_clusters1 + j)
                value.append(confusion_matrix.iloc[i, j])

    fig = go.Figure(
        data=[
            go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = "black", width = 0.5),
                    label = plot_labels,
                    color = color_palette.as_hex()[:n_clusters1] + color_palette.as_hex()[:n_clusters2]
                ),
                link = dict(
                    source = source,
                    target = target,
                    value = value
                )
            )
        ]
    )
    fig.update_layout(title_text=title, font_size=10)
    file_name = f'../html/sankey'
    if title is not None:
        camel_title = title.replace(' ', '_')
        file_name += f'_{camel_title}'
    file_name += '.html'
    pyo.plot(fig, filename=file_name, auto_open=False)
    fig.show()