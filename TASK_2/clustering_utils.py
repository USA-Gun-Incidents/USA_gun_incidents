import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def bss(X, clusters, centroids): # TODO: è specifica per centroid based (?)
    '''
    This function computes the between-cluster sum of squares.

    :param X: matrix of data points
    :param clusters: cluster labels
    :param centroids: cluster centroids
    :return: between-cluster sum of squares
    '''

    centroid = X.mean(axis=0)
    sizes = np.bincount(clusters)
    return np.sum(np.sum(np.square((centroids - centroid)), axis=1)*sizes)

def sse_per_point(X, clusters, centroids): # TODO: è specifica per centroid based (?)
    '''
    This function computes the sum of squares error for each point.

    :param X: matrix of data points
    :param clusters: cluster labels
    :param centroids: cluster centroids
    :return: sum of squares error for each point
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
            axs[int(id/ncols)][id%ncols].scatter(df[x], df[y], s=20, c=colors)
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
        figsize=(36, 36)
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
    _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    id = 0
    for feature in features:
        df.boxplot(column=feature, by=cluster_column, ax=axs[int(id/ncols)][id%ncols])
        id += 1
    for ax in axs[nrows-1, id%ncols:]:
        ax.remove()

def plot_hists_by_cluster(
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
    '''
    
    n_clusters = df['cluster'].unique().shape[0]
    axes = df[feature].hist(by=df[cluster_column], bins=bins, layout=(1,n_clusters), figsize=figsize)
    plt.suptitle(f'Distribution of {feature} in each cluster', fontweight='bold')
    for i, ax in enumerate(axes):
        ax.set_title(f'Cluster {i}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Number of incidents')