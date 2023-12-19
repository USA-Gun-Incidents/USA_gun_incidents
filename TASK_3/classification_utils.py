import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def scatter_by_label(
        df,
        features,
        label_column,
        ncols=3,
        figsize=(35, 60),
        color_palette=sns.color_palette(),
        title=None
    ):
    '''
    This function plots a scatter plot of each pair of the given features in the given dataframe,
    coloring the points according to their label.

    :param df: dataframe containing the data
    :param features: list of features to plot
    :param cluster_column: name of the dataframe column containing the label
    :param ncols: number of columns of the plot
    :param figsize: size of the figure
    :param color_palette: color palette to use
    '''
    
    ncols = ncols
    nplots = len(features)*(len(features)-1)/2
    nrows = int(nplots / ncols)
    if nplots % ncols != 0:
        nrows += 1

    colors = [color_palette[c] for c in df[label_column]]
    f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    id = 0
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            x, y = df[features].columns[i], df[features].columns[j]
            axs[int(id/ncols)][id%ncols].scatter(df[x], df[y], s=20, c=colors, edgecolor="k")
            axs[int(id/ncols)][id%ncols].set_xlabel(x)
            axs[int(id/ncols)][id%ncols].set_ylabel(y)
            id += 1
    if nrows > 1:
        for ax in axs[nrows-1, id%ncols:]:
            ax.remove()

    legend_elements = []
    label_ids = df[label_column].unique()
    for c in sorted(label_ids):
        legend_elements.append(Line2D(
            [0], [0], marker='o', color='w', label=f'{label_column} {c}', markerfacecolor=color_palette[c]))
    f.legend(handles=legend_elements, loc='lower center', ncols=len(label_ids))

    plt.suptitle(title, fontsize=20)

def scatter_pca_features_by_label(
        X_pca,
        n_components,
        labels,
        palette,
        title=None
    ):
    '''
    This function plots a scatter plot of the first n_components of the X_pca matrix, coloring the points according to
    labels.

    :param X_pca: PCA matrix
    :param n_components: number of components to plot
    :param labels: abels
    :param palette: color palette to use
    :param hue_order: order of the hue levels
    :param title: title of the plot
    '''

    pca_data = {}
    for i in range(n_components):
        pca_data[f'Component {i+1}'] = X_pca[:,i]
    pca_data['Label'] = labels
    df_pca = pd.DataFrame(data=pca_data)
    f = sns.pairplot(
        df_pca,
        hue='Label',
        plot_kws=dict(edgecolor="k"),
        palette=palette,
        corner=True
    )
    f.fig.suptitle(title)