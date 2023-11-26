import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
from sklearn.utils import resample
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, normalized_mutual_info_score

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

def compute_se_per_point(X, clusters, centroids):
    '''
    This function computes the squared error for each point.

    :param X: matrix of data points
    :param clusters: cluster labels
    :param centroids: cluster centroids
    :return: squared error for each point
    '''

    return np.sum(np.square((X - centroids[clusters])), axis=(1 if X.ndim > 1 else 0))

def compute_purity_per_cluster(pred_labels, true_labels):
    '''
    This function computes the purity of each cluster.

    :param pred_labels: the predicted labels
    :param true_labels: the true labels
    :return: the purity of each cluster
    '''
    
    cm = confusion_matrix(pred_labels, true_labels)
    return np.max(cm, axis=1) / np.sum(cm, axis=1)

def compute_overall_purity(pred_labels, true_labels):
    '''
    This function computes the overall purity of the clustering.

    :param pred_labels: the predicted labels
    :param true_labels: the true labels
    :return: the overall purity of the clustering
    '''
    
    purity_per_cluster = compute_purity_per_cluster(pred_labels, true_labels)
    cm = confusion_matrix(pred_labels, true_labels)
    return np.sum((purity_per_cluster * np.sum(cm, axis=1)) / np.sum(cm))

def compute_entropy_per_cluster(pred_labels, true_labels):
    '''
    This function computes the entropy of each cluster.

    :param pred_labels: the predicted labels
    :param true_labels: the true labels
    :return: the entropy of each cluster
    '''
    
    cm = confusion_matrix(pred_labels, true_labels)
    probs = cm / np.sum(cm, axis=1)
    log_probs = np.log2(probs, out=np.zeros_like(probs), where=(probs!=0)) # 0 if prob=0
    return -np.sum(np.multiply(probs, log_probs), axis=1)

def compute_overall_entropy(pred_labels, true_labels):
    '''
    This function computes the overall entropy of the clustering.

    :param pred_labels: the predicted labels
    :param true_labels: the true labels
    :return: the overall entropy of the clustering
    '''
    
    cm = confusion_matrix(pred_labels, true_labels)
    entropy_per_cluster = compute_entropy_per_cluster(pred_labels, true_labels)
    return np.sum((entropy_per_cluster * np.sum(cm, axis=1)) / np.sum(cm))

def align_labels(label1, label2):
    '''
    This function applies pivoting to the confusion matrix between the two sets of labels
    to maximize the sum of the entries on the diagonal.

    :param label1: first set of labels
    :param label2: second set of labels
    :return: label1 (invariant) and aligned label2
    '''

    cm = confusion_matrix(label1, label2)
    cm_argmax = cm.argmax(axis=0)
    label2 = np.array([cm_argmax[i] for i in label2])
    return label1, label2

def compute_external_metrics(
        df,
        cluster_column,
        external_features
    ):
    '''
    This function computes metrics to compare the cluster labels with external features.

    :param df: dataframe containing the external features and the cluster labels
    :param cluster_column: name of the column containing the cluster labels
    :param external_features: list of names of the columns containing the external features
    :return: dataframe containing the metrics
    '''
    
    metrics_df = pd.DataFrame()
    equal_size_features = []
    accuracy = []
    f1 = []
    precision = []
    recall = []
    purity = []
    entropy = []

    n_clusters = df[cluster_column].unique().shape[0]

    for feature in external_features:
        if df[feature].unique().shape[0] != n_clusters:
            continue
        equal_size_features.append(feature)
        classes = df[feature].astype('category').cat.codes
        _, cluster_labels = align_labels(classes, df['cluster'])
        accuracy.append(accuracy_score(y_true=classes, y_pred=cluster_labels))
        f1.append(f1_score(y_true=classes, y_pred=cluster_labels, average='weighted'))
        precision.append(precision_score(y_true=classes, y_pred=cluster_labels, average='weighted', zero_division=0))
        recall.append(recall_score(y_true=classes, y_pred=cluster_labels, average='weighted', zero_division=0))
        purity.append(compute_overall_purity(true_labels=classes, pred_labels=cluster_labels))
        entropy.append(compute_overall_entropy(true_labels=classes, pred_labels=cluster_labels))

    metrics_df['feature'] = equal_size_features
    metrics_df['accuracy'] = accuracy
    metrics_df['f1'] = f1
    metrics_df['precision'] = precision
    metrics_df['recall'] = recall
    metrics_df['purity'] = purity
    metrics_df['entropy'] = entropy
    metrics_df.set_index(['feature'], inplace=True)
    return metrics_df

def compute_permutation_invariant_external_metrics(
        df,
        cluster_column,
        external_features
    ):
    '''
    This function computes permutation invariant metrics to compare the cluster labels with external features.

    :param df: dataframe containing the external features and the cluster labels
    :param cluster_column: name of the column containing the cluster labels
    :param external_features: list of names of the columns containing the external features
    :return: dataframe containing the metrics
    '''
    
    metrics_df = pd.DataFrame()
    adj_rand_scores = []
    homogeneity_scores = []
    completeness_scores = []
    mutual_info_scores = []

    for feature in external_features:
        adj_rand_scores.append(adjusted_rand_score(df[feature], df[cluster_column]))
        mutual_info_scores.append(normalized_mutual_info_score(df[feature], df[cluster_column], average_method='arithmetic'))
        homogeneity_scores.append(homogeneity_score(df[feature], df[cluster_column]))
        completeness_scores.append(completeness_score(df[feature], df[cluster_column]))

    metrics_df['feature'] = external_features
    metrics_df['adjusted rand score'] = adj_rand_scores
    metrics_df['normalized mutual information'] = mutual_info_scores
    metrics_df['homogeneity'] = homogeneity_scores
    metrics_df['completeness'] = completeness_scores


    metrics_df.set_index(['feature'], inplace=True)
    return metrics_df

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

    _, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 2, 1]}, sharey=True)
    df[feature].value_counts().sort_index().plot(kind='bar', ax=axs[0], color=sns.color_palette('hls').as_hex())
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
    
    plot_clusters_size(clusters=df[cluster_column], ax=axs[2], title='Clusters size', color_palette=sns.color_palette('tab10'))
    plt.show()

def scatter_by_cluster(
        df,
        features,
        cluster_column,
        centroids=None,
        ncols=3,
        figsize=(35, 60),
        color_palette=sns.color_palette(),
        title=None
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
    if nrows > 1:
        for ax in axs[nrows-1, id%ncols:]:
            ax.remove()

    legend_elements = []
    clusters_ids = df[cluster_column].unique()
    for c in sorted(clusters_ids):
        legend_elements.append(Line2D(
            [0], [0], marker='o', color='w', label=f'Cluster {c}', markerfacecolor=color_palette[c]))
    f.legend(handles=legend_elements, loc='lower center', ncols=len(clusters_ids))

    plt.suptitle(title, fontsize=20)

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
        figsize=(36,36),
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

def plot_hists_by_cluster(
        df,
        feature,
        cluster_column,
        bins=20,
        figsize=(20, 5),
        title=None,
        color_palette=sns.color_palette()
    ):
    '''
    This function plots a histogram of the given feature in the given dataframe grouped by cluster,
    as well as in the whole dataset.

    :param df: dataframe containing the data
    :param feature: feature to plot
    :param cluster_column: name of the dataframe column containing the cluster labels
    :param bins: number of bins for the histogram
    :param figsize: size of the figure
    :param title: title of the figure
    :param color_palette: color palette to use
    '''

    n_clusters = df[cluster_column].unique().shape[0]
    fig, axs = plt.subplots(nrows=1, ncols=n_clusters+1, figsize=figsize, sharex=True, sharey=True)
    sns.histplot(df[feature], ax=axs[0], bins=bins, color='black', kde=True)
    axs[0].set_title('Whole dataset')
    axs[0].set_xlabel(feature)
    axs[0].set_ylabel('Number of incidents')
    for i in range(1, n_clusters+1):
        sns.histplot(df[feature][df[cluster_column] == i-1], ax=axs[i], bins=bins, color=color_palette[i-1], kde=True)
        axs[i].set_title(f'Cluster {i-1}')
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel('Number of incidents')
    fig.suptitle(title, fontweight='bold')

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

def plot_scores_per_point(score_per_point, clusters, score_name, ax, color_palette=sns.color_palette(), title=None, minx=-0.1):
    '''
    This function plots the clustering score for each point, grouped by cluster.

    :param score_per_point: clustering score for each point
    :param clusters: cluster labels
    :param score_name: name of the clustering score
    :param ax: axis to plot on
    :param color_palette: color palette to use
    :param title: title of the plot
    '''

    n_clusters = len(np.unique(clusters))
    y_lower = 0
    for i in range(n_clusters):
        ith_cluster_score = score_per_point[np.where(clusters == i)[0]]
        ith_cluster_score.sort()
        size_cluster_i = ith_cluster_score.shape[0]
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_score,
            facecolor=color_palette[i],
            edgecolor=color_palette[i],
            alpha=0.7,
        )
        ax.text(minx, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper

    ax.axvline(x=score_per_point.mean(), color="k", linestyle="--", label='Average')
    if title is None:
        title = f"{score_name} for each point in each cluster"
    ax.set_title(title)
    ax.set_xlabel(score_name)
    ax.set_ylabel("Cluster label")
    ax.legend(loc='best')
    ax.set_yticks([])

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

def plot_distance_matrices(X, n_samples, clusters, random_state=None):
    '''
    This function plots the distance matrix and the ideal distance matrix (where points are sorted by cluster).

    :param X: the data points
    :param n_samples: the number of samples to randomly select from X stratifying by clusters
    :param clusters: the cluster labels
    :return: the distance matrix and the ideal distance matrix
    '''
    
    if n_samples < X.shape[0]:
        X_sub, clusters_sub = resample(
            X,
            clusters,
            random_state=random_state,
            stratify=clusters,
            n_samples=n_samples
        )
    else:
        X_sub = X
        clusters_sub = clusters
    
    X_sub_sorted = X_sub[np.argsort(clusters_sub)]
    clusters_sub_sorted = clusters_sub[np.argsort(clusters_sub)]

    dm = squareform(pdist(X_sub_sorted))

    mask = (clusters_sub_sorted[:, None] == clusters_sub_sorted[None, :])
    idm = np.ones_like(dm)
    idm[mask] = 0

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    sns.heatmap(dm, ax=axs[0])
    axs[0].set_title('Distance matrix sorted by cluster')
    sns.heatmap(idm, ax=axs[1])
    axs[1].set_title('Ideal distance matrix sorted by cluster')
    
    corr_coef = np.corrcoef(dm.flatten(), idm.flatten())
    fig.suptitle(f'Pearson Correlation Coefficient = {corr_coef[0,1]:0.2f}', fontweight='bold', y=-0.01)
    
    return dm, idm

def compute_score_between_clusterings(
        clusterings,
        labels,
        score_fun,
        score_name,
        figsize=(8, 5)
    ):
    '''
    This function applies score_fun to all the possible pairs of clusterings
    and returns and plot the matrix with the results.

    :param clusterings: list of clusterings
    :param labels: list of labels for each clustering
    :param score_fun: score function to apply to each pair of clusterings
    :param score_name: name of the score
    :param figsize: size of the figure
    :return: matrix with the scores
    '''

    scores = np.ones((len(clusterings), len(clusterings)))
    for i in range(len(clusterings)):
        for j in range(0, i):
            scores[i][j] = score_fun(clusterings[i], clusterings[j])

    fig, axs = plt.subplots(1, figsize=figsize)
    sns.heatmap(
        scores,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        mask=np.triu(scores),
        ax=axs
    )
    plt.grid(False)
    plt.suptitle(f'{score_name} between different clusterings', fontweight='bold')

    return scores

def write_clusters_to_csv(clusters, file_path):
    '''
    This function writes the clusters to a csv file.

    :param clusters: the clusters to write
    :param file_path: the path of the file to write
    '''

    clusters_df = pd.DataFrame(clusters, columns=['cluster'])
    clusters_df.to_csv(file_path)

def plot_dbscan(X, db, columns, axis_labels, figsize=(10, 10)): 
    labels = db.labels_ 
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True # create an array of booleans where True = core point
    # core point = point that has at least min_samples in its eps-neighborhood

    plt.figure(figsize=figsize)

    colors = [plt.cm.rainbow_r(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k # array of booleans where True = point in cluster k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, columns[0]],
            xy[:, columns[1]],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=10,
            label=f'Cluster {k}'
        )

        # plot noise points
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, columns[0]],
            xy[:, columns[1]],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor=col,
            markersize=6,
            label=f'Cluster {k}'
        )

    plt.grid()
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.legend()
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

def plot_hists_by_cluster_dbscan(df, db, column, figsize=(15, 8)):
    # plot hist for poverty_perc for each cluster
    n_clusters = len(np.unique(db.labels_))
    fig, ax = plt.subplots(int(np.ceil(n_clusters)/2), 2, figsize=figsize, sharex=True, sharey=True)
    for i in range(6):
        ax[int(i/2), i%2].hist(df[column].values[db.labels_==i-1], 
            bins=int(1+3.3*np.log(df[df['cluster']==i-1].shape[0])), 
            label='Noise' if i==0 else f'Cluster {i-1}', 
            edgecolor='black', linewidth=0.8, alpha=0.7,
            color=sns.color_palette(n_colors=df['cluster'].unique().shape[0])[i])
        ax[int(i/2), i%2].set_xlabel(column, fontsize=8)
        ax[int(i/2), i%2].set_yscale('log')
        ax[int(i/2), i%2].tick_params(axis='both', which='major', labelsize=6)
        ax[int(i/2), i%2].legend(fontsize=8)
        ax[int(i/2), i%2].grid(linestyle='--', linewidth=0.5, alpha=0.6)
    fig.suptitle(f'Histograms of {column} by cluster')
    fig.tight_layout()
    plt.show()