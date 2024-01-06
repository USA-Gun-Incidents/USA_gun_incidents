import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import LearningCurveDisplay
from scipy.stats import rankdata

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

def plot_confusion_matrix(
    y_true,
    y_pred,
    path=None,
    title=None,
    ):
    '''
    This function plots the confusion matrix of the given estimator.

    :param y_true: true labels
    :param y_pred: predicted labels
    :param path: path where to save the plot
    '''

    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=['Non-Fatal', 'Fatal'],
        cmap=plt.cm.Blues,
        normalize=None,
    )
    if title is not None:
        plt.title(title)
    if path is not None:
        plt.savefig(path)

def plot_predictions_in_features_space(
    df,
    features,
    true_labels,
    pred_labels,
    figsize=(15, 15),
    path=None
    ):
    '''
    This function plots the true and predicted labels in the feature space.

    :param df: dataframe containing the features
    :param features: list of features to use for the plot
    :param true_labels: list of true labels
    :param pred_labels: list of predicted labels
    :param figsize: size of the figure
    :param path: path where to save the figure
    '''
    
    true_labels = ['red' if x==1 else 'blue' for x in true_labels]
    pred_labels = ['red' if x==1 else 'blue' for x in pred_labels]
    errors = ['gray' if x==y else 'orange' for x, y in zip(true_labels, pred_labels)]

    ncols = 3
    import math
    nrows = math.ceil(len(features)*(len(features)-1)/2)

    f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    id = 0

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            x, y = df[features].columns[i], df[features].columns[j]

            axs[id][0].scatter(df[x], df[y], s=20, c=true_labels, edgecolor="k")
            axs[id][0].set_xlabel(x)
            axs[id][0].set_ylabel(y)
            axs[id][0].set_title('True labels')
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Non-Fatal', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Fatal', markerfacecolor='red', markersize=10)
            ]
            axs[id][0].legend(handles=legend_elements, loc='best')

            axs[id][1].scatter(df[x], df[y], s=20, c=pred_labels, edgecolor="k")
            axs[id][1].set_xlabel(x)
            axs[id][1].set_ylabel(y)
            axs[id][1].set_title('Predicted labels')
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Non-Fatal', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Fatal', markerfacecolor='red', markersize=10)
            ]
            axs[id][1].legend(handles=legend_elements, loc='best')

            axs[id][2].scatter(df[x], df[y], s=20, c=errors, edgecolor="k")
            axs[id][2].set_xlabel(x)
            axs[id][2].set_ylabel(y)
            axs[id][2].set_title('Missclassification')
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Correct', markerfacecolor='gray', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Wrong', markerfacecolor='orange', markersize=10)
            ]
            axs[id][2].legend(handles=legend_elements, loc='best')

            id += 1
    if path:
        plt.savefig(path)

def plot_roc(y_true, y_probs, names, ax=None):
    '''
    This function plots the ROC curves for the given y_trues and y_probs.

    :param y_trues: list of true labels
    :param y_probs: list of predicted probabilities
    :param names: list of names for the ROC curves
    '''
    if ax is None:
        _, ax = plt.subplots()
    plot_chance_level = False
    for i, (y_prob, name) in enumerate(zip(y_probs, names)):
        if i==len(names)-1:
            plot_chance_level = True
        RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_prob, ax=ax, name=name, plot_chance_level=plot_chance_level)
    _ = ax.set_title("ROC curves")

def plot_PCA_decision_boundary(
    train_set,
    features,
    train_label,
    classifier,
    classifier_name,
    axs,
    scale=False,
    pca=True
    ):
    '''
    This function plots the decision boundary in the frist two principal components of the given classifier.

    :param train_set: training set
    :param features: list of features to use
    :param train_label: training labels
    :param classifier: classifier
    :param classifier_name: name of the classifier
    :param axs: axis where to plot the decision boundary
    :param scale: whether to scale the data before applying PCA
    :param pca: whether to apply PCA
    '''

    if type(train_set) == pd.DataFrame:
        X = train_set[features].values
    else:
        X = train_set
    y = train_label
    
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    if pca:
        pca = PCA(n_components = 2)
        X = pca.fit_transform(X)

    classifier.fit(X, y)

    plot_decision_regions(X=X, y=y, clf=classifier, legend=2, ax=axs, colors='blue,red')

    legend_Text = axs.get_legend().get_texts()
    if legend_Text[0].get_text()=='0':
        legend_Text[0].set_text('Not-Fatal')
        legend_Text[1].set_text('Fatal')
    else:
        legend_Text[0].set_text('Fatal')
        legend_Text[1].set_text('Not-Fatal')

    axs.set_title('Decision boundary of ' + classifier_name)

def plot_learning_curve(
    classifier,
    classifier_name,
    train_set,
    labels,
    ax,
    train_sizes=np.linspace(0.1, 1.0, 5),
    metric='f1'
    ):
    '''
    This function plots the learning curve of the given classifier.

    :param classifier: classifier
    :param classifier_name: name of the classifier
    :param train_set: training set
    :param labels: training labels
    :param ax: axis where to plot the learning curve
    :param train_sizes: training sizes
    :param metric: metric to use for the learning curve
    '''

    lc_params = { # by default uses stratified 5 fold cross validation
        "X": train_set,
        "y": labels,
        "train_sizes": train_sizes,
        "score_type": "both",
        "n_jobs": -1,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "scoring": metric,
    }
    LearningCurveDisplay.from_estimator(classifier, **lc_params, ax=ax)
    ax.set_title(f'Learning curve of {classifier_name}')

def plot_scores_varying_params(
    gs_results_df,
    param_of_interest,
    fixed_params,
    score_name,
    axs,
    title=None
    ):
    '''
    This function plots the mean and std cv score varying the given parameter of interest.

    :param gs_results_df: grid search results
    :param param_of_interest: parameter of interest
    :param fixed_params: other parameters values
    :param score_name: name of the score
    :param axs: axis where to plot the learning curve
    '''

    gs_results_df = gs_results_df.copy()
    for param_name, param_value in fixed_params.items():
        if param_value is not None:
            gs_results_df = gs_results_df[gs_results_df['param_'+param_name] == param_value]
        else:
            gs_results_df = gs_results_df[gs_results_df['param_'+param_name].isnull()]
    gs_results_df = gs_results_df.sort_values(by='param_'+param_of_interest)
    axs.plot(gs_results_df['param_'+param_of_interest], gs_results_df['mean_test_score'], label=f'cv mean, params: {fixed_params}', marker='o')
    axs.fill_between(
        np.array(gs_results_df['param_'+param_of_interest], dtype='float'),
        np.array(gs_results_df['mean_test_score'] - gs_results_df['std_test_score']),
        np.array(gs_results_df['mean_test_score'] + gs_results_df['std_test_score']),
        alpha=0.3,
        label=f'cv std, params: {fixed_params}'
    )
    axs.set_xlabel(param_of_interest)
    axs.set_ylabel(score_name)
    if title:
        axs.set_title(title)
    axs.legend()

def plot_distribution_missclassifications(
    true_labels,
    pred_labels,
    data,
    attribute,
    kind,
    pie_perc_threshold=5,
    figsize=(15, 5),
    bins=20,
    title=None
    ):
    '''
    This function plots the distribution of the given attribute in the missclassified incidents and in all incidents.

    :param true_labels: true labels
    :param pred_labels: predicted labels
    :param data: dataframe containing the data
    :param attribute: attribute to plot
    :param kind: kind of plot to use ('bar', 'hist', or 'pie')
    :param pie_perc_threshold: threshold to use to group categories in pie plots
    '''
    
    errors = pred_labels != true_labels

    if kind=='bar':
        fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
        data[errors][attribute].value_counts(sort=False).sort_index().plot.bar(
            xlabel=attribute,
            ylabel='count',
            title='Missclassified incidents',
            ax=axs[0]
        )
        data[attribute].value_counts(sort=False).sort_index().plot.bar(
            xlabel=attribute,
            ylabel='count',
            title='All incidents',
            ax=axs[1]
        )
    elif kind=='hist':
        fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
        data[errors][attribute].plot.hist(
            bins=bins,
            xlabel=attribute,
            ylabel='count',
            title='Missclassified incidents',
            ax=axs[0]
        )
        data[attribute].plot.hist(
            bins=bins,
            xlabel=attribute,
            ylabel='count',
            title='All incidents',
            ax=axs[1]
        )
    elif kind=='pie':
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        nunique = data[attribute].nunique()
        if nunique > 2: # group small percentages in 'Other'
            perc = (data[errors][attribute].value_counts()/len(data[errors][attribute])*100).to_frame().reset_index()
            perc.loc[perc[attribute] < pie_perc_threshold, 'index'] = 'Other'
            perc = perc.groupby('index').sum().reset_index()
            axs[0].pie(perc[attribute], labels=perc['index'], autopct='%1.1f%%');
            axs[0].set_title('Missclassified incidents')
            perc = (data[attribute].value_counts()/len(data[attribute])*100).to_frame().reset_index()
            perc.loc[perc[attribute] < pie_perc_threshold, 'index'] = 'Other'
            perc = perc.groupby('index').sum().reset_index()
            axs[1].pie(perc[attribute], labels=perc['index'], autopct='%1.1f%%');
            axs[1].set_title('All incidents')
        else:
            data[errors][attribute].value_counts().plot.pie(
                autopct='%1.1f%%',
                title='Missclassified incidents',
                ax=axs[0]
            )
            data[attribute].value_counts().plot.pie(
                autopct='%1.1f%%',
                title='All incidents',
                ax=axs[1]
            )
    else:
        raise ValueError("kind must be either 'bar', 'hist' or 'pie'")
    if title:
        fig.suptitle(title, fontweight='bold')


def compute_clf_scores(
        y_true,
        y_pred,
        train_time,
        score_time,
        params,
        clf_name,
        prob_pred=None,
        path=None
    ):
    '''
    This function computes the scores of the given classifier.

    :param y_true: true labels
    :param y_pred: predicted labels
    :param train_time: training time
    :param score_time: score time
    :param params: parameters of the classifier
    :param clf_name: name of the classifier
    :param prob_pred: predicted probabilities
    :param path: path where to save the scores
    '''

    report_dict = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, target_names=['Non-Fatal', 'Fatal'])
    report_df = pd.DataFrame(report_dict).T
    report_df.drop(columns=['support'], inplace=True)
    report_df.drop(index=['accuracy'], inplace=True)
    report_series = report_df.stack()
    report_unstacked_df = pd.DataFrame([report_series.values],  columns=[f'{j}-{i}' for i, j in report_series.index])
    report_unstacked_df['accuracy'] = report_dict['accuracy']
    auroc = None
    if prob_pred is not None:
        auroc = roc_auc_score(y_true=y_true, y_score=prob_pred[:,1])
    report_unstacked_df['auroc'] = auroc
    report_unstacked_df['train_time'] = train_time
    report_unstacked_df['score_time'] = score_time
    report_unstacked_df['params'] = str(params)
    report_unstacked_df.index = [clf_name]
    if path:
        report_unstacked_df.to_csv(path)
    return report_unstacked_df

def display_feature_importances(
    feature_names,
    feature_importances,
    axs,
    title=None,
    path=None
    ):
    '''
    This function plots the feature importances.

    :param feature_names: list of feature names
    :param feature_importances: list of feature importances
    :param axs: axis where to plot the feature importances
    :param title: title of the plot
    :param path: path where to save the feature importances
    '''

    sorted_idx = np.argsort(feature_importances)
    sorted_features_names = [feature_names[i] for i in sorted_idx]
    sorted_features_imp = [feature_importances[i] for i in sorted_idx]
    if path is not None:
        pd.DataFrame({
            'features': sorted_features_names,
            'importances': sorted_features_imp,
            'rank': rankdata([-imp for imp in sorted_features_imp], method='dense')
        }).to_csv(path)
    axs.barh(
        y=sorted_features_names,
        width=sorted_features_imp
    )
    axs.set_xlabel('Feature importance')
    if title is not None:
        axs.set_title(title)