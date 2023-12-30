# %%
# TODO: spiegare che lo LIME facciamo per tutti essendo agnostic
# (anche per quelli già explainable, magari per esse confrontare queste spiegazioni con le originali)

# %%
import pandas as pd
import numpy as np
import json
import lime
import lime.lime_tabular
from IPython.display import HTML, Image
import pydotplus
from sklearn.tree import export_graphviz
from explanation_utils import *

# %%
RANDOM_STATE = 42

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# display white background for explanations
def show_explanation(explanation):
    display(HTML(f"<div style='background-color:white;'>{explanation.as_html()}</div>"))

def get_lime_importance_from_explanation(lime_explanation, prediction):
    pred_explanation = lime_explanation.local_exp[prediction]
    feature_importances = np.zeros(len(pred_explanation))
    for tuple in pred_explanation:
        feature_importances[tuple[0]] = tuple[1]
    return feature_importances

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()

incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

# load the names of the features
features_db = json.loads(open('../data/clf_indicators_names_distance_based.json').read())
features_rb = json.loads(open('../data/clf_indicators_names_rule_based.json').read())

# TODO: when sampling we want to ranndomly choose values assumed in the training set
# (non vogliamo siano fuori dai range, alcune vogliamo siano intere (n_participants con rule based))
categorical_features_db = ['day_x', 'day_y', 'day_of_week_x', 'day_of_week_y', 'month_x', 'month_y', 'year', # ?
    'democrat', 'gun_law_rank',
    'aggression', 'accidental', 'defensive', 'suicide', 'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]
categorical_features_rb = [ 'day', 'day_of_week', 'month', 'year', # ?
    'democrat', 'gun_law_rank',
    'aggression', 'accidental', 'defensive', 'suicide',
    'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]

# project on the used features
indicators_train_db_df = incidents_train_df[features_db]
indicators_train_rb_df = incidents_train_df[features_rb]
indicators_test_db_df = incidents_test_df[features_db]
indicators_test_rb_df = incidents_test_df[features_rb]

clf_names = [clf.value for clf in Classifiers]
rb_clf_names = [Classifiers.DT.value, Classifiers.RF.value, Classifiers.XGB.value, Classifiers.RIPPER.value]


preds = get_classifiers_predictions('../data/classification_results/')
classifiers = get_classifiers_objects('../data/classification_results/')

# %%
# TODO: cambiare i parametri sotto??

# %% [markdown]
# We will use default parameters for LIME, i.e.:
# - exponential kernel
# - kernel width = sqrt (number of features) * 0.75
# - 'auto' feature selection, if number of features > 6 uses 'highest_weights' (selects the features that have the highest product of absolute weight * original data point when learning with all the features)
# - discretize_continuous = True (continuous features are discretized into quartiles)
# - discretizer = 'quartile'
# - sample_around_instance = False (sample from a normal centered on the mean of the feature data)
# 
# Also to explain instances we will use deault parameters, i.e.:
# - num_samples = 5000, number of samples to generate
# - distance_metric = 'euclidean', distance metric to use for weights
# - model_regressor = Ridge Regression, model to use for explanations

# %%
explainer_db = lime.lime_tabular.LimeTabularExplainer(
    indicators_train_db_df.values,
    feature_names=features_db,
    categorical_features=[features_db.index(cat_feature) for cat_feature in categorical_features_db],
    random_state=RANDOM_STATE
)
explainer_rb = lime.lime_tabular.LimeTabularExplainer(
    indicators_train_rb_df.values,
    feature_names=features_rb,
    categorical_features=[features_rb.index(cat_feature) for cat_feature in categorical_features_rb],
    random_state=RANDOM_STATE
)

NUM_SAMPLES = 5000
DISTANCE_METRIC ='euclidean'
REGRESSOR = None # Ridge regression

# %% [markdown]
# ## Attempted Suicide

# %%
selected_records_to_explain_df = pd.read_csv('../data/explanation_results/selected_records_to_explain.csv', index_col=0)
attempted_suicide_pos = selected_records_to_explain_df[selected_records_to_explain_df['instance names']=='Attempted Suicide']['positions'][0]

# %% [markdown]
# ## Decision Tree

# %%
explanation = explainer_rb.explain_instance(
    indicators_test_rb_df.iloc[attempted_suicide_pos].values,
    classifiers[Classifiers.DT.value].predict_proba,
    num_features=len(features_rb),
    num_samples=NUM_SAMPLES,
    distance_metric=DISTANCE_METRIC,
    model_regressor=REGRESSOR,
    #sampling_method='gaussian' # TODO: dalla documentazione sembra si possa sceglierne anche un altro ma non funziona
)
show_explanation(explanation)

# %%
dot_data = export_graphviz(
    classifiers[Classifiers.DT.value],
    out_file=None, 
    feature_names=list(indicators_train_rb_df.columns),
    filled=True,
    rounded=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# %% [markdown]
# ## Random Forest

# %%
explanation = explainer_rb.explain_instance(
    indicators_test_rb_df.iloc[attempted_suicide_pos].values,
    classifiers[Classifiers.RF.value].predict_proba,
    num_features=len(features_rb),
    num_samples=NUM_SAMPLES,
    distance_metric=DISTANCE_METRIC,
    model_regressor=REGRESSOR,
)
show_explanation(explanation)

# %% [markdown]
# ## Extreme gradient boosting

# %%
# explanation = explainer_rb.explain_instance(
#     indicators_test_rb_df.iloc[attempted_suicide_pos].values,
#     classifiers[XGB].predict_proba,
#     num_features=len(features_rb),
#     num_samples=NUM_SAMPLES,
#     distance_metric=DISTANCE_METRIC,
#     model_regressor=REGRESSOR,
# )
# show_explanation(explanation)

# %% [markdown]
# ## K Nearest Neighbors

# %%
explanation = explainer_db.explain_instance(
    indicators_test_db_df.iloc[attempted_suicide_pos].values,
    classifiers[Classifiers.KNN.value].predict_proba,
    num_features=len(features_db),
    num_samples=NUM_SAMPLES,
    distance_metric=DISTANCE_METRIC,
    model_regressor=REGRESSOR,
)
show_explanation(explanation)

# %%
if classifiers[Classifiers.KNN.value].get_params()['n_neighbors'] < 3:
    n_neighbors = 3
else:
    n_neighbors = classifiers[Classifiers.KNN.value].get_params()['n_neighbors']
distances, indeces = classifiers[Classifiers.KNN.value].kneighbors(
    X=indicators_test_db_df.iloc[attempted_suicide_pos].values.reshape(-1,len(features_db)),
    n_neighbors=n_neighbors,
    return_distance=True
)

# %%
neighbors_df = incidents_train_df.iloc[indeces[0]].copy()
neighbors_df['distance'] = distances[0]
# put distance column first
neighbors_df = neighbors_df[['distance'] + [col for col in neighbors_df.columns if col != 'distance']]
neighbors_df.style.background_gradient(cmap='Blues', subset='distance')

# %% [markdown]
# ## Support Vector Machine

# %%
explanation = explainer_db.explain_instance(
    indicators_test_db_df.iloc[attempted_suicide_pos].values,
    classifiers[Classifiers.SVM.value].predict_proba,
    num_features=len(features_db),
    num_samples=NUM_SAMPLES,
    distance_metric=DISTANCE_METRIC,
    model_regressor=REGRESSOR,
)
show_explanation(explanation)

# %%
hyperplane_dists = classifiers[Classifiers.SVM.value].decision_function(
    X=pd.concat([
        indicators_test_db_df.iloc[attempted_suicide_pos].to_frame().T.reset_index(),
        indicators_train_db_df.iloc[indeces[0]].reset_index() # neighbors (could be both mortal or not)
    ]).drop(columns=['index'])
)
probas = classifiers[Classifiers.SVM.value].predict_proba(
    X=pd.concat([
        indicators_test_db_df.iloc[attempted_suicide_pos].to_frame().T.reset_index(),
        indicators_train_db_df.iloc[indeces[0]].reset_index() # neighbors (could be both mortal or not)
    ]).drop(columns=['index'])
)
dist_probs = {
    'distance_from_hyperplane': hyperplane_dists,
    'fatal_probability': probas[:,1]
}
pd.DataFrame(dist_probs)

# %% [markdown]
# ## Feed Forward Neural Networks

# %%
# explanation = explainer_db.explain_instance(
#     indicators_test_db_df.iloc[attempted_suicide_pos].values,
#     classifiers[NN].predict_proba,
#     num_features=len(features_db),
#     num_samples=NUM_SAMPLES,
#     distance_metric=DISTANCE_METRIC,
#     model_regressor=REGRESSOR,
# )
# show_explanation(explanation)

# %% [markdown]
# ## TabNet

# %%
explanation = explainer_db.explain_instance(
    indicators_test_db_df.iloc[attempted_suicide_pos].values,
    classifiers[Classifiers.TN.value].predict_proba,
    num_features=len(features_db),
    num_samples=NUM_SAMPLES,
    distance_metric=DISTANCE_METRIC,
    model_regressor=REGRESSOR,
)
show_explanation(explanation)

# %%
# TODO: fare lo stesso per mass shooting

# %% [markdown]
# ## Evaluation

# %%
# load the already computed default values for features
non_fatal_db_default = pd.read_csv('../data/classification_results/non_fatal_db_default_features.csv').to_numpy()[0]
fatal_db_default = pd.read_csv('../data/classification_results/fatal_db_default_features.csv').to_numpy()[0]
features_db_defaults = [non_fatal_db_default, fatal_db_default]
non_fatal_rb_default = pd.read_csv('../data/classification_results/non_fatal_rb_default_features.csv').to_numpy()[0]
fatal_rb_default = pd.read_csv('../data/classification_results/fatal_rb_default_features.csv').to_numpy()[0]
features_rb_default = [non_fatal_rb_default, fatal_rb_default]

# %%
clf_names = [Classifiers.DT.value, Classifiers.RF.value] # TODO: da togliere quando li abbiamo tutti

# %%
positions_to_explain = selected_records_to_explain_df['positions'].to_list()
instance_names_to_explain = selected_records_to_explain_df['instance names'].to_list()
true_labels_to_explain = selected_records_to_explain_df['true labels'].to_list()

metrics_selected_records = []
for clf_name in clf_names:
    classifier = classifiers[clf_name]
    if clf_name in rb_clf_names:
        explainer = explainer_rb
        feature_defaults = features_rb_default
        instances = indicators_test_rb_df.iloc[positions_to_explain].values
    else:
        explainer = explainer_db
        feature_defaults = features_db_defaults
        instances = indicators_test_db_df.iloc[positions_to_explain].values

    clf_metrics = {}
    for i in range(instances.shape[0]):
        prediction = classifier.predict(instances[i].reshape(1,-1))[0]
        explanation = explainer.explain_instance(instances[i], classifier.predict_proba, num_features=instances.shape[1], top_labels=1)
        feature_importances = get_lime_importance_from_explanation(explanation, prediction)
        feature_default = feature_defaults[0] if true_labels_test[i] == 1 else feature_defaults[1]
        sample_metric = evaluate_explanation(classifier, instances[i], feature_importances, feature_default)
        clf_metrics[instance_names_to_explain[i]] = sample_metric

    clf_metrics_df = pd.DataFrame(clf_metrics).T
    clf_metrics_df.columns = pd.MultiIndex.from_product([clf_metrics_df.columns, [clf_name]])
    metrics_selected_records.append(clf_metrics_df)

metrics_selected_records_df = metrics_selected_records[0].join(metrics_selected_records[1:]).sort_index(level=0, axis=1)
metrics_selected_records_df['True Label'] = true_labels_to_explain

# save faithfulness
faithfulness_df = metrics_selected_records_df[['faithfulness']]
faithfulness_df.columns = faithfulness_df.columns.droplevel()
faithfulness_df.to_csv('../data/explanation_results/lime_faithfulness_selected_records.csv')

# save monotonity
monotonity_df = metrics_selected_records_df[['monotonicity']]
monotonity_df.columns = monotonity_df.columns.droplevel()
monotonity_df.to_csv('../data/explanation_results/lime_monotonicity_selected_records.csv')

metrics_selected_records_df

# %%
random_records_to_explain_df = pd.read_csv('../data/explanation_results/random_records_to_explain.csv', index_col=0)
positions_to_explain = random_records_to_explain_df['positions'].to_list()
true_labels_to_explain = random_records_to_explain_df['true labels'].to_list()

metrics_random_records = {}
for clf_name in clf_names:
    classifier = classifiers[clf_name]
    if clf_name in rb_clf_names:
        explainer = explainer_rb
        feature_defaults = features_rb_default
        instances = indicators_test_rb_df.iloc[positions_to_explain].values
    else:
        explainer = explainer_db
        feature_defaults = features_db_defaults
        instances = indicators_test_db_df.iloc[positions_to_explain].values

    faithfulness = []
    for i in range(instances.shape[0]):
        prediction = classifier.predict(instances[i].reshape(1,-1))[0]
        explanation = explainer.explain_instance(instances[i], classifier.predict_proba, num_features=instances.shape[1], top_labels=1)
        feature_importances = get_lime_importance_from_explanation(explanation, prediction)
        feature_default = feature_defaults[0] if true_labels_test[i] == 1 else feature_defaults[1]
        sample_metric = evaluate_explanation(classifier, instances[i], feature_importances, feature_default)
        faithfulness.append(sample_metric['faithfulness'])
    
    metrics_random_records[clf_name] = {}
    metrics_random_records[clf_name]['mean faithfulness'] = np.nanmean(faithfulness)
    metrics_random_records[clf_name]['std faithfulness'] = np.nanstd(faithfulness)

metrics_random_records_df = pd.DataFrame(metrics_random_records)
metrics_random_records_df.to_csv('../data/explanation_results/lime_metrics_random_records.csv')
metrics_random_records_df


