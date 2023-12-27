# %%
import pandas as pd
import json
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.augmentations import ClassificationSMOTE
from sklearn.model_selection import train_test_split
from classification_utils import *
from time import time
RESULTS_DIR = '../data/classification_results'
clf_name = 'TabNetClassifier'
# paper: https://arxiv.org/pdf/1908.07442.pdf

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
incidents_test_df = pd.read_csv('../data/clf_scaled_indicators_test.csv', index_col=0)

true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

# load the names of the features to use for the classification task
features_for_clf = json.loads(open('../data/clf_indicators_names_distance_based.json').read())

# project on the features to use
indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# scale the data
# FIXME: se poi facciamo grid search bisogna fare come in nn.py
scaler = MinMaxScaler()
indicators_train_scaled = scaler.fit_transform(indicators_train_df)

# split the data into train and validation sets
train_set, val_set, train_labels, val_labels = train_test_split( # FIXME: qui di nuovo il validation è scalato
    indicators_train_scaled,
    true_labels_train,
    test_size=0.2
)

# %%
aug = ClassificationSMOTE(p=0.2)
# TODO: valutare p
# TODO: embedding di feature cateoriche come fanno qui?
# https://github.com/dreamquark-ai/tabnet/blob/develop/census_example.ipynb

# %%
tabnet = TabNetClassifier()
fit_start = time()
tabnet.fit(
  train_set, train_labels,
  eval_set=[(train_set, train_labels), (val_set, val_labels)],
  eval_name=['train', 'val'],
  eval_metric=['balanced_accuracy', 'logloss'],
  max_epochs=2, # TODO: cambiare
  augmentations=aug, # TODO: valutare se togliere
)
fit_time = time()-fit_start

# TODO: provare altr parametri? (in fondo al readme https://github.com/dreamquark-ai/tabnet)

# %%
plt.plot(tabnet.history['loss'], label='Train')
plt.xlabel('Epochs');
plt.ylabel('Training Loss');

# %%
plt.plot(tabnet.history['train_logloss'], label='Train')
plt.plot(tabnet.history['val_logloss'], label='Validation')
plt.xlabel('Epochs');
plt.ylabel('Training Loss');

# %%
plt.plot(tabnet.history['train_balanced_accuracy'], label='Train')
plt.plot(tabnet.history['val_balanced_accuracy'], label='Validation')
plt.xlabel('Epochs');
plt.ylabel('Validation Accuracy');

# %%
# get the predictions on the training data
train_score_start = time()
pred_labels_train = tabnet.predict(indicators_train_df.values)
train_score_time = time()-train_score_start
pred_probas_train = tabnet.predict_proba(indicators_train_df.values)

# get the predictions on the test data
test_score_start = time()
pred_labels_test = tabnet.predict(indicators_test_df.values)
test_score_time = time()-test_score_start
pred_probas_test = tabnet.predict_proba(indicators_test_df.values)

# save the predictions
pd.DataFrame(
   {'labels': pred_labels_test, 'probs': pred_probas_test[:,1]}
).to_csv(f'{RESULTS_DIR}/{clf_name}_preds.csv')

# save the model
tabnet.save_model(f'{RESULTS_DIR}/{clf_name}.pkl')

# TODO:
# save the cv results
# heatmaps params

# %%
compute_clf_scores(
    y_true=true_labels_train,
    y_pred=pred_labels_train,
    train_time=fit_time,
    score_time=train_score_time,
    params=tabnet.get_params(),
    prob_pred=pred_probas_train,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_train_scores.csv'
)

# %%
compute_clf_scores(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    train_time=fit_time,
    score_time=test_score_time,
    params=tabnet.get_params(),
    prob_pred=pred_probas_test,
    clf_name=clf_name,
    path=f'{RESULTS_DIR}/{clf_name}_test_scores.csv'
)

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=pred_labels_test,
    title=clf_name
)

# %%
plot_roc(y_true=true_labels_test, y_probs=[pred_probas_test[:,1]], names=[clf_name])

# %%
plot_predictions_in_features_space(
    df=incidents_test_df,
    features=['n_males_prop', 'n_child_prop', 'n_participants'], # TODO: farlo con features significativve
    true_labels=true_labels_test,
    pred_labels=pred_labels_test,
    figsize=(15, 15)
)

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plot_PCA_decision_boundary(
  train_set=indicators_train_df,
  features=indicators_train_df.columns, # TODO: eventualmente usare solo le numeriche
  train_label=true_labels_train,
  classifier=tabnet,
  classifier_name=clf_name,
  axs=axs
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'n_killed',
    'bar'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'suicide',
    'pie'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'incident_characteristics1',
    'pie'
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'incident_characteristics2',
    'pie',
    pie_perc_threshold=2,
    figsize=(20, 5)
)

# %%
plot_distribution_missclassifications(
    true_labels_test,
    pred_labels_test,
    incidents_test_df,
    'location_imp',
    'hist',
    bins=5
)

# %%
# TODO: spostare ciò che segue nel task di explainability

# %%
explain_matrix, masks = tabnet.explain(train_set)
fig, axs = plt.subplots(1, 3, figsize=(20,40), sharey=True)
for i in range(3):
    axs[i].imshow(masks[i][:50].T) # TODO: + o - di 50?
    axs[i].set_title(f"mask {i}")
    axs[i].set_yticks(np.arange(len(features_for_clf)))
    axs[i].set_yticklabels(labels = features_for_clf)
fig.tight_layout()
# vedere qui https://www.mdpi.com/2227-7390/11/9/2030 per capire come interpretarle?

# %%
explain_matrix.shape

# %%
len(masks)

# %%
len(masks[0])

# %%
fig, axs = plt.subplots(1, 1, figsize=(8,4))
display_feature_importances(
    features_for_clf,
    tabnet.feature_importances_,
    axs,
    title='TabNetClassifier feature importances',
    path=f'{RESULTS_DIR}/{clf_name}_feature_importances.csv'
)


