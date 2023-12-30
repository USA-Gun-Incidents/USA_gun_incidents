# %% [markdown]
# # Intrinsic feature importance

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

# %%
# load models and predictions
data_dir = '../data/classification_results/'
DT = 'DecisionTreeClassifier'
RF = 'RandomForestClassifier'
XGB = 'XGBClassifier'
NC = 'NearestCentroidClassifier'
KNN = 'KNearestNeighborsClassifier'
SVM = 'SupportVectorMachineClassifier'
NN = 'NeuralNetworkClassifier'
TN = 'TabNetClassifier'
RIPPER = 'RipperClassifier'

# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()

incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)

# load the names of the features
features_db = json.loads(open('../data/clf_indicators_names_distance_based.json').read())
features_rb = json.loads(open('../data/clf_indicators_names_rule_based.json').read())

# project on the used features
indicators_train_db_df = incidents_train_df[features_db]
indicators_train_rb_df = incidents_train_df[features_rb]
indicators_test_db_df = incidents_test_df[features_db]
indicators_test_rb_df = incidents_test_df[features_rb]

# %%
clf_names = [DT, RF, XGB, SVM, TN] # NN, NC, KNN
feature_imp = {}
for clf in clf_names:
    feature_imp[clf] = {}
    clf_feature_imp = pd.read_csv(data_dir+clf+'_feature_importances.csv')
    feature_imp[clf]['features_names'] = clf_feature_imp['features']
    feature_imp[clf]['features_importance'] = clf_feature_imp['importances']
    feature_imp[clf]['features_rank'] = clf_feature_imp['rank']

# %%
ncols = 2
nplots = len(clf_names)
nrows = int(nplots/ncols)
if nplots % ncols != 0:
    nrows += 1
f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,15), squeeze=False)
for i, clf in enumerate(clf_names):
    axs[int(i/ncols)][i%ncols].bar(
        x=feature_imp[clf]['features_names'],
        height=feature_imp[clf]['features_importance'],
    )
    axs[int(i/ncols)][i%ncols].set_title(clf)
    axs[int(i/ncols)][i%ncols].set_xlabel('Features')
    axs[int(i/ncols)][i%ncols].set_ylabel('Importance')
    for tick in axs[int(i/ncols)][i%ncols].get_xticklabels():
        tick.set_rotation(90);
plt.tight_layout()

# %%
feature_ranks = {}
for clf in clf_names:
    feature_ranks[clf] = {}
    if clf in [DT, RF, XGB]:
        features = features_rb
    else: # SVM, TN
        features = features_db
    for feature in features:
        clf_features = feature_imp[clf]['features_names'].tolist()
        if feature in clf_features:
            feature_pos = clf_features.index(feature)
            feature_ranks[clf][feature] = feature_imp[clf]['features_rank'].tolist()[feature_pos]
        else:
            feature_ranks[clf][feature] = np.nan

feature_imp_df = pd.DataFrame(feature_ranks)
feature_imp_df['mean_rank'] = feature_imp_df.mean(axis=1)
feature_imp_df.style.background_gradient(cmap='Blues', axis=0) # TODO: change axis?

# %% [markdown]
# ## Nearest Centroid

# %%
with open(data_dir+NC+'.pkl', 'rb') as file:
    nc = pickle.load(file)

# %%
pd.DataFrame(nc.centroids_, columns=features_db).T.plot(
    kind='line',
    title='Centroids',
    xlabel='Features',
    rot=90,
    xticks=range(len(features_db)),
)

# %% [markdown]
# ## K Nearest Neighbors

# %%
with open(data_dir+KNN+'.pkl', 'rb') as file:
    knn = pickle.load(file)

# %%
# calcola le distanza tra i primi 1000 esempi
graph_matrix = knn.kneighbors_graph(
    indicators_train_db_df[:1000],
    mode='distance',
    n_neighbors=1000
).toarray()

# %%
G = nx.from_numpy_array(graph_matrix[:,:1000])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)

#labels = nx.get_edge_attributes(G, 'weight')
#nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.show()

# %% [markdown]
# ## TabNet

# %%
# load the model
tn = TabNetClassifier()
tn.load_model(data_dir+TN+'.pkl.zip')

# %%
# explain
minmax_scaler = MinMaxScaler()
indicators_train_scaled = minmax_scaler.fit_transform(indicators_train_db_df)
explain_matrix, masks = tn.explain(indicators_train_scaled)

# %%
# TODO: sottolineare il fatto che le maschere sono diverso sia per livelli che per esempi
# TODO: provare a farle su altri sotto-esempi

# %%
fig, axs = plt.subplots(1, 3, figsize=(20,10), sharey=True)
for i in range(3):
    axs[i].imshow(masks[i][:50].T)
    axs[i].set_title(f"mask {i}")
    axs[i].set_yticks(np.arange(len(features_db)))
    axs[i].set_yticklabels(labels = features_db)
    axs[i].set_xlabel('First 50 samples')
fig.tight_layout()

# %%
fig, axs = plt.subplots(1, 3, figsize=(20,40), sharey=True)
for i in range(3):
    axs[i].imshow(masks[i][np.where(indicators_train_db_df['n_child_prop']!=0)[0][:50]].T)
    axs[i].set_title(f"mask {i}")
    axs[i].set_yticks(np.arange(len(features_db)))
    axs[i].set_yticklabels(labels = features_db)
    axs[i].set_xlabel('50 samples of incidents with n_child_prop!=0')
fig.tight_layout()

# %%
fig, axs = plt.subplots(1, 3, figsize=(20,40), sharey=True)
for i in range(3):
    axs[i].imshow(masks[i][np.where(indicators_train_db_df['n_participants']>2)[0][:50]].T)
    axs[i].set_title(f"mask {i}")
    axs[i].set_yticks(np.arange(len(features_db)))
    axs[i].set_yticklabels(labels = features_db)
    axs[i].set_xlabel('50 samples of incidents with n_participants>2')
fig.tight_layout()

# %% [markdown]
# ## Decision Tree

# %%
# load the model
with open(data_dir+DT+'.pkl', 'rb') as file:
    dt = pickle.load(file)

# %%
dot_data = export_graphviz(
    dt,
    out_file=None, 
    feature_names=list(indicators_train_rb_df.columns),
    filled=True,
    rounded=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# %% [markdown]
# ## Ripper

# %%
# load the model
with open(data_dir+RIPPER+'.pkl', 'rb') as file:
    ripper = pickle.load(file)

# %%
ripper.out_model()

# %% [markdown]
# ## Naive Bayes

# %%
# TODO: ordinare le feature in base alla massima differenza tra feature_log_prob_ (una sorta di feature importance)


