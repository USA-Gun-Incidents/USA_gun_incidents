# %% [markdown]
# ### Distance based classifiers

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import statistics
from classification_utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from itertools import product

DATA_FOLDER = '../data/'
SEED = 42

# %%
incidents_df = pd.read_csv(DATA_FOLDER + 'clf_incidents_indicators.csv', index_col=0)
indicators_df = pd.read_csv(DATA_FOLDER + 'clf_indicators.csv', index_col=0)

# %%
# create is_killed column
label_name = 'is_killed'
incidents_df[label_name] = incidents_df['n_killed'].apply(lambda x: 1 if x >= 1 else 0)
indicators_df[label_name] = incidents_df[label_name]

# %%
indicators_df.isna().sum()

# %%
indicators_df.dropna(inplace=True)

indicators_df

# %%
indicators_df.dtypes

# %%
# drop columns with categorical data since we're using distance
categorical_features = ['month', 'day_of_week']
indicators_df.drop(columns=categorical_features, inplace=True)

# %%
# if needed apply normalization
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(indicators_df.values)

# %%
X_minmax_df = pd.DataFrame(X_minmax, columns=indicators_df.columns, index=indicators_df.index) # normalized dataframe

X_minmax_df

# %%
# classes distrubution
X_minmax_df[label_name].value_counts(normalize=True)

# %%
# scatterplot

features = indicators_df.columns
features = features.drop(label_name)
scatter_by_label(
        indicators_df,
        #features,
        ["age_range", "avg_age", "n_child_prop", "n_teen_prop", "n_males_prop"],
        label_name,
        figsize=(35, 60)
        )

# %%
# split dataset
label = X_minmax_df.pop(label_name)
# we apply stratification since we have unbalanced data
train_set, test_set, train_label, test_label = train_test_split(X_minmax_df, label, stratify=label, test_size=0.30, random_state=SEED)

n_fold = 5
kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
fold_indices = list(kf.split(train_set, train_label))

# %% [markdown]
# ## KNN

# %%
# KNN

# cross validation on number of neighbours
k_values = [1, 5, 10, 20, 50, 100, 200, 400]
best_mcc = -1 # as we have unbalanced classifier we use MCC to evaluate performance
best_k = -1
best_models = []
best_model_predictions = []
val_mcc_per_k = []
for k in k_values:
    
    fold_predictions = []
    mcc_scores = []
    models = []
    for fold in range(1, n_fold + 1):
        # split deployment set in training and validation
        train_indices, val_indices = fold_indices[fold - 1]
        train_fold_set = train_set.iloc[train_indices]
        train_fold_label = label.iloc[train_indices]
        val_fold_set = train_set.iloc[val_indices]
        val_fold_label = label.iloc[val_indices]

        # training
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', metric='minkowski').fit(train_fold_set, train_fold_label)
        models.append(knn)

        # evaluate the model
        val_pred_knn = knn.predict(val_fold_set)
        fold_predictions.append(val_pred_knn)
        mcc_scores.append(matthews_corrcoef(val_fold_label, val_pred_knn))

    # check for the best k
    mean_mcc = np.mean(mcc_scores)
    val_mcc_per_k.append(mean_mcc)
    print("k = " + str(k) + "\tMCC = " + str(mean_mcc))
    if mean_mcc > best_mcc:
        best_mcc = mean_mcc
        best_k = k
        best_model_predictions = fold_predictions
        best_models = models

# %%
def voting_schema_predictions(voting_models, X):
    predictions = [voting_models[i].predict(X) for i in range(len(voting_models))] # get predictions of all models

    # voting schema
    final_predictions = []
    for j in range(len(predictions[0])):
        votes = []
        for i in range(len(predictions)):
            votes.append(predictions[i][j])
        final_predictions.append(statistics.mode(votes))
    
    return final_predictions

# %%
deploy_predictions = [best_models[i].predict(train_set) for i in range(len(best_models))]

# voting schema
final_predictions = []
for j in range(len(deploy_predictions[0])):
    votes = []
    for i in range(len(deploy_predictions)):
        votes.append(deploy_predictions[i][j])
    final_predictions.append(statistics.mode(votes))

print(classification_report(train_label, final_predictions))

# %%
final_predictions = voting_schema_predictions(best_models, train_set)
print(classification_report(train_label, final_predictions))

# %%
final_test_predictions = voting_schema_predictions(best_models, test_set)
print(classification_report(test_label, final_test_predictions))

# %%
# plot curve of MCC for diffenrent k

#avg_acc = val_mcc_per_k
avg_acc = [0.002666614831852752, -5.8079602608956425e-05, -0.0032178177351965182, -0.005367026083421151, -0.00013658472204129725, -0.002019849927929783, 0.0, 0.0]
k_values = [1, 5, 10, 20, 50, 100, 200, 400]

plt.plot(k_values, avg_acc, 'b', label='Validation Accuracy')
plt.title('Validation Accuracy - KNN')
plt.xlabel('K')
plt.ylabel('MCC')
plt.legend()
plt.show()

# %% [markdown]
# ## SVM

# %%
# cross validation
#kernel = ['linear', 'poly', 'rbf', 'sigmoid']
#gamma = ['scale', 'auto']
#C = [0.001, 0.01, 0.1, 1.0]

# per ora facciamo un grid search giocattolo
#kernels = ['linear','rbf']
#gammas = ['scale', 'auto']
#Cs = [0.01, 1.0]

kernels = ['rbf']
gammas = ['auto']
Cs = [0.01, 1.0]


best_kernel = ''
best_gamma = ''
best_C = 1.0
best_mcc = -1
best_models = []
best_model_predictions = []

for kernel in kernels:
    for gamma in gammas:
        for c in Cs:
            fold_predictions = []
            mcc_scores = []
            models = []
            for fold in range(1, n_fold + 1):
                # split deployment set in training and validation
                train_indices, val_indices = fold_indices[fold - 1]
                train_fold_set = train_set.iloc[train_indices]
                train_fold_label = label.iloc[train_indices]
                val_fold_set = train_set.iloc[val_indices]
                val_fold_label = label.iloc[val_indices]

                # training
                svm = SVC(kernel=kernel, C=c, gamma=gamma).fit(train_fold_set, train_fold_label)
                models.append(svm)

                # evaluate the model
                val_pred_svm = svm.predict(val_fold_set)
                fold_predictions.append(val_pred_svm)
                mcc_scores.append(matthews_corrcoef(val_fold_label, val_pred_svm))

            # check for the best parameters
            mean_mcc = np.mean(mcc_scores)
            print("kernel = " + str(kernel) + ", gamma = " + str(gamma) + ", C = " + str(c) + "\tMCC = " + str(mean_mcc))
            if mean_mcc > best_mcc:
                best_mcc = mean_mcc
                best_kernel = kernel
                best_gamma = gamma
                best_C = c
                best_model_predictions = fold_predictions
                best_models = models

# %%
deploy_predictions = [best_models[i].predict(train_set) for i in range(len(best_models))]

# voting schema
final_predictions = []
for j in range(len(deploy_predictions[0])):
    votes = []
    for i in range(len(deploy_predictions)):
        votes.append(deploy_predictions[i][j])
    final_predictions.append(statistics.mode(votes))

print(classification_report(train_label, final_predictions))

# %%
#final_predictions = voting_schema_predictions(best_models, train_set)
final_predictions = voting_schema_predictions(models, train_set)
print(classification_report(train_label, final_predictions))

# %%
# test set prediction
test_predictions = [best_models[i].predict(test_set) for i in range(len(best_models))]

final_test_predictions = []
for j in range(len(test_predictions[0])):
    votes = []
    for i in range(len(test_predictions)):
        votes.append(test_predictions[i][j])
    final_test_predictions.append(statistics.mode(votes))


print(classification_report(test_label, final_test_predictions))

# %%
final_test_predictions = voting_schema_predictions(best_models, test_set)
print(classification_report(test_label, final_test_predictions))

# %%
# roc curves
# confusion matrix
# scatter plot
# decision boundaries?


# sankey


# %%
knn_giocattolo = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree', metric='minkowski')
knn_giocattolo.fit(train_set, train_label)

svm_giocattolo = SVC(kernel='rbf', C=0.01, gamma='auto', probability=True)
svm_giocattolo.fit(train_set, train_label)

# %%
test_pred_knn = knn_giocattolo.predict(test_set)
test_pred_svm_prob = svm_giocattolo.predict_proba(test_set)
skplt.metrics.plot_roc(test_label.values, test_pred_svm_prob)
plt.show()

# %%
test_pred_svm = svm_giocattolo.predict(test_set)

# %%
# compute confusion matrix

cm = confusion_matrix(test_label, test_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot()
plt.show()

# %%
cm = confusion_matrix(test_label, test_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot()
plt.show()


