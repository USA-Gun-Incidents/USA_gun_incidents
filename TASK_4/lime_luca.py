# %% [markdown]
# ### LIME

# %%
import lime
import lime.lime_tabular
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

DATA_FOLDER = '../data/'
BEST_MODELS_DIR = "../TASK_3/best_models"
SEED = 42

# %%
# load trained models
BEST_MODELS_DIR = "../TASK_3/best_models"

knn_best_model = joblib.load(BEST_MODELS_DIR + "/knn.pkl")
svm_best_model = joblib.load(BEST_MODELS_DIR + "/svm.pkl")

# %%
# TODO: sostituire con i dataset presi dai csv

incidents_df = pd.read_csv(DATA_FOLDER + 'clf_incidents_indicators.csv', index_col=0)
indicators_df = pd.read_csv(DATA_FOLDER + 'clf_indicators.csv', index_col=0)

# create is_killed column
label_name = 'is_killed'
incidents_df[label_name] = incidents_df['n_killed'].apply(lambda x: 1 if x >= 1 else 0)
indicators_df[label_name] = incidents_df[label_name]

indicators_df.dropna(inplace=True)

# drop columns with categorical data since we're using distance
categorical_features = ['month', 'day_of_week']
indicators_df.drop(columns=categorical_features, inplace=True)

# apply normalization
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(indicators_df.values)

X_minmax_df = pd.DataFrame(X_minmax, columns=indicators_df.columns, index=indicators_df.index) # normalized dataframe

# split dataset
label = X_minmax_df.pop(label_name)
# we apply stratification since we have unbalanced data
train_set, test_set, train_label, test_label = train_test_split(X_minmax_df, label, stratify=label, test_size=0.30, random_state=SEED)

# %%
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(train_set),
                                                   feature_names=train_set.columns,
                                                   class_names=label_name,
                                                   discretize_continuous=True)

# %%
record = test_set.iloc[0]

# %%
expl = explainer.explain_instance(record, knn_best_model.predict_proba, num_features=3)

# %%
expl.show_in_notebook(show_table=True, show_all=True)

expl.score


