# %% [markdown]
# # Naive Bayes Classifier

# %% [markdown]
# Simple Bayesian classifiers have been gaining popularity
# lately, and have been found to perform surprisingly well
# (Friedman 1997; Friedman et al. 1997; Sahami 1996;
# Langley et al. 1992). These probabilistic approaches
# make strong assumptions about how the data is generated, and posit a probabilistic model that embodies
# these assumptions; then they use a collection of labeled
# training examples to estimate the parameters of the
# generative model. Classification on new examples is
# performed with Bayes’ rule by selecting the class that
# is most likely to have generated the example.
# The naive Bayes classifier is the simplest of these
# models, in that it assumes that all attributes of the
# examples are independent of each other given the context of the class. This is the so-called “naive Bayes
# assumption.”
# 
# Because of the independence assumption, the parameters for each attribute
# can be learned separately

# %% [markdown]
# The Naive Bayes classifier demonstrates remarkable flexibility by effectively processing incomplete information, where only a subset of attributes is observed in each instance. 
# Notably, this classifier can handle missing values within test instances. In situations where no attributes are observed for a particular class, it use the prior probability as an estimate for the posterior probability. 
# 
# This adaptability makes Naive Bayes an invaluable tool for scenarios characterized by uncertainty and data incompleteness.

# %% [markdown]
# We import libraries

# %%
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from classification_utils import compute_clf_scores, plot_confusion_matrix

# %% [markdown]
# We load data

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv')
incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv')

# %%
incidents_train_df.head(2)

# %%
true_labels_train = np.where(incidents_train_df['n_killed'] > 0, True, False)
true_labels_test = np.where(incidents_test_df['n_killed'] > 0, True, False)

# %%
print(f'Number of label True in train set: {np.sum(true_labels_train)}, ({np.sum(true_labels_train)/len(true_labels_train)*100}%)')
print(f'Number of label False in train set: {len(true_labels_train)-np.sum(true_labels_train)}, ({(len(true_labels_train)-np.sum(true_labels_train))/len(true_labels_train)*100}%)')
print(f'Number of label True in test set: {np.sum(true_labels_test)}, ({np.sum(true_labels_test)/len(true_labels_test)*100}%)')
print(f'Number of label False in test set: {len(true_labels_test)-np.sum(true_labels_test)}, ({(len(true_labels_test)-np.sum(true_labels_test))/len(true_labels_test)*100}%)')    

# %% [markdown]
# ## Classification

# %% [markdown]
# GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian.
# 
# Bernulli (requires samples to be represented as binary-valued feature vectors)
# 
# Multi-variate Bernoulli does not capture the number of
# times each word occurs, and that it explicitly includes
# the non-occurrence probability of words that do not appear in the document.
# 
# In contrast to the multi-variate Bernoulli event model,
# the multinomial model captures word frequency information in documents.
# 
# ComplementNB implements the complement naive Bayes (CNB) algorithm. CNB is an adaptation of the standard multinomial naive Bayes (MNB) algorithm that is particularly suited for imbalanced data sets

# %% [markdown]
# ### Gaussian Naive Bayes Classifier

# %% [markdown]
# Choose features for classification:

# %%
features_for_clf = [
    'latitude', 'longitude', 'state_code', 'congressional_district', 
    'poverty_perc', 'location_imp', 'year', 'month',
    'age_range', 'max_age', 'avg_age',
    'n_participants', 'n_child_prop', 'n_teen_prop', 'n_males_prop',
    # tag
    #'aggression', 'accidental', 'defensive', 'suicide', 'road', 'house', 'school', 'business',
    #'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
    ]

indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# %% [markdown]
# Prepare Train set and Test set:

# %%
# train set
indicators_train_df.dropna(inplace=True)
true_labels_train = np.where(incidents_train_df['n_killed'][indicators_train_df.index] > 0, True, False)

#test set
indicators_test_df.dropna(inplace=True)
true_labels_test = np.where(incidents_test_df['n_killed'][indicators_test_df.index] > 0, True, False)

# %% [markdown]
# Oversampling of the minority class:

# %%
ros = RandomOverSampler(random_state=42)
indicators_train_df, true_labels_train = ros.fit_resample(indicators_train_df, true_labels_train)

# %%
print(f'Number of label True in train set: {np.sum(true_labels_train)}, ({np.sum(true_labels_train)/len(true_labels_train)*100}%)')
print(f'Number of label False in train set: {len(true_labels_train)-np.sum(true_labels_train)}, ({(len(true_labels_train)-np.sum(true_labels_train))/len(true_labels_train)*100}%)')

# %% [markdown]
# Classification:

# %%
gnb = GaussianNB()
gnb.fit(indicators_train_df, true_labels_train)
y_pred = gnb.predict(indicators_test_df)

print("Number of mislabeled points out of a total %d points : %d" % (indicators_test_df.shape[0], (true_labels_test != y_pred).sum()))

# %%
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=y_pred,
    params=None,
    train_time=None,
    score_time=None,
    prob_pred=None,
    clf_name='GaussianNB',
)
test_scores

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=y_pred,
    title='GaussianNB'
)

# %% [markdown]
# ### Multinomial Naive Bayes Classifier

# %% [markdown]
# Choose features for classification:

# %%
features_for_clf = [
    'state_code', 'congressional_district', 'month',
    'age_range', 'max_age',
    'n_participants', 'n_adult',
    #'n_injured', 'n_arrested', 'n_unharmed',
    # tag
    'aggression', 'accidental', 'defensive', 'suicide', 'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
    ]

indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# %% [markdown]
# Discretize latitude and longitude:

# %%
print(f'Latidude: max value {max(incidents_train_df["latitude"].max(), incidents_test_df["latitude"].max())}, \
    min value {min(incidents_train_df["latitude"].min(), incidents_test_df["latitude"].min())}')
print(f'Longitude: max value {max(incidents_train_df["longitude"].max(), incidents_test_df["longitude"].max())}, \
    min value {min(incidents_train_df["longitude"].min(), incidents_test_df["longitude"].min())}')

# %%
# discetize lat and long
lat_bins = np.linspace(19, 72, 72-19)
long_bins = np.linspace(-166, -67, 166-67)

# train set
indicators_train_df.loc[:, 'latitude_disc'] = pd.cut(incidents_train_df['latitude'], bins=lat_bins, labels=False)
indicators_train_df.loc[:, 'longitude_disc'] = pd.cut(incidents_train_df['longitude'], bins=long_bins, labels=False)

# test set
indicators_test_df.loc[:, 'latitude_disc'] = pd.cut(incidents_test_df['latitude'], bins=lat_bins, labels=False)
indicators_test_df.loc[:, 'longitude_disc'] = pd.cut(incidents_test_df['longitude'], bins=long_bins, labels=False)

# %% [markdown]
# Prepare Train set and Test set:

# %%
# train set
indicators_train_df.dropna(inplace=True)
true_labels_train = np.where(incidents_train_df['n_killed'][indicators_train_df.index] > 0, True, False)

#test set
indicators_test_df.dropna(inplace=True)
true_labels_test = np.where(incidents_test_df['n_killed'][indicators_test_df.index] > 0, True, False)

# %% [markdown]
# Oversampling of the minority class:

# %%
ros = RandomOverSampler(random_state=42)
indicators_train_df, true_labels_train = ros.fit_resample(indicators_train_df, true_labels_train)

# %%
print(f'Number of label True in train set: {np.sum(true_labels_train)}, ({np.sum(true_labels_train)/len(true_labels_train)*100}%)')
print(f'Number of label False in train set: {len(true_labels_train)-np.sum(true_labels_train)}, ({(len(true_labels_train)-np.sum(true_labels_train))/len(true_labels_train)*100}%)')

# %% [markdown]
# Classification:

# %%
mnb = MultinomialNB()
mnb.fit(indicators_train_df, true_labels_train)
y_pred = mnb.predict(indicators_test_df)

print("Number of mislabeled points out of a total %d points : %d" % (indicators_test_df.shape[0], (true_labels_test != y_pred).sum()))

# %%
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=y_pred,
    params=None,
    train_time=None,
    score_time=None,
    prob_pred=None,
    clf_name='MultinomialNB',
)
test_scores

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=y_pred,
    title='MultinomialNB'
)

# %% [markdown]
# ### Complement Naive Bayes Classifier

# %%
cnb = ComplementNB()
cnb.fit(indicators_train_df, true_labels_train)
y_pred = cnb.predict(indicators_test_df)

print("Number of mislabeled points out of a total %d points : %d" % (indicators_test_df.shape[0], (true_labels_test != y_pred).sum()))

# %%
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=y_pred,
    params=None,
    train_time=None,
    score_time=None,
    prob_pred=None,
    clf_name='ComplementNB',
)
test_scores

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=y_pred,
    title='ComplementNB'
)

# %% [markdown]
# ### Bernulli Naive Bayes Classifier

# %%
features_for_clf = [
    # tag: (binary data)
    'aggression', 'accidental', 'defensive', 'suicide', 'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
    ]

indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# %%
def one_hot_encoder(data_train, data_test):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(data_train)
    return ohe.transform(data_train), ohe.transform(data_test)

# %%
# binarize lat and long: one hot encoding
lat_bins = np.linspace(19, 72, 10)
long_bins = np.linspace(-166, -67, 10)

lat_train, lat_test = one_hot_encoder(
    data_train=np.array(pd.cut(incidents_train_df['latitude'], bins=lat_bins, labels=False)).reshape(-1, 1), 
    data_test=np.array(pd.cut(incidents_test_df['latitude'], bins=lat_bins, labels=False)).reshape(-1, 1))
long_train, long_test = one_hot_encoder(
    data_train=np.array(pd.cut(incidents_train_df['longitude'], bins=long_bins, labels=False)).reshape(-1, 1), 
    data_test=np.array(pd.cut(incidents_test_df['longitude'], bins=long_bins, labels=False)).reshape(-1, 1))

for i in range(9):
    # train set
    indicators_train_df.loc[:, f'latitude_{i}'] = lat_train[:, i]
    indicators_train_df.loc[:, f'longitude_{i}'] = long_train[:, i]
    # test set
    indicators_test_df.loc[:, f'latitude_{i}'] = lat_test[:, i]
    indicators_test_df.loc[:, f'longitude_{i}'] = long_test[:, i]

# %%
# binarize age_range
age_range_train, age_range_test = one_hot_encoder(
    data_train=np.array(incidents_train_df['age_range']).reshape(-1, 1), 
    data_test=np.array(incidents_test_df['age_range']).reshape(-1, 1))
for i in range(age_range_train.shape[1]):
    indicators_train_df.loc[:, f'age_range_{i}'] = age_range_train[:, i] # train set
    indicators_test_df.loc[:, f'age_range_{i}'] = age_range_test[:, i] # test set

# %%
# binarize n_participants
n_participants_train, n_participants_test = one_hot_encoder(
    data_train=np.array(incidents_train_df['n_participants']).reshape(-1, 1), 
    data_test=np.array(incidents_test_df['n_participants']).reshape(-1, 1))
for i in range(n_participants_train.shape[1]):
    indicators_train_df.loc[:, f'n_participants_{i}'] = n_participants_train[:, i] # train set
    indicators_test_df.loc[:, f'n_participants_{i}'] = n_participants_test[:, i] # test set

"""# binarize n_adult
n_adult_train, n_adult_test = one_hot_encoder(
    data_train=np.array(incidents_train_df['n_adult']).reshape(-1, 1), 
    data_test=np.array(incidents_test_df['n_adult']).reshape(-1, 1))
for i in range(n_adult_train.shape[1]):
    indicators_train_df.loc[:, f'n_adult_{i}'] = n_adult_train[:, i] # train set
    indicators_test_df.loc[:, f'n_adult_{i}'] = n_adult_test[:, i] # test set"""


# %% [markdown]
# Prepare train set and Test set:

# %%
# train set
indicators_train_df.dropna(inplace=True)
true_labels_train = np.where(incidents_train_df['n_killed'][indicators_train_df.index] > 0, True, False)

#test set
indicators_test_df.dropna(inplace=True)
true_labels_test = np.where(incidents_test_df['n_killed'][indicators_test_df.index] > 0, True, False)

# %% [markdown]
# Oversampling of the minority class:

# %%
ros = RandomOverSampler(random_state=42)
indicators_train_df, true_labels_train = ros.fit_resample(indicators_train_df, true_labels_train)

# %%
print(f'Number of label True in train set: {np.sum(true_labels_train)}, ({np.sum(true_labels_train)/len(true_labels_train)*100}%)')
print(f'Number of label False in train set: {len(true_labels_train)-np.sum(true_labels_train)}, ({(len(true_labels_train)-np.sum(true_labels_train))/len(true_labels_train)*100}%)')

# %% [markdown]
# Classification:

# %%
bnb = BernoulliNB()
bnb.fit(indicators_train_df, true_labels_train)
y_pred = bnb.predict(indicators_test_df)

print("Number of mislabeled points out of a total %d points : %d" % (indicators_test_df.shape[0], (true_labels_test != y_pred).sum()))

# %%
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=y_pred,
    params=None,
    train_time=None,
    score_time=None,
    prob_pred=None,
    clf_name='BernoulliNB',
)
test_scores

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=y_pred,
    title='BernoulliNB'
)

# %% [markdown]
# ### Bernulli NB using Normalized Data

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardization(train_df, test_df, columns, standardizer='Zscore'):
    if standardizer == 'Zscore':
        standardizer = StandardScaler()
    if standardizer == 'MinMax':
        standardizer = MinMaxScaler()
    standardizer.fit(train_df[columns].values)
    return standardizer.transform(train_df[columns].values), standardizer.transform(test_df[columns].values)

# %% [markdown]
# Standardize data using Zscore

# %%
indicators_train_df_std, indicators_test_df_std = standardization(indicators_train_df, indicators_test_df,
    indicators_train_df.columns, standardizer='Zscore')

# %%
bnb_std = BernoulliNB()
bnb_std.fit(indicators_train_df_std, true_labels_train)
y_pred = bnb_std.predict(indicators_test_df_std)

print("Number of mislabeled points out of a total %d points : %d" % (indicators_test_df.shape[0], (true_labels_test != y_pred).sum()))

# %%
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=y_pred,
    params=None,
    train_time=None,
    score_time=None,
    prob_pred=None,
    clf_name='BernulliNB, Zscore',
)
test_scores

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=y_pred,
    title='BernulliNB, Zscore'
)

# %% [markdown]
# Standardize data using MinMax standardizer

# %%
indicators_train_df_minmax, indicators_test_df_minmax = standardization(indicators_train_df, indicators_test_df,
    indicators_train_df.columns, standardizer='MinMax')

# %%
bnb_minmax = BernoulliNB()
bnb_minmax.fit(indicators_train_df_minmax, true_labels_train)
y_pred = bnb_minmax.predict(indicators_test_df_minmax)

print("Number of mislabeled points out of a total %d points : %d" % (indicators_test_df.shape[0], (true_labels_test != y_pred).sum()))

# %%
test_scores = compute_clf_scores(
    y_true=true_labels_test,
    y_pred=y_pred,
    params=None,
    train_time=None,
    score_time=None,
    prob_pred=None,
    clf_name='BernulliNB, MinMax',
)
test_scores

# %%
plot_confusion_matrix(
    y_true=true_labels_test,
    y_pred=y_pred,
    title='BernulliNB, MinMax'
)


