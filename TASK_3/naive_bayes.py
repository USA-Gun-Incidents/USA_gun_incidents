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
from classification_utils import compute_clf_scores, plot_confusion_matrix

# %% [markdown]
# We load data

# %%
# load the data
incidents_train_df = pd.read_csv('../data/clf_indicators_train.csv', index_col=0)
incidents_test_df = pd.read_csv('../data/clf_indicators_test.csv', index_col=0)
true_labels_train_df = pd.read_csv('../data/clf_y_train.csv', index_col=0)
true_labels_train = true_labels_train_df.values.ravel()
true_labels_test_df = pd.read_csv('../data/clf_y_test.csv', index_col=0)
true_labels_test = true_labels_test_df.values.ravel()

# load the names of the features to use for the classification task
features_for_clf = [
    'location_imp', 'latitude', 'longitude', 'state_code', 'congressional_district', 
    'age_range', 'avg_age', 'n_child_prop', 'n_teen_prop', 'n_males_prop', 'n_participants', 
    'day', 'day_of_week', 'month', 'poverty_perc', 'democrat', 
    'aggression', 'accidental', 'defensive', 'suicide', 'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]

# project on the features_to_use
indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

# %%
indicators_train_df.head(2)

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
# ### Bernulli Naive Bayes Classifier

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
# ### Multinomial Naive Bayes Classifier

# %%
# remuving not positive features
features_for_clf = [
    'location_imp', 'state_code', 'congressional_district', 
    'age_range', 'avg_age', 'n_child_prop', 'n_teen_prop', 'n_males_prop', 'n_participants', 
    'day', 'day_of_week', 'month', 'poverty_perc', 'democrat', 'aggression', 'accidental',
    'defensive', 'suicide', 'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction']

indicators_train_df = incidents_train_df[features_for_clf]
indicators_test_df = incidents_test_df[features_for_clf]

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


