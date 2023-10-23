# -*- coding: utf-8 -*-
# %% [markdown]
# # Task 1.2 Data Preparation

# %% [markdown]
# ## Utilities

# %% [markdown]
# We import the libraries:

# %%
import math
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath('..\\')) # TODO: c'è un modo per farlo meglio?
import plot_utils

# %% [markdown]
# We define constants and settings for the notebook:

# %%
DIRNAME = os.path.dirname(' ')
DATA_FOLDER_PATH = os.path.join(DIRNAME, 'data')
class counter:
    count = 0
    def get(self):
        self.count += 1
        return self.count - 1
RANDOM_STATE = counter()

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %% [markdown]
# We read the dataset and drop the duplicates rows:

# %%
incidents_path = os.path.join(DATA_FOLDER_PATH, 'incidents.csv')
incidents_data = pd.read_csv(incidents_path, low_memory=False)

incidents_data.drop_duplicates(inplace=True)

# %% [markdown]
# We assess the correct loading of the dataset printing the first 2 rows:

# %%
incidents_data.head(2)

# %% [markdown]
# To avoid having to recompute the data every time the kernel is interrupted and to make the results reproducible in a short execution time, we decided to save the data to CSV files at the end of each data preparation phase.
#
# Below, we provide two specific functions to perform this task.

# %%
LOAD_DATA_FROM_CHECKPOINT = True # boolean: True if you want to load data, False if you want to compute it
CHECKPOINT_FOLDER_PATH = 'data/checkpoints/'

def checkpoint(df, checkpoint_name):
    df.to_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv')

def load_checkpoint(checkpoint_name, casting={}):
    #d_p = pd.datetools.to_datetime
    if casting:
        return pd.read_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv', low_memory=False, index_col=0, parse_dates=['date'], dtype=casting)
    else:
        return pd.read_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv', low_memory=False, index_col=0, parse_dates=['date'])

# %%
incidents_data.info()

# %% [markdown]
# ## Date

# %% [markdown]
# We initially cast the dates to a format that is convenient to manipulate 

# %%
incidents_data['date'] = incidents_data.apply(lambda row : pd.to_datetime(row['date'], format="%Y-%m-%d"), axis = 1)

# %% [markdown]
# check the result of the operation
#

# %%
print(type(incidents_data['date'][0]))
incidents_data.sample(3, random_state = RANDOM_STATE.get())

# %% [markdown]
# We can observe that all dates are syntactically correct
#
# we check the distribution of dates for obvious errors and outliers

# %% [markdown]
# ### Distribution analysis

# %%
# plot range data
tot_row = len(incidents_data.index)

# one binth for every month in the range
min_date = incidents_data['date'].min()
max_date = incidents_data['date'].max()
n_bin = int((max_date - min_date).days / 30) 
n_bin_2 = int(1 + math.log2(tot_row)) #Sturge's rule

equal_freq_bins=incidents_data['date'].sort_values().quantile(np.arange(0,1, 1/n_bin)).to_list()
equal_freq_bins2=incidents_data['date'].sort_values().quantile(np.arange(0,1, 1/n_bin_2)).to_list()

fig, axs = plt.subplots(2, sharex=True, sharey=True)
fig.set_figwidth(14)
fig.set_figheight(6)
fig.suptitle('Dates distribution')

colors_palette = iter(mcolors.TABLEAU_COLORS)
bins = [n_bin, n_bin_2]
ylabels = ['fixed binning', 'Sturge\'s rule']
for i, ax in enumerate(axs):
    ax.hist(incidents_data['date'], bins=bins[i], color=next(colors_palette), density=True)

    ax.set_ylabel(ylabels[i])
    ax.grid(axis='y')
    ax.axvline(min_date, color='k', linestyle='dashed', linewidth=1)
    ax.axvline(max_date, color='k', linestyle='dashed', linewidth=1)
axs[1].set_xlabel('dates')


print('Range data: ', incidents_data['date'].min(), ' - ', incidents_data['date'].max())

# %% [markdown]
# We immediately notice that the dates are distributed over a period (highlighted by the dotted lines) ranging from 2013-01-01 to 2030-11-28, the first error that is easy to notice is that many data exceed the maximum limit of the feature

# %%
ticks = []
labels = []
for i in range(2012, 2032):
    ticks.append(mdates.date2num(pd.to_datetime(str(i) + '-01-01', format="%Y-%m-%d")))
    labels.append(str(i))

boxplot = plt.boxplot(x=mdates.date2num(incidents_data['date']), labels=['dates'])
print()
plt.yticks(ticks, labels)
plt.grid()
dates_data = plot_utils.get_box_plot_data(['dates'], boxplot)
dates_data

# %% [markdown]
# From this graph we can see more clearly how the distribution of dates is concentrated between 2015-07-12 and 2017-08-09 (first and third quartiles respectively) and the values that can be considered correct end around 2018-03-31. This is followed by a large period with no pattern, and finally we find all the outliers previously defined as errors.
#
# It is natural to deduce that one must proceed to correct the problems identified. However, it is difficult to define an error correction method because there are no obvious links between the date and the other features in the dataset, so missing or incorrect values cannot be inferred from them. We try to proceed in 2 ways:
# - the first is to try to find the cause of the error and correct it, based on this assumption: the date could have been entered manually using a numeric keypad, so any errors found could be trivial typos, so let's try subtracting 10 from all dates that are out of range.
# - The second is to replace the incorrect values with the mean or median of the distribution, accepting the inaccuracy if it does not affect the final distribution too much.

# %% [markdown]
# in order to calculate the correlation I convert all the data stored as objects into categories, and replace the value with the associated code (it would be better not to do this for numbers but there are errors)

# %% [markdown]
# ### Error correction

# %% [markdown]
# Let us then try replacing the values in 3 different ways, the first by subtracting 10 years from all the wrong dates, the second by subtracting 11 and the third by replacing them with the median

# %%
#let's try to remove 10 years from the wrong dates, considering the error, a typo
actual_index = incidents_data.index.tolist()

def subtract_ten_if(x):
        if x['date'] > dates_data['upper_whisker'][0].to_datetime64(): 
                return x['date'] - pd.DateOffset(years=10)
        else: return x['date']

def subtract_eleven_if(x):
        if x['date'] > dates_data['upper_whisker'][0].to_datetime64(): 
                return x['date'] - pd.DateOffset(years=11)
        else: return x['date']

def replace_with_median(x):
        ret = x['date']
        while ret > dates_data['upper_whisker'][0].to_datetime64(): 
                ret = dates_data['median'][0].to_datetime64()
        return ret

mod1 = incidents_data.apply(lambda row : subtract_ten_if(row), axis = 1)
mod2 = incidents_data.apply(lambda row : subtract_eleven_if(row), axis = 1)
mod3 = incidents_data.apply(lambda row : replace_with_median(row), axis = 1)

# for hist
dates = [incidents_data['date'],  mod1, mod2, mod3]
ylabels = ['original', 'mod 1', 'mod2', 'mod3']

# %%
print(len(incidents_data.loc[incidents_data['date'] > dates_data['upper_whisker'][0].to_datetime64()]))

# %% [markdown]
# We then observe the distributions thus obtained, in comparison with the original one
#
# the dotted lines represent the low whiskers, the first quartile, the median, the third quartile and the high whiskers. 

# %%
dates_num = []
for i in dates:
    dates_num.append(mdates.date2num(i))

boxplot = plt.boxplot(x=dates_num, labels=ylabels)
plt.yticks(ticks, labels)
plt.grid()

dates_data = plot_utils.get_box_plot_data(ylabels, boxplot)
dates_data

# %%
dates_data['upper_whisker'][1]

# %%
int((dates_data['upper_whisker'][1] - dates_data['lower_whisker'][1]).days / 30)

# %%
# one binth for every month in the range
fixed_bin = int((dates_data['upper_whisker'][0] - dates_data['lower_whisker'][0]).days / 30)
fixed_bin_2 = int(1 + math.log2(tot_row)) #Sturge's rule
'''
prop_bin = []
prop_bin.append(incidents_data['date'].sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list())
prop_bin.append(mod1.sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list())
prop_bin.append(mod2.sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list())
prop_bin.append(mod3.sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list())'''


fig, axs = plt.subplots(4, 2, sharex=True, sharey=False)
fig.set_figwidth(14)
fig.set_figheight(5)
fig.suptitle('Dates distribution')

colors_palette = iter(mcolors.TABLEAU_COLORS)
bins = [n_bin, equal_freq_bins]

for i, ax in enumerate(axs):
    for el in dates_data.loc[0][1:]:
        ax[0].axvline(el, color='k', linestyle='dashed', linewidth=1, alpha=0.4)
        ax[1].axvline(el, color='k', linestyle='dashed', linewidth=1, alpha=0.4)
        
    c = next(colors_palette)
    
    if i == 0:
        n, bins_first_hist, pathces = ax[0].hist(dates[i], bins=fixed_bin, color=c, density=True)
        n,bins_first_hist_2, pathces = ax[1].hist(dates[i], bins=fixed_bin_2, color=c, density=True)
    else:
        ax[0].hist(dates[i], bins=bins_first_hist, color=c, density=True)
        ax[1].hist(dates[i], bins=bins_first_hist_2, color=c, density=True)

    ax[0].set_ylabel(ylabels[i])
    ax[0].grid(axis='y')
    ax[1].grid(axis='y')
    
plt.show()   

# %% [markdown]
# None of the methods used are satisfactory, as they all introduce either large variations in the distribution. On the other hand, using strategies other than those tested could make the date feature unreliable, also because the total number of errors is 23008, 9.5% of the total. The best solution is to remove all the incorrect values and take this into account when applying the knowledge extraction algorithms.
# 
# So we create a new record with the correct date column

# %%
def replace_with_none(x):
        ret = x['date']
        while ret > dates_data['upper_whisker'][0].to_datetime64(): 
                ret = pd.NaT
        return ret

checkpoint_date = pd.DataFrame(index=incidents_data.index, columns=['date'])
checkpoint_date['date'] = incidents_data.apply(lambda row : replace_with_none(row), axis = 1)

# %%
checkpoint_date.sample(3, random_state=RANDOM_STATE.get())

# %%
checkpoint_date['date'].isna().sum()

# %%
checkpoint(checkpoint_date, 'checkpoint_date')

# %% [markdown]
# ## Geographic data

# %%
checkpoint_geo = load_checkpoint('checkpoint_date')
checkpoint_geo.dtypes

# %% [markdown]
# ## Age, gender and number of participants data

# %% [markdown]
# ### Features

# %% [markdown]
# Columns of the dataset are considered in order to verify the correctness and consistency of data related to age, gender, and the number of participants for each incident:
# - *participant_age1*
# - *participant_age_group1*
# - *participant_gender1*
# - *min_age_participants*
# - *avg_age_participants*
# - *max_age_participants*
# - *n_participants_child*
# - *n_participants_teen*
# - *n_participants_adult*
# - *n_males*
# - *n_females*
# - *n_killed*
# - *n_injured*
# - *n_arrested*
# - *n_unharmed*
# - *n_participants*

# %%
# participant_age1,participant_age_group1,participant_gender1,min_age_participants,avg_age_participants,max_age_participants,n_participants_child,n_participants_teen,n_participants_adult,n_males,n_females,n_killed,n_injured,n_arrested,n_unharmed,n_participants
age_data = incidents_data[['participant_age1', 'participant_age_group1', 'participant_gender1', 
    'min_age_participants', 'avg_age_participants', 'max_age_participants',
    'n_participants_child', 'n_participants_teen', 'n_participants_adult', 
    'n_males', 'n_females',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants']]

# %%
age_data.head(10)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
age_data.info()

# %%
age_data['participant_age_group1'].unique()

# %% [markdown]
# ### Studying Data Consistency

# %% [markdown]
# We create some functions to identify and, if possible, correct missing and inconsistent data.
# Below, we provide a brief summary of all the functions used to check data consistency:

# %% [markdown]
# First of all, we convert all the values to type int if the values were consistent (i.e., values related to age and the number of participants for a particular category must be a positive number), all the values that are out of range or contain alphanumeric strings were set to *NaN*.

# %% [markdown]
# Checks done to evaluate the consistency of data related to the minimum, maximum, and average ages of participants, as well as the composition of the age groups:
#
# - min_age_participants $<$ avg_age_participants $<$ max_age_participants
# - n_participants_child $+$ n_participants_teen $+$ n_participants_adult $>$ 0
#
# - $if$ min_age_participants $<$ 12 $then$ n_participants_child $>$ 0
# - $if$ 12 $\leq$ min_age_participants $<$ 18 $then$ n_participants_teen $>$ 0
# - $if$ min_age_participants $\geq$ 18 $then$ n_participants_adult $>$ 0
#
# - $if$ max_age_participants $<$ 12 $then$ n_participants_child $>$ 0 and n_participants_teen $=$ 0 and n_participants_adult $=$ 0
# - $if$ max_age_participants $<$ 18 $then$ n_participants_teen $>$ 0 or n_participants_child $>$ 0 and n_participants_adult $=$ 0
# - $if$ max_age_participants $\geq$ 18 $then$ n_participants_adult $>$ 0
#
# Note that: child = 0-11, teen = 12-17, adult = 18+

# %% [markdown]
# Checks done to evaluate the consistency of data related to number of participants divided by gender and other participants class:
#
# - n_participants $\geq$ 0
# - n_participants $==$ n_males $+$ n_females
# - n_killed $+$ n_injured $\leq$ n_participants
# - n_arrested $\leq$ n_participants
# - n_unharmed $\leq$ n_participants

# %% [markdown]
# We also considered data of participants1, a randomly chosen participant whose data related to gender and age are reported in the dataset. For participants, we have the following features: *participant_age1*, *participant_age_group1*, *participant_gender1*.
#
# Values related to participant_age_group1 and participant_gender1 have been binarized using one-hot encoding, thus creating the boolean features *participant1_child*, *participant1_teen*, *participant1_adult*, *participant1_male*, *participant1_female*.
#
# The following checks are done in order to verify the consistency of the data among them and with respect to the other features of the incident:
#
# - $if$ participant_age1 $<$ 12 $then$ participant_age_group1 $=$ *Child*
# - $if$ 12 $\leq$ participant_age1 $<$ 18 $then$ participant_age_group1 $=$ *Teen*
# - $if$ participant_age1 $\geq$ 18 $then$ participant_age_group1 $==$ *Adult*
#
# - $if$ participant_age_group1 $==$ *Child* $then$ n_participants_child $>$ 0
# - $if$ participant_age_group1 $==$ *Teen* $then$ n_participants_teen $>$ 0
# - $if$ participant_age_group1 $==$ *Adult* $then$ n_participants_adult $>$ 0
#
# - $if$ participant_gender1 $==$ *Male* $then$ n_males $>$ 0
# - $if$ participant_gender1 $==$ *Female* $then$ n_females $>$ 0

# %% [markdown]
# In the initial phase, only the values that were not permissible were set to *NaN*. 
#
# We kept track of the consistency of admissible values by using variables (which could take on the boolean value *True* if they were consistent, *False* if they were not, or *NaN* in cases where data was not present). 
#
# These variables were temporarily included in the dataframe so that we could later replace them with consistent values, if possible, or remove them if they were outside the acceptable range.
#
# Variables:
# - *consistency_age*: Values related to the minimum, maximum, and average ages consistent with the number of participants by age groups.
# - *consistency_n_participant*: The number of participants for different categories consistent with each other.
# - *consistency_gender*: The number of participants by gender consistent with the total number of participants.
# - *consistency_participant1*: Values of features related to participant1 consistent with each other.
#
# - *consistency_participants1_wrt_n_participants*: If *consistency_participants1_wrt_n_participants*, *participant1_age_range_consistency_wrt_all_data*, and *participant1_gender_consistency_wrt_all_data* are all *True*.
#
# - *participant1_age_consistency_wrt_all_data*: Age of participant1 consistent with the minimum and maximum age values of the participants.
# - *participant1_age_range_consistency_wrt_all_data*: Value of the age range (*Child*, *Teen*, or *Adult*) consistent with the age groups of the participants.
# - *participant1_gender_consistency_wrt_all_data*: Gender value of participant1 consistent with the gender breakdown values of the group.
#
# - *nan_values*: Presence of "NaN" values in the row.

# %%
from utils import check_age_gender_data_consistency

if LOAD_DATA_FROM_CHECKPOINT: # load data
    age_temporary_data = load_checkpoint('checkpoint3')
else: # compute data
    age_temporary_data = age_data.apply(lambda row: check_age_gender_data_consistency(row), axis=1)
    checkpoint(age_temporary_data, 'checkpoint3') # save data

# %% [markdown]
# ### Data Exploration without Out-of-Range Data

# %%
age_temporary_data.head(2)

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
age_temporary_data.info()

# %% [markdown]
# We assess the correctness of the checks performed by printing the consistency variable for the first 5 rows and providing a concise summary of their most frequent values.

# %%
age_temporary_data[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].head(5)

# %%
age_temporary_data[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].describe()

# %% [markdown]
# Below, we print the number of rows with 'NaN' or inconsistent data.

# %%
print('Number of rows with null values: ', age_temporary_data[age_temporary_data['nan_values'] == True].shape[0])
print('Number of rows with inconsistent values in age data: ', age_temporary_data[age_temporary_data['consistency_age'] == False].shape[0])
print('Number of rows with inconsistent values in number of participants data: ', age_temporary_data[age_temporary_data[
    'consistency_n_participant'] == False].shape[0])
print('Number of rows with inconsistent values in gender data: ', age_temporary_data[age_temporary_data['consistency_gender'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 data: ', age_temporary_data[age_temporary_data[
    'consistency_participant1'] == False].shape[0])

# %%
print('Number of rows with inconsistent values for participants1: ', age_temporary_data[age_temporary_data[
    'consistency_participant1'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt all other data: ', age_temporary_data[age_temporary_data[
    'consistency_participants1_wrt_n_participants'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age data: ', age_temporary_data[age_temporary_data[
    'participant1_age_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age range data: ', age_temporary_data[age_temporary_data[
    'participant1_age_range_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt gender data: ', age_temporary_data[age_temporary_data[
    'participant1_gender_consistency_wrt_all_data'] == False].shape[0])

# %%
print('Number of rows with null values in age data: ', age_temporary_data[age_temporary_data['consistency_age'].isna()].shape[0])
print('Number of rows with null values in number of participants data: ', age_temporary_data[age_temporary_data[
    'consistency_n_participant'].isna()].shape[0])
print('Number of rows with null values in gender data: ', age_temporary_data[age_temporary_data['consistency_gender'].isna()].shape[0])
print('Number of rows with null values in participants1 data: ', age_temporary_data[age_temporary_data[
    'consistency_participant1'].isna()].shape[0])

# %%
print('Number of rows with all null data: ', age_temporary_data.isnull().all(axis=1).sum())

# %% [markdown]
# We can notice that:
# - The data in our dataset related to participant1, excluding the 1295 cases where age and age group data were inconsistent with each other, always appear to be consistent with the data in the rest of the dataset and can thus be used to fill in missing or incorrect data.
# - In the data related to age and gender, some inconsistencies are present, but they account for only 2.09% and 6.75% of the total dataset rows, respectively.
# - In 93806 rows, at least one field had a *NaN* value.

# %% [markdown]
# Since we noticed that some age data contained impossible values, we have set the age range between 0 and 100 years old. Below, we have verified this by printing the range.

# %%
print('Range age: ', age_temporary_data['min_age_participants'].min(), '-', age_temporary_data['max_age_participants'].max())

# %% [markdown]
# We printed the distribution of participants1 in the age range when age was equal to 18 to verify that the majority of the data were categorized as adults.

# %%
age_data[age_data['participant_age1'] == 18]['participant_age_group1'].value_counts()

# %% [markdown]
# We plotted the age distribution of participant1 and compared it to the distribution of the minimum and maximum participants' age for each group.

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

ax0.hist(age_temporary_data['participant_age1'], bins=100, edgecolor='black', linewidth=0.8)
ax0.set_xlabel('Age')
ax0.set_ylabel('Frequency')
ax0.set_title('Distribution of age participant1')

ax1.hist(age_temporary_data['min_age_participants'], bins=100, edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of min age participants')

ax2.hist(age_temporary_data['max_age_participants'], bins=100, edgecolor='black', linewidth=0.8)
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of max age participants')

plt.show()

# %% [markdown]
# Observing the similar shapes of the distributions provides confirmation that the data pertaining to participant1 is accurate and reliable. Therefore, we can confidently use participant1's data to fill gaps in cases involving groups with a single participant.

# %% [markdown]
# We visualized the number of unique values for the cardinality of participants in each incident and provided a brief summary of this feature below.

# %%
print('Values of n_participants: ', age_temporary_data['n_participants'].unique())
display(age_temporary_data['n_participants'].describe())

# %% [markdown]
# From the data above, it is evident that the third quartile is equal to two participants, and the maximum number of participants per incident reaches the value of 103.
#
# Below, we have presented the distribution of the number of participants for each incident. In order to make the histograms more comprehensible, we have chosen to represent the data on two separate histograms.

# %%
# distribuition number of participants
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

ax0.hist(age_temporary_data['n_participants'], bins=15, range=(0,15), edgecolor='black', linewidth=0.8)
ax0.set_xlabel('Number of participants')
ax0.set_ylabel('Frequency')
ax0.set_title('Distribution of number of participants (1-15 participants)')

ax1.hist(age_temporary_data['n_participants'], bins=15, range=(15,103), edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Number of participants')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of number of participants (15-103 participants)')
plt.show()

# %% [markdown]
# Note that: the chart on the left shows the distribution of data for groups with a number of participants between 0 and 15, while the one on the right displays data for groups between 15 and 103. The y-axes are not equal.

# %% [markdown]
# In the table below, we can see how many data related to the *number of participants* are clearly out of range, divided by age groups.

# %%
age_temporary_data[age_temporary_data['n_participants_adult'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
age_temporary_data[age_temporary_data['n_participants_teen'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
age_temporary_data[age_temporary_data['n_participants_child'] >= 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %% [markdown]
# Based on the tables above, we have evidence to set the maximum number of participants to 103.

# %% [markdown]
# We have provided additional information below for two of the rows with values out of range.

# %%
age_temporary_data.loc[175445]

# %%
age_temporary_data.iloc[236017]

# %% [markdown]
# This data visualization has been helpful in understanding the exceptions in the dataset and correcting them when possible, using other data from the same entry.
#
# In cases where we were unable to obtain consistent data for a certain value, we have set the corresponding field to *NaN*.

# %% [markdown]
# ### Fix Inconsistent Data

# %% [markdown]
# We have created a new DataFrame where we have recorded the corrected and consistent data. The functions and checks necessary for this process are documented in the 'utils.py' file. \
# It's important to note that all these checks are performed based on the assumptions made in previous stages of the analysis.

# %%
# TODO GIULIA: ho settato il valori del dataframe a int ma poi quando li stampo vengono con .0, verificare!!!

# %%
from utils import  set_gender_age_consistent_data

if LOAD_DATA_FROM_CHECKPOINT: # load data
    new_age_data = load_checkpoint('checkpoint4')
else: # compute data
    new_age_data = age_temporary_data.apply(lambda row: set_gender_age_consistent_data(row), axis=1)
    checkpoint(age_temporary_data, 'checkpoint4') # save data

# %% [markdown]
# We display the first 2 rows and a concise summary of the DataFrame:

# %%
new_age_data.head(2)

# %%
new_age_data.info()

# %%
print('Number of rows in which all data are null: ', new_age_data.isnull().all(axis=1).sum())
print('Number of rows with some null data: ', new_age_data.isnull().any(axis=1).sum())
print('Number of rows in which number of participants is null: ', new_age_data[new_age_data['n_participants'].isnull()].shape[0])
print('Number of rows in which number of participants is 0: ', new_age_data[new_age_data['n_participants'] == 0].shape[0])
print('Number of rows in which number of participants is null and n_killed is not null: ', new_age_data[
    new_age_data['n_participants'].isnull() & new_age_data['n_killed'].notnull()].shape[0])

# %%
print('Total rows with null value for n_participants: ', new_age_data['n_participants'].isnull().sum())
print('Total rows with null value for n_participants_child: ', new_age_data['n_participants_child'].isnull().sum())
print('Total rows with null value for n_participants_teen: ', new_age_data['n_participants_teen'].isnull().sum())
print('Total rows with null value for n_participants_adult: ', new_age_data['n_participants_adult'].isnull().sum())
print('Total rows with null value for n_males: ', new_age_data['n_males'].isnull().sum())
print('Total rows with null value for n_females: ', new_age_data['n_females'].isnull().sum())

# %% [markdown]
# We can observe that only for 209 entries in the dataset, all data related to age and gender are *NaN*, while for 104,736 entries, almost one value is *NaN*. From the plot below, we can visualize the null values (highlighted).
#
# It's important to note that we have complete data for *n_killed* and *n_injured* entries, and the majority of missing data are related to age-related features.

# %%
sns.heatmap(new_age_data.isnull(), cbar=False)

# %% [markdown]
# Below, we have provided the distribution of the total number of participants and the number of participants divided by age range for each incident. Once again, to make the histograms more comprehensible, we have opted to present the data on separate histograms.

# %%
# distribuition number of participants
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

ax0.hist(age_temporary_data['n_participants'], bins=15, range=(0,15), edgecolor='black', linewidth=0.8)
ax0.set_xlabel('Number of participants')
ax0.set_ylabel('Frequency')
ax0.set_title('Distribution of number of participants (1-15 participants)')

ax1.hist(age_temporary_data['n_participants'], bins=15, range=(15,103), edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Number of participants')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of number of participants (15-103 participants)')
plt.show()

# %%
print('Max number of participants: ', new_age_data['n_participants'].max())
print('Max number of children: ', new_age_data['n_participants_child'].max())
print('Max number of teens: ', new_age_data['n_participants_teen'].max())
print('Max number of adults: ', new_age_data['n_participants_adult'].max())

# %%
new_age_data[new_age_data['n_participants_adult'] > 60][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
# distribuition number of participants divided by age group
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 8), sharey=False)

ax0.hist(age_temporary_data['n_participants_child'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='blue', label='Children')
ax0.hist(age_temporary_data['n_participants_teen'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='magenta', label='Teens')
ax0.hist(age_temporary_data['n_participants_adult'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='green', label='Adults')
ax0.set_xlabel('Number of participants')
ax0.set_ylabel('Frequency')
ax0.legend()
ax0.set_title('Distribution of number of participants (1-10 participants)')

ax1.hist(age_temporary_data['n_participants_child'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='blue', label='Children')
ax1.hist(age_temporary_data['n_participants_teen'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='magenta', label='Teens')
ax1.hist(age_temporary_data['n_participants_adult'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='green', label='Adults')
ax1.set_xlabel('Number of participants')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.set_title('Distribution of number of participants (10-30 participants)')

ax2.hist(age_temporary_data['n_participants_child'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='blue', label='Children')
ax2.hist(age_temporary_data['n_participants_teen'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='magenta', label='Teens')
ax2.hist(age_temporary_data['n_participants_adult'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='green', label='Adults')
ax2.set_xlabel('Number of participants')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.set_title('Distribution of number of participants (30-103 participants)')

plt.show()

# %% [markdown]
# We observe that in incidents involving children and teenagers under the age of 18, the total number of participants was less than 7 and 27, respectively. In general, incidents involving a single person are much more frequent than other incidents, and most often, they involve teenagers and children, with a smaller percentage involving adults. On the other hand, incidents with multiple participants mostly consist of adults, and as the number of participants increases, the frequency of such incidents decreases. 
#
# Note that the y-axis of the histograms is not equal.

# %% [markdown]
# We also plot the distribution of the number of incidents divided by gender:

# %%
# distribuition number of participants divided by gender
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 8), sharey=False)

ax0.hist(age_temporary_data['n_males'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='blue', label='Males')
ax0.hist(age_temporary_data['n_females'], bins=15, range=(0,10), density=False, histtype='step',
    linewidth=4, color='red', label='Females')
ax0.set_xlabel('Number of participants')
ax0.set_ylabel('Frequency')
ax0.legend()
ax0.set_title('Distribution of number of participants (1-10 participants)')

ax1.hist(age_temporary_data['n_males'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='blue', label='Males')
ax1.hist(age_temporary_data['n_females'], bins=15, range=(10,30), density=False, histtype='step',
    linewidth=4, color='red', label='Females')
ax1.set_xlabel('Number of participants')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.set_title('Distribution of number of participants (10-30 participants)')

ax2.hist(age_temporary_data['n_males'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='blue', label='Males')
ax2.hist(age_temporary_data['n_females'], bins=15, range=(30,103), density=False, histtype='step',
    linewidth=4, color='red', label='Females')
ax2.set_xlabel('Number of participants')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.set_title('Distribution of number of participants (30-103 participants)')  

plt.show()

# %% [markdown]
# From the plot, we can notice that when women are involved in incidents, most of the time, there is only one woman, while in incidents with more than two participants of the same gender, it is more frequent for the participants to be men.
#
# Note that for 1567 entries in the dataset, we have the total number of participants, but we do not have the number of males and females

# %% [markdown]
# Below, we plot the distribution of the average age of participants in each incident.

# %%
plt.figure(figsize=(20, 8))
plt.hist(new_age_data['avg_age_participants'], bins=100, density=False, edgecolor='black', linewidth=0.8)
plt.xlim(0, 100)
plt.xlabel('Participants average age')
plt.ylabel('Frequency')
plt.title('Distribution of participants average age')
plt.show()

# %% [markdown]
# We note that the most frequent average age for incidents falls between 15 and 40 years old.

# %% [markdown]
# ### Final check

# %%
print('Total rows with all values null: ', new_age_data.isnull().all(axis=1).sum())
print('Total rows with null value for n_participants: ', new_age_data['n_participants'].isnull().sum())
print('Total rows with null value for n_participants_child: ', new_age_data['n_participants_child'].isnull().sum())
print('Total rows with null value for n_participants_teen: ', new_age_data['n_participants_teen'].isnull().sum())
print('Total rows with null value for n_participants_adult: ', new_age_data['n_participants_adult'].isnull().sum())
print('Total rows with null value for n_males: ', new_age_data['n_males'].isnull().sum())
print('Total rows with null value for n_females: ', new_age_data['n_females'].isnull().sum())

# %%
#TODO GIULIA: sostituire colonne nuove a quelle vecchie nel dataframe con tutti i dati

# %% [markdown]
# ## Incident Characteristics Data

# %%
# TODO LUCA: ho copiato qua sotto il codice che avevi scritto nel tuo notebook, ho cambiato il nome del data frame e 
# del file csv dove vengono salvate le colonne nuove per essere coerente con quello scritto sopra
# Quando hai tempo riesci a mettere i markdown con dei commenti per spiegare cosa fanno i vari blocchi di codice? Grazie!
#
# Alle fine ho concatenato le colonne nuove che avevi fatto tu con quelle dell'età e genere per fare i check sulla consistenza, 
# la funzione è in 'utils.py' forse si possono aggiungere altri check
#
# Poi non so se alla fine può essere fatto qualche plot in più per visualizzare i dati

# %% [markdown]
# We read the dataset and start working only with characteristics

# %%
# plot characteristics
characteristics_data = incidents_data[['incident_characteristics1', 'incident_characteristics2']]
characteristics_data.head(10)

# %% [markdown]
# List all characteristics from both attributes and their count

# %%
# main characteristics
print(incidents_data.pivot_table(columns=['incident_characteristics1'], aggfunc='size').sort_values(ascending=False))

# %%
#detailed characteristics
print(incidents_data.pivot_table(columns=['incident_characteristics2'], aggfunc='size').sort_values(ascending=False))

# %%
print(incidents_data['incident_characteristics1'].nunique())
print(incidents_data['incident_characteristics2'].nunique())

# %% [markdown]
# We create a DataFrame to count how many times each characteristic in "incident_characteristics1" is mapped to every characteristic in "incident_characteristcs2"

# %%
#create dataframe of characteristics corrispondence
dictionary_characteristics = {}
for characteristic1 in incidents_data['incident_characteristics1'].unique():
    util = {}
    for characteristic2 in incidents_data['incident_characteristics2'].unique():
        util[characteristic2] = 0
    dictionary_characteristics[characteristic1] = util

df_characteristics = pd.DataFrame(dictionary_characteristics)

# %%
def characteristic2_to_int(characteristic):
    try:
        return df_characteristics.axes[0].get_loc(characteristic)
    except ValueError as ve:
        return -1
    
def characteristic1_to_int(characteristic):
    try:
        return df_characteristics.axes[1].get_loc(characteristic)
    except ValueError as ve:
        return -1

# %%
#set corrispondence values in the dataframe
for index, record in characteristics_data.iterrows():
    df_characteristics.iloc[characteristic2_to_int(record["incident_characteristics2"]), [characteristic1_to_int(record["incident_characteristics1"])]] += 1

df_characteristics

# %%
characteristic_to_check = "Shots Fired - No Injuries"

sns.heatmap(df_characteristics[[characteristic_to_check]].sort_values(by=characteristic_to_check, inplace=False, 
    ascending=False).tail(-1))

# %% [markdown]
# Changing the characteristic we want to analyze, we see in the heatmap above that every characteristic in "incident_characteristics1" is coupled with a very small subset of characteristics in "incident_characteristics2". Moreover, the ones with an important number of couplings are even a very few subset of them.

# %% [markdown]
# We create a dictionary to count how many incidents show each certain characteristic

# %%
count_c1 = incidents_data.pivot_table(columns=['incident_characteristics1'], aggfunc='size').sort_values(ascending=False)
count_c2 = incidents_data.pivot_table(columns=['incident_characteristics2'], aggfunc='size').sort_values(ascending=False)

count_dict_c1 = count_c1.to_dict()
count_dict_c2 = count_c2.to_dict()

#merge the two dictionaries to have a list of all the characteristiocs with a count
for key in count_dict_c2.keys():
    if not key in count_dict_c1.keys():
        count_dict_c1[key] = count_dict_c2[key]
    else:
        count_dict_c1[key] += count_dict_c2[key]

characteristcs_count = sorted(count_dict_c1.items(), key=lambda x: x[1], reverse=True) #sort the dictionary by value
characteristcs_count

# %% [markdown]
# We create a Dataframe that associate the characteristics of every incidents to the relative tags

# %%
#add tags to dataframe

tags = ["Firearm", "Air Gun", "Shots", "Aggression", "Suicide", "Injuries", "Death", "Road", "Illegal holding", "House", "School", "Children", "Drugs", "Officers", "Organized", "Social reasons", "Defensive", "Workplace", "Abduction", "Unintentional"]
zeros = [False] * characteristics_data.shape[0]

for tag in tags:
    characteristics_data.insert(characteristics_data.shape[1], tag, zeros)
  
characteristics_data

# %%
#read csv to populate tags

CHARACTERISTIC_FOLDER_PATH = 'data/characteristics_tags_mapping/'
filename = CHARACTERISTIC_FOLDER_PATH + 'characteristics_tags_mapping.csv'
characteristics_tags_list = []
with open(filename, mode='r') as file: #create list with categories and tags
    for line in file:
        line_read = line.rstrip()
        characteristics_tags_list.append(line_read.split(';'))

indicization_list = []
for lst in characteristics_tags_list:
    indicization_list.append(lst[0])

# %%
#given characteristic
#return all the tags 
def get_tags(characteristic):
    if not isinstance(characteristic, str): #if is nan
        return []
    index = indicization_list.index(characteristic)
    tags = characteristics_tags_list[index]
    if len(tags) == 1:
        return []
    return tags[1:]

# %%
for index, record in characteristics_data.iterrows():
    tags = set(get_tags(record['incident_characteristics1']) + get_tags(record['incident_characteristics2']))
    for tag in tags: #set values to tags binary mask
        characteristics_data.at[index, tag] = True

characteristics_data

# %% [markdown]
# We finally check consistency between tag and data

# %%
#add a tag that shows if tags are consistent with data
tag_consistency_attr_name = "tag_consistency"
col = [True] * characteristics_data.shape[0] #tag consistency assumed true
characteristics_data.insert(characteristics_data.shape[1], tag_consistency_attr_name, col)

# %%
characteristics_age_data = pd.concat([new_age_data, characteristics_data], axis=1)

# %%
characteristics_age_data.head(2)

# %%
from utils import check_consistency_tag
for index, row in characteristics_age_data.iterrows():
    characteristics_age_data.at[index, tag_consistency_attr_name] = check_consistency_tag(row)

# %%
# save data
checkpoint(characteristics_age_data, 'checkpoint5') 

# %%
print('Number of rows with incosistency btw tags and other attributes: ', characteristics_age_data[
    characteristics_age_data[tag_consistency_attr_name] == False].shape[0])

# %% [markdown]
# TAG: Firearm, Shots, Aggression, Suicide, Injuries, Death, Road, Illegal holding, House, 
# School, Children, Drugs, Officers, Organized, Social reasons, Defensive, Workplace, Tag Consistency

# %% [markdown]
# ## Final dataset

# %%
# TODO: concatenare data, colonne dati geografici a characteristics_age_data e salvaree


