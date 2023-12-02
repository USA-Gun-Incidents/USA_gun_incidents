# %% [markdown]
# **Data mining Project - University of Pisa, acedemic year 2023/24**
# 
# **Authors**: Giacomo Aru, Giulia Ghisolfi, Luca Marini, Irene Testa

# %% [markdown]
# # Task 1 - Incidents Data Understanding and Preparation

# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, mpld3
from matplotlib.legend_handler import HandlerPathCollection
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px
import plotly.offline as pyo
import plotly.subplots as sp
from plotly_calplot import calplot
import plotly.graph_objs as go
import math
import os
import calendar
import sys
from sklearn.neighbors import KNeighborsClassifier
from geopy import distance as geopy_distance
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from pyproj import Transformer
import zipfile
import builtins
sys.path.append(os.path.abspath('..'))
from plot_utils import *

# %% [markdown]
# We define constants and settings for the notebook:

# %%
%matplotlib inline

DATA_FOLDER_PATH = '../data/'

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %% [markdown]
# We load the datasets:

# %%
incidents_path = DATA_FOLDER_PATH + 'incidents.csv'
elections_path = DATA_FOLDER_PATH + 'year_state_district_house_cleaned.csv'
poverty_path = DATA_FOLDER_PATH + 'poverty_by_state_year_cleaned.csv'
incidents_df = pd.read_csv(incidents_path, low_memory=False)
elections_df = pd.read_csv(elections_path, low_memory=False)
poverty_df = pd.read_csv(poverty_path, low_memory=False)

# %% [markdown]
# We assess the correct loading of the dataset printing 2 random rows:

# %%
incidents_df.sample(n=2, random_state=1)

# %% [markdown]
# This dataset contains information about gun incidents in the USA.
# 
# In the following table we provide the characteristics of each attribute of the dataset. To define the type of the attributes we used the categorization described by Pang-Ning Tan, Michael Steinbach and Vipin Kumar in the book *Introduction to Data Mining*. For each attribute, we also reported the desidered pandas dtype for later analysis.
# 
# | # | Name | Type | Description | Desired dtype |
# | :-: | :--: | :--: | :---------: | :-----------: |
# | 0 | date | Numeric (Interval) | Date of incident occurrence| datetime |
# | 1 | state | Categorical (Nominal) | Dtate where incident took place | object |
# | 2 | city_or_county | Categorical (Nominal) | City or county where incident took place | object |
# | 3 | address | Categorical (Nominal) | Address where incident took place | object |
# | 4 | latitude | Numeric (Interval) | Latitude of the incident | float64 |
# | 5 | longitude | Numeric (Interval) | Longitude of the incident | float64 |
# | 6 | congressional_district | Categorical (Nominal) | Congressional district where the incident took place | int64 |
# | 7 | state_house_district | Categorical (Nominal) | State house district | int64 |
# | 8 | state_senate_district | Categorical (Nominal) | State senate district where the incident took place | int64 |
# | 9 | participant_age1 | Numeric (Ratio) | Exact age of one (randomly chosen) participant in the incident | int64 |
# | 10 | participant_age_group1 | Categorical (Ordinal) | Exact age group of one (randomly chosen) participant in the incident | object |
# | 11 | participant_gender1 | Categorical (Nominal) | Exact gender of one (randomly chosen) participant in the incident | object |
# | 12 |min_age_participants | Numeric (Ratio) | Minimum age of the participants in the incident | int64 |
# | 13 | avg_age_participants | Numeric (Ratio) | Average age of the participants in the incident | float64 |
# | 14 | max_age_participants | Numeric (Ratio) | Maximum age of the participants in the incident | int64 |
# | 15 | n_participants_child | Numeric (Ratio) | Number of child participants 0-11 | int64 |
# | 16 | n_participants_teen | Numeric (Ratio) | Number of teen participants 12-17 | int64 |
# | 17 | n_participants_adult | Numeric (Ratio) | Number of adult participants (18 +) | int64 |
# | 18 | n_males | Numeric (Ratio) | Number of males participants | int64 |
# | 19 | n_females | Numeric (Ratio) | Number of females participants | int64 |
# | 20 | n_killed | Numeric (Ratio) | Number of people killed | int64 |
# | 21 | n_injured | Numeric (Ratio) | Number of people injured | int64 |
# | 22 | n_arrested | Numeric (Ratio) | Number of arrested participants | int64 |
# | 23 | n_unharmed | Numeric (Ratio) | Number of unharmed participants | int64 |
# | 24 | n_participants | Numeric (Ratio) | Number of participants in the incident | int64 |
# | 25 | notes | Categorical (Nominal) | Additional notes about the incident | object |
# | 26 | incident_characteristics1 | Categorical (Nominal) | Incident characteristics | object |
# | 27 | incident_characteristics2 | Categorical (Nominal) | Incident characteristics (not all incidents have two available characteristics) | object |

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
incidents_df.info()

# %% [markdown]
# We notice that:
# - `congressional_district`, `state_house_district`, `state_senate_district`, `participant_age1`, `n_males`, `n_females`, `n_arrested`, `n_unharmed`, `n_participants` are stored as `float64` while should be `int64`
# - `min_age_participants`, `avg_age_participants`, `max_age_participants`, `n_participants_child`, `n_participants_teen`, `n_participants_adult` are stored as `object` while should be `int64`, this probably indicates the presence of syntactic errors (not in the domain)
# - the presence of missing values within many attributes; the only attributes without missing values are the following: `date`, `state`, `city_or_county`, `n_killed`, `n_injured`, `n_participants`
# 
# We display descriptive statistics of the DataFrame so to better understand how to cast the data:

# %%
incidents_df.describe(include='all')

# %% [markdown]
# We cast the attributes to the correct type:

# %%
# NUMERIC ATTRIBUTES

numerical_features = [
    'participant_age1',
    'n_males',
    'n_females',
    'n_killed',
    'n_injured',
    'n_arrested',
    'n_unharmed',
    'n_participants',
    'min_age_participants',
    'avg_age_participants',
    'max_age_participants',
    'n_participants_child',
    'n_participants_teen',
    'n_participants_adult',
    # (the following attributes should be categorical, but for convenience we keep them numeric)
    'congressional_district',
    'state_house_district',
    'state_senate_district'
    ]
incidents_df[numerical_features] = incidents_df[numerical_features].apply(pd.to_numeric, errors='coerce')

# DATE
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')

# CATEGORICAL ATTRIBUTES
# nominal
incidents_df['participant_gender1'] = incidents_df['participant_gender1'].astype("category")
# ordinal
incidents_df['participant_age_group1'] = incidents_df['participant_age_group1'].astype(
    pd.api.types.CategoricalDtype(categories = ["Child 0-11", "Teen 12-17", "Adult 18+"], ordered = True))

# %% [markdown]
# We display again information about the dataset to check the correctness of the casting and the number of missing values:

# %%
incidents_df.info()

# %% [markdown]
# We drop duplicates:

# %%
print(f"# of rows before dropping duplicates: {incidents_df.shape[0]}")
incidents_df.drop_duplicates(inplace=True)
print(f"# of rows after dropping duplicates: {incidents_df.shape[0]}")

# %% [markdown]
# Now we visualize missing values:

# %%
fig, ax = plt.subplots(figsize=(12,8)) 
sns.heatmap(incidents_df.isnull(), cbar=False, xticklabels=True, ax=ax)
ax.set_title('Null values Heatmap')
ax.set_ylabel('row index')

# %% [markdown]
# We observe that:
# - The following attributes are missing together:
#     - `latitude` and `longitude`
#     - `n_participants_child`, `n_participants_teen`, `n_participants_adult`
#     - `n_males`, `n_females`
#     - `n_arrested`, `n_unharmed`
# - There are many missing values for the following attributes:
#     - `participant_age1`
#     - `min_age_participants`, `avg_age_participants`, `max_age_participants` (often missing together but not always)
#     - `notes`
#     - `incident_characteristics2`
# - Often `participant_age1` is missing but `participant_age_group1` is not and the same holds for `state_house_district` w.r.t `state_senate_district`.
# - `latitude` and `longitude` are often available and could be used to recover the missing values of `address`, `congressional_district`, `state_house_district` and `state_senate_district` (using external libraries that offer this service).

# %% [markdown]
# We display descriptive statistics:

# %%
incidents_df.describe(include='all')

# %% [markdown]
# We can already make some considerations about the dataset:
# - incidents happened in 51 different states (we probably have at least one incident for each state)
# - the most frequent value for the attrbute `state` is Illinois and the most frequent value for  `city_or_county` is Chicago (which is in Illinois, it is consistent)
# - 148 incidents happened at the address "2375 International Pkwy" (an airport in Dallas, Texsas)
# - the majority of incidents involved males
# - there are 52 unique values for the attribute `incidents_characteristics1` and the most frequent is "Shot - Wounded/Injured" (at the time of data collection, it is likely that the values this attribute could take on were limited to a predefined set)
# - there are 90 unique values for the attribute `incidents_characteristicsch2` and the most frequent is "Officer Involved Incident"; this attribute presents more missing values than `incidents_characteristics1` (at the time of data collection, it is likely that the values this attribute could take on were limited to a predefined set)
# - the most frequent value for the attribute notes is "man shot", but the number of unique values this attribute assumes is very high (at the time of data collection the values this attribute could take on were not defined)
# - there are many inconsistencies and/or erros, for example:
#     - the maximum value for the attribute `date` is 2030-11-28
#     - the range of the attributes `age`, `min_age_participants`, `avg_age_participants`, `max_age_participants`, `n_participants_child`, `n_participants_teen`, `n_participants_adult` is outside the domain of the attributes (e.g. the maximum value for the attribute age is 311)
# 
# In the following sections of this notebook we will analyze each attribute in detail.
# 
# To avoid re-running some cells, we save checkpoints of the dataframe at different stages of the analysis and load the dataframe from the last checkpoint using the following functions:

# %%
LOAD_DATA_FROM_CHECKPOINT = True
CHECKPOINT_FOLDER_PATH = 'checkpoints/'

def save_checkpoint(df, checkpoint_name):
    df.to_csv(CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv')

def load_checkpoint(checkpoint_name, casting=None, date_cols=None):
    df = pd.read_csv(
        CHECKPOINT_FOLDER_PATH + checkpoint_name + '.csv',
        low_memory=False,
        index_col=0,
        parse_dates=date_cols,
        dtype=casting
        )
    return df

# %% [markdown]
# ### Date attribute: exploration and preparation

# %% [markdown]
# We plot the distribution of the dates using different binning strategies:

# %%
def plot_dates(df_column, title=None, color=None):
    def iqr_fence(x):
        q1 = x.quantile(0.25)
        med = x.quantile(0.5)
        q3 = x.quantile(0.75)
        IQR = q3 - q1
        u = x.max()
        l = x.min()
        Lower_Fence = builtins.max(builtins.min(x[x > q1 - (1.5 * IQR)], default=pd.Timestamp.min), l)
        Upper_Fence = builtins.min(builtins.max(x[x < q3 + (1.5 * IQR)], default=pd.Timestamp.max), u)
        return [Lower_Fence, q1, med, q3, Upper_Fence]
    relevant_positions = iqr_fence(df_column)
    n_items = len(df_column.index)
    min = df_column.min()
    max = df_column.max()

    fig, axs = plt.subplots(3, sharex=True, figsize=(14, 6))
    fig.suptitle(title)

    # one bin per month
    n_bin = int((max - min).days / 30)
    axs[0].hist(df_column, bins=n_bin, density=True, color=color)
    axs[0].set_ylabel("One bin per month")
    axs[0].grid(axis='y')

    # number of bins computed using Sturge's rule
    n_bin = int(1 + math.log2(n_items))
    axs[1].hist(df_column, bins=n_bin, density=True, color=color)
    axs[1].set_ylabel("Sturge\'s rule binning")
    axs[1].grid(axis='y')

    axs[2].boxplot(x=mdates.date2num(df_column), labels=[''], vert=False)
    axs[2].set_xlabel('date')

    for i in range(2):
        axs[i].axvline(x = relevant_positions[0], color = 'black', linestyle = '--', alpha=0.75)
        axs[i].axvline(x = relevant_positions[1], color = 'black', linestyle = '-.', alpha=0.75)
        axs[i].axvline(x = relevant_positions[2], color = 'black', linestyle = '-.', alpha=0.75)
        axs[i].axvline(x = relevant_positions[3], color = 'black', linestyle = '-.', alpha=0.75)
        axs[i].axvline(x = relevant_positions[4], color = 'black', linestyle = '--', alpha=0.75)



plot_dates(incidents_df['date'], title='Dates distribution')
print('Range data: ', incidents_df['date'].min(), ' - ', incidents_df['date'].max())
print('Unique years: ', sorted(incidents_df['date'].dt.year.unique()))
num_oor = incidents_df[incidents_df['date'].dt.year>2018].shape[0]
print(f'Number of rows with out of range value for the attribute date: {num_oor} ({num_oor/incidents_df.shape[0]*100:.2f}%)')

# %% [markdown]
# These plots show that the number of incidents with an out of range value for the attribute date is non negligible (9.6%) and, excluding these points, there are no incidents happened after the year 2018.
# Instead of discarding rows with out-of-range dates, we will try to correct the errors to prevent excessive data loss.
# Since there are no other features that could suggest the timeframe of the incident, we can only proceed using one of the following approaches:
# - check if those records have duplicates with a correct date
# - suppose dates were entered manually using a numeric keypad and that the errors are typos (e.g. 2030 is actually 2020)
# - replace the errors with the mean or median value
# 
# Let's check if there are duplicates with a correct date:

# %%
incidents_future = incidents_df[incidents_df['date'].dt.year>2018].drop(columns=['date'])
incidents_past = incidents_df[incidents_df['date'].dt.year<2019].drop(columns=['date'])
incidents_past[incidents_past.isin(incidents_future).any(axis=1)].size!=0

# %% [markdown]
# Since there are no duplicates, we proceed with the second and third approach:

# %%
incidents_df['year'] = incidents_df['date'].dt.year
mean_date = incidents_df[incidents_df['year']<2019]['date'].mean()
median_date = incidents_df[incidents_df['year']<2019]['date'].median()

incidents_df['date_minus10'] = incidents_df['date']
incidents_df['date_minus10'] = incidents_df['date'].apply(lambda x : x - pd.DateOffset(years=10) if x.year>2018 else x)
incidents_df['date_minus11'] = incidents_df['date']
incidents_df['date_minus11'] = incidents_df['date'].apply(lambda x : x - pd.DateOffset(years=11) if x.year>2018 else x)
incidents_df['date_mean'] = incidents_df['date']
incidents_df['date_mean'] = incidents_df['date'].apply(lambda x : mean_date if x.year>2018 else x)
incidents_df['date_mean'] = pd.to_datetime(incidents_df['date_mean'], format='%Y-%m-%d') # discard hours, minutes and seconds
incidents_df['date_median'] = incidents_df['date']
incidents_df['date_median'] = incidents_df['date'].apply(lambda x : median_date if x.year>2018 else x)

plot_dates(incidents_df['date_minus10'], 'Dates distribution (year - 10 for oor)')
plot_dates(incidents_df['date_minus11'], 'Dates distribution (year - 11 for oor)', color='orange')
plot_dates(incidents_df['date_mean'], 'Dates distribution (oor replaced with mean)', color='green')
plot_dates(incidents_df['date_median'], 'Dates distribution (oor replaced with median)', color='red')

# %% [markdown]
# Unfortunately, these methods lead to unsatisfactory results, as they significantly alter the distribution. Therefore, we decided to split the date attribute into year, month and day, and set to nan the date attribute if larger than 2018. We also recover the day of the week in which the incident occurred.

# %%
incidents_df.drop(columns=['date_minus10', 'date_minus11', 'date_mean', 'date_median'], inplace=True)
incidents_df['date_original'] = incidents_df['date']
incidents_df['date'] = incidents_df['date'].apply(lambda x : pd.NaT if x.year>2018 else x)
incidents_df['year'] = incidents_df['date'].dt.year.astype('UInt64')
incidents_df['month'] = incidents_df['date'].dt.month.astype('UInt64')
incidents_df['month_name'] = incidents_df['date'].dt.month_name()
incidents_df['day'] = incidents_df['date'].dt.day.astype('UInt64')
incidents_df['day_of_week'] = incidents_df['date'].dt.dayofweek.astype('UInt64')
incidents_df['day_of_week_name'] = incidents_df['date'].dt.day_name()

# %% [markdown]
# We visualize the number of incidents per month:

# %%
incidents_df.groupby('month').size().plot(
    kind='bar',
    figsize=(10, 5),
    title='Number of incidents per month',
    xlabel='Month',
    ylabel='Number of incidents'
)
plt.xticks(range(12), calendar.month_name[1:13], rotation=45);
plt.savefig("../html/incidents_per_month.svg")

# %% [markdown]
# We visualize the number of incidents per day of the week:

# %%
fig = incidents_df.groupby('day_of_week').size().plot(
    kind='bar',
    figsize=(10, 5),
    title='Number of incidents per day of the week',
    xlabel='Day of the week',
    ylabel='Number of incidents'
)
plt.xticks(range(7), calendar.day_name[0:7], rotation=45);
plt.savefig("../html/incidents_per_week_day.svg")

# %% [markdown]
# We display the number of incidents per day over the years:

# %%
def group_by_day(df, date_col):
    counts_by_day = df[date_col].groupby([df[date_col].dt.year, df[date_col].dt.month, df[date_col].dt.day]).size().rename_axis(['year', 'month', 'day']).to_frame('Number of incidents').reset_index()
    counts_by_day[['year', 'month', 'day']] = counts_by_day[['year', 'month', 'day']].astype('int64')
    # add missing days
    for day in pd.date_range(start='2017-01-01', end='2017-12-31'): # 2017%4!=0, this will exclude 29 February
        for year in counts_by_day['year'].unique():
            row_exists = (
                (counts_by_day['year']==year) &
                (counts_by_day['month']==day.month) &
                (counts_by_day['day']==day.day)
                ).any()
            if not row_exists:
                counts_by_day = pd.concat([
                        counts_by_day,
                        pd.DataFrame({'year': [year], 'month': [day.month], 'day': [day.day], 'Number of incidents': [0]})
                    ])
    counts_by_day.sort_values(by=['year', 'month', 'day'], inplace=True)
    counts_by_day['Day'] = counts_by_day.apply(lambda x: f'{x["day"]} {calendar.month_name[x["month"]]}', axis=1)
    return counts_by_day

# %%
incidents_counts_by_day = group_by_day(
    incidents_df[~((incidents_df['date'].dt.day==29) & (incidents_df['date'].dt.month==2))], # exclude 29 february
    'date_original'
)
fig = px.line(
    incidents_counts_by_day,
    x='Day',
    y='Number of incidents',
    title='Number of incidents per day',
    labels={'Day': 'Day of the year', 'Number of incidents': 'Number of incidents'},
    facet_col='year',
    width=1200,
    height=800,
    facet_col_wrap=3
)
fig.update_xaxes(tickangle=-90)
fig.show()
pyo.plot(fig, filename='../html/incidents_per_day_line.html', auto_open=False);

# %% [markdown]
# We display the number of incidents per day over the years as heatmap, using a calendar plot:

# %%
incidenst_per_day = incidents_df.groupby('date').size()

fig = calplot(
    incidenst_per_day.reset_index(),
    x="date",
    y=0,
    title="Number of incidents per day"
)
fig.show()
pyo.plot(fig, filename='../html/incidents_per_day_heatmap.html', auto_open=False);

# %% [markdown]
# We visualize the frequency of incidents occurring across different festivities.
# We consider the following holidays (most of them are [federal holidays](https://en.wikipedia.org/wiki/Federal_holidays_in_the_United_States))
# 
# | Holiday | Date |
# | :------------: | :------------: |
# | New Year’s Day | January 1 |
# | Martin Luther King’s Birthday | 3rd Monday in January |
# | Washington’s Birthday | 3rd Monday in February |
# | Memorial Day | last Monday in May |
# | Juneteenth National Independence Day | June 19 |
# | Independence Day | July 4 |
# | Labor Day | 1st Monday in September |
# | Columbus Day | 2nd Monday in October |
# | Veterans’ Day | November 11 |
# | Thanksgiving Day | 4th Thursday in November |
# | Christmas Day | December 25 |
# | Easter | Sunday (based on moon phase) |
# | Easter Monday | Day after Easter |
# | Black Friday | Day after Thanksgiving |
# 

# %%
holiday_dict = {'New Year\'s Day': ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01'],
    'Martin Luther King\'s Day': ['2013-01-21', '2014-01-20', '2015-01-19', '2016-01-18', '2017-01-16', '2018-01-15'],
    'Washington\'s Birthday': ['2013-02-18', '2014-02-17', '2015-02-16', '2016-02-15', '2017-02-20', '2018-02-19'],
    'Saint Patrick\'s Day': ['2013-03-17', '2014-03-17', '2015-03-17', '2016-03-17', '2017-03-17', '2018-03-17'],
    'Easter': ['2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27', '2017-04-16', '2018-04-01'], 
    'Easter Monday': ['2013-04-01', '2014-04-21', '2015-04-06', '2016-03-28', '2017-04-17', '2018-04-02'],
    'Memorial Day': ['2013-05-27', '2014-05-26', '2015-05-25', '2016-05-30', '2017-05-29', '2018-05-28'],
    'Juneteenth National Independence Day': ['2013-06-19', '2014-06-19', '2015-06-19', '2016-06-19', '2017-06-19', '2018-06-19'],
    'Independence Day': ['2013-07-04', '2014-07-04', '2015-07-03', '2016-07-04', '2017-07-04', '2018-07-04'],
    'Labor Day': ['2013-09-02', '2014-09-01', '2015-09-07', '2016-09-05', '2017-09-04', '2018-09-03'],
    'Columbus Day': ['2013-10-14', '2014-10-13', '2015-10-12', '2016-10-10', '2017-10-09', '2018-10-08'],
    'Veterans\' Day': ['2013-11-11', '2014-11-11', '2015-11-11', '2016-11-11', '2017-11-11', '2018-11-11'],
    'Thanksgiving Day': ['2013-11-28', '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23', '2018-11-22'],
    'Black Friday': ['2013-11-29', '2014-11-28', '2015-11-27', '2016-11-25', '2017-11-24', '2018-11-23'],
    'Christmas Day': ['2013-12-25', '2014-12-25', '2015-12-25', '2016-12-26', '2017-12-25', '2018-12-25']}

# %% [markdown]
# We display the number of incidents occurring on each holiday for each year:

# %%
dfs = []
for holiday in holiday_dict.keys():
    holiday_data = {
        'holiday': holiday,
        'n_incidents_2013': incidents_df[incidents_df['date'].isin([holiday_dict[holiday][0]])].shape[0],
        'n_incidents_2014': incidents_df[incidents_df['date'].isin([holiday_dict[holiday][1]])].shape[0],
        'n_incidents_2015': incidents_df[incidents_df['date'].isin([holiday_dict[holiday][2]])].shape[0],
        'n_incidents_2016': incidents_df[incidents_df['date'].isin([holiday_dict[holiday][3]])].shape[0],
        'n_incidents_2017': incidents_df[incidents_df['date'].isin([holiday_dict[holiday][4]])].shape[0],
        'n_incidents_2018': incidents_df[incidents_df['date'].isin([holiday_dict[holiday][5]])].shape[0],
        'n_incidents_total': incidents_df[incidents_df['date'].isin(holiday_dict[holiday])].shape[0]
    }

    df = pd.DataFrame([holiday_data])
    dfs.append(df)
dfs.append(pd.DataFrame([{
    'holiday': 'Total incidents during the year',
    'n_incidents_2013': incidents_df[incidents_df['date'].dt.year==2013].shape[0],
    'n_incidents_2014': incidents_df[incidents_df['date'].dt.year==2014].shape[0],
    'n_incidents_2015': incidents_df[incidents_df['date'].dt.year==2015].shape[0],
    'n_incidents_2016': incidents_df[incidents_df['date'].dt.year==2016].shape[0],
    'n_incidents_2017': incidents_df[incidents_df['date'].dt.year==2017].shape[0],
    'n_incidents_2018': incidents_df[incidents_df['date'].dt.year==2018].shape[0],
    'n_incidents_total': incidents_df.shape[0]}]))
holidays_df = pd.concat(dfs, ignore_index=True)
holidays_df.sort_values(by=['n_incidents_2017'], ascending=False, inplace=True)
holidays_df

# %% [markdown]
# We display the number of incidents occurring on each holiday for each year as a percentage:

# %%
holidays_df_percents = holidays_df.copy()

holidays_df_percents['n_incidents_2013'] = holidays_df_percents['n_incidents_2013'] / holidays_df_percents[
    holidays_df_percents['holiday']=='Total incidents during the year']['n_incidents_2013'].values[0] * 100
holidays_df_percents['n_incidents_2014'] = holidays_df_percents['n_incidents_2014'] / holidays_df_percents[
    holidays_df_percents['holiday']=='Total incidents during the year']['n_incidents_2014'].values[0] * 100
holidays_df_percents['n_incidents_2015'] = holidays_df_percents['n_incidents_2015'] / holidays_df_percents[
    holidays_df_percents['holiday']=='Total incidents during the year']['n_incidents_2015'].values[0] * 100
holidays_df_percents['n_incidents_2016'] = holidays_df_percents['n_incidents_2016'] / holidays_df_percents[
    holidays_df_percents['holiday']=='Total incidents during the year']['n_incidents_2016'].values[0] * 100
holidays_df_percents['n_incidents_2017'] = holidays_df_percents['n_incidents_2017'] / holidays_df_percents[
    holidays_df_percents['holiday']=='Total incidents during the year']['n_incidents_2017'].values[0] * 100
holidays_df_percents['n_incidents_2018'] = holidays_df_percents['n_incidents_2018'] / holidays_df_percents[
    holidays_df_percents['holiday']=='Total incidents during the year']['n_incidents_2018'].values[0] * 100
holidays_df_percents.drop(holidays_df.index[0], inplace=True)

holidays_df_percents.sort_values(by=['n_incidents_2017'], ascending=False, inplace=True)
holidays_df_percents

# %% [markdown]
# We visualize the data using a bar plot:

# %%
fig = px.bar(
    holidays_df.drop(holidays_df.index[0]),
    x='holiday',
    y=['n_incidents_2013', 'n_incidents_2014', 'n_incidents_2015', 'n_incidents_2016', 'n_incidents_2017', 'n_incidents_2018'],
    title='Number of incidents per holiday',
    labels={'holiday': 'Holiday', 'value': 'Number of incidents', 'variable': 'Year'},
    barmode='group',
)
fig.show()
pyo.plot(fig, filename='../html/incidents_per_holiday.html', auto_open=False);

# %% [markdown]
# We notice that the distribution of the number of incidents during each holiday remains consistent over the years. This consistency aligns with our expectations, given the similarity in the distribution of incidents across days throughout the years.
# 
# New Year's Eve, followed by the Independence day are the holiday with the highest number of incidents, while Christmas and Thanksgiving are the holidays with the lowest number of incidents.

# %% [markdown]
# ### Geospatial features: exploration and preparation

# %% [markdown]
# We check if the values of the attribute `state` are admissible comparing them with an official list of states:

# %%
usa_states_df = pd.read_csv(
    'https://www2.census.gov/geo/docs/reference/state.txt',
    sep='|',
    dtype={'STATE': str, 'STATE_NAME': str}
)
usa_name_alphcode = usa_states_df.set_index('STATE_NAME').to_dict()['STUSAB']
states = incidents_df['state'].unique()
not_existing_states = False
missing_states = False

for state in states:
    if state not in usa_name_alphcode:
        not_existing_states = True
        print(f"State {state} does not exist")

for state in usa_name_alphcode:
    if state not in states:
        missing_states = True
        print(f"State {state} is missing")

if not_existing_states == False:
    print("All the values of the attribute 'states' are actually USA states (there are no misspelling or other errors).")
if missing_states == False:
    print("There is at least one incident for each USA state.")

# %% [markdown]
# We now check if, given a certain value for the attributes `latitude` and `longitude`, the attribute `city_or_county` has always the same value:

# %%
incidents_df.groupby(['latitude', 'longitude'])['city_or_county'].unique()[lambda x: x.str.len() > 1].to_frame()

# %% [markdown]
# That is not true and is due to the fact that sometimes the attribute `city_or_county` takes on the value of the city, other times the value of the county (as in the first row displayed above). Furthermore, we notice that even when the attribute refers to the same county it could be written in different ways (e.g. "Bethel (Newtok)", "Bethel (Napaskiak)", "Bethel"). 

# %% [markdown]
# We now check if a similar problem occurs for the attribute `address`:

# %%
incidents_df.groupby(['latitude', 'longitude'])['address'].unique()[lambda x: x.str.len() > 1].to_frame()

# %% [markdown]
# Still this attribute may be written in different ways (e.g. "Avenue" may also be written as "Ave", or "Highway" as "Hwy"). There could also be some errors (e.g. the same point corresponds to the address "33rd Avenue", "Kamehameha Highway" and "Kilauea Avenue extension").
# 
# We plot on a map the location of the incidents:

# %%
fig = px.scatter_mapbox(
    lat=incidents_df['latitude'],
    lon=incidents_df['longitude'],
    zoom=0, 
    height=500,
    width=800
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# %% [markdown]
# There are some points in China that are clearly wrong. We display the rows of the dataset that correspond to one of these points:

# %%
incidents_df[(incidents_df['latitude'] == 37.6499) & (incidents_df['longitude'] == 97.4331)]

# %% [markdown]
# That point has probably the correct values for the attributes `state` and `city_or_county`.

# %% [markdown]
# To fix these inconsistencies we used the library [GeoPy]((https://geopy.readthedocs.io/en/stable/)). This library allows to retrieve the address (state, county, suburb, city, town, village, location name, and other features) corresponding to a given latitude and longitude. We queried the library using all the latitudes and longitudes of the points in the dataset and we saved the results in the CSV file we now load:

# %%
geopy_path = os.path.join(DATA_FOLDER_PATH, 'external_data/geopy.csv')
geopy_df = pd.read_csv(geopy_path, index_col=['index'], low_memory=False, dtype={})
geopy_df.sample(2, random_state=1)

# %% [markdown]
# The rows in this dataframe correspond to the rows in the original dataset. Its column *coord_presence* is false if the corresponding row in the original dataset did not have latitude and longitude values.
# 
# Among all the attributes returned by GeoPy, we selected and used the following:
# - *importance*: Numerical value $\in [0,1]$, indicating the importance of the location (compared to other locations)
# - *addresstype*: Address type (e.g., "house," "street," "postcode")
# - *state*: State of the location
# - *county*: County of the location
# - *suburb*: Suburb of the location
# - *city*: City of the location
# - *town*: Town of the location
# - *village*: Village of the location
# - *display_name*: User-friendly representation of the location, often formatted as a complete address

# %%
print(f"Number of rows in which surburb is null: {geopy_df.loc[geopy_df['suburb_geopy'].isna()].shape[0]}\n")
print('Coordinate presence:')
display(geopy_df['coord_presence'].value_counts())
print('Importance presence:')
display(geopy_df['importance_geopy'].notna().value_counts())
print(f"Number of rows in which city is null and town is not null: {geopy_df[(geopy_df['city_geopy'].isnull()) & (geopy_df['town_geopy'].notnull())].shape[0]}\n")
print("Values of addresstype:")
print(geopy_df['addresstype_geopy'].unique())
print(f"\nNumber of rows in which addresstype is null: {geopy_df[geopy_df['addresstype_geopy'].isnull()].shape[0]}")

# %% [markdown]
# We also downloaded from [Wikipedia](https://en.wikipedia.org/wiki/County_(United_States)) the list of the counties (or their equivalent) in each state. 
# 
# This data was used in case incidents data did not match GeoPy data and when latitude and longitude where not available to check whether the county actually belonged to the state.

# %%
counties_path = os.path.join(DATA_FOLDER_PATH, 'external_data/counties.csv')
counties_df = pd.read_csv(counties_path)
counties_df.head()

# %% [markdown]
# We now check and correct the consistency of the geographic data:

# %%
from TASK_1.data_preparation_utils import check_geographical_data_consistency

if LOAD_DATA_FROM_CHECKPOINT:
    incidents_df = load_checkpoint('checkpoint_1', date_cols=['date', 'date_original'])
else:
    geo_df = incidents_df[['state', 'city_or_county', 'address', 'latitude', 'longitude']]
    geo_df = pd.concat([geo_df, geopy_df.loc[incidents_df.index]], axis=1)
    geo_df = geo_df.apply(lambda row: check_geographical_data_consistency(row, additional_data=counties_df), axis=1)
    incidents_df[geo_df.columns] = geo_df[geo_df.columns]
    save_checkpoint(incidents_df, 'checkpoint_1')

# %% [markdown]
# The function called above performs the following operations:
# 
# - converts to lowercase the attributes *state*, *county*, and *city*
# - if *city_or_county* contains values for both city and county, splits them into two different fields
# - removes from *city_or_county* the words 'city of' and 'county'
# - removes from *city_or_county* punctuation and numerical values
# - removes frequent words from *address* and *display_name* (e.g., "Street," "Avenue," "Boulevard")
# 
# When latitude and longitude are available and therefore Geopy provided information for the corresponding location:
# - checks for equality between *state* and *state_geopy*
# - checks for equality between *county* and *county_geopy* or between *county* and *suburb_geopy*
# - checks for equality between *city* and *city_geopy*, or between *city* and *town_geopy*, or between *city* and *village_geopy*
# 
# If these comparisons fail, it checks for potential typos in the string. This is done using the Damerau-Levenshtein distance (defined below), with a threshold to decide the maximum distance for two strings to be considered equal. The thresholds were set after several preliminary tests. We decided to use different thresholds for state and city or county.
# 
# The **Damerau-Levenshtein distance** between two strings $s$ and $t$ is define as\
# $D(i, j) = \min
# \begin{cases}
# D(i-1, j) + 1 \\
# D(i, j-1) + 1 \\
# D(i-1, j-1) + \delta \\
# D(i-2, j-2) + \delta & \text{if } s[i] = t[j] \text{ and } s[i-1] = t[j-1]
# \end{cases}$
# 
# where:
# - $D(i, j)$ is the Damerau-Levenshtein distance between the first $i$ letters of a string $s$ and the first $j$ letters of a string $t$
# - $\delta$ is 0 if the current letters $s[i]$ and $t[j]$ are equal, otherwise, it is 1
# - $D(i-2, j-2) + \delta$ represents transposition (swapping two adjacent letters) if the current letters $s[i]$ and $t[j]$ are equal, and the preceding letters $s[i-1]$ and $t[j-1]$ are also equal
# 
# If the comparison still fails, it compares the *address* field from our dataset with GeoPy's *display_name* (using again the Damerau-Levenshtein distance).
# 
# In case of a match with GeoPy data, we set the values for the fields *state*, *county*, *city*, *latitude*, *longitude*, *importance*, *address_type* to the corresponding fileds provided by GeoPy. Otherwise we check for matches with the Wikipedia data (using again the Damerau-Levenshtein distance).

# %%
tot_row = incidents_df.index.size
print('Number of rows with all null values: ', incidents_df.isnull().all(axis=1).sum(), ' / ', incidents_df.isnull().all(axis=1).sum()*100/tot_row, '%')
print('Number of rows with null value for state: ', incidents_df['state'].isnull().sum(), ' / ', incidents_df['state'].isnull().sum()*100/tot_row, '%')
print('Number of rows with null value for county: ', incidents_df['county'].isnull().sum(), ' / ', incidents_df['county'].isnull().sum()*100/tot_row, '%')
print('Number of rows with null value for city: ', incidents_df['city'].isnull().sum(), ' / ', incidents_df['city'].isnull().sum()*100/tot_row, '%')
print('Number of rows with null value for latitude: ', incidents_df['latitude'].isnull().sum(), ' / ', incidents_df['latitude'].isnull().sum()*100/tot_row, '%')
print('Number of rows with null value for longitude: ', incidents_df['longitude'].isnull().sum(), ' / ', incidents_df['longitude'].isnull().sum()*100/tot_row, '%')

# %%
sns.heatmap(incidents_df.isnull(), cbar=False, xticklabels=True);

# %% [markdown]
# Now all the entries in the dataset have at least the state value not null and consistent. Only 12,796 data points, which account for 5.34% of the dataset, were found to have inconsistent latitude and longitude values.

# %%
incidents_df.groupby(['state_consistency','county_consistency','address_consistency']).size().to_frame().rename(columns={0:'count'}).sort_index(ascending=False)

# %%
stats = {}
stats_columns = ['#null_val', '#not_null', '#value_count']
for col in ['state', 'county', 'city', 'latitude', 'longitude', 'state_consistency',
       'county_consistency', 'address_consistency', 'location_importance', 'address_type']:
    stats[col] = []
    stats[col].append(incidents_df[col].isna().sum())
    stats[col].append(len(incidents_df[col]) - incidents_df[col].isna().sum())
    stats[col].append(len(incidents_df[col].value_counts()))
    
clean_geo_stat_stats = pd.DataFrame(stats, index=stats_columns).transpose()
clean_geo_stat_stats

# %%
incidents_df[['latitude', 'county', 'city']].isna().groupby(['latitude', 'county', 'city']).size().to_frame().rename(columns={0:'count'})

# %%
incidents_df[['latitude']].isna().groupby(['latitude']).size().to_frame().rename(columns={0:'count'})

# %%
incidents_df_not_null = incidents_df[incidents_df['latitude'].notna()]
print('Number of entries with not null values for latitude and longitude: ', len(incidents_df_not_null))
plot_scattermap_plotly(incidents_df_not_null, 'state', zoom=2, title='Incidents distribution by state')

# %%
incidents_df_nan_county = incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].isna()) & 
    (incidents_df['city'].notna())]
print('Number of entries with not null values for county but not for lat/lon and city: ', len(incidents_df_nan_county))
plot_scattermap_plotly(incidents_df_nan_county, 'state', zoom=2, title='Missing county')

# %%
incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].isna()) & (incidents_df['city'].notna())].groupby('city').count()

# %%
incidents_df[(incidents_df['latitude'].notna()) & (incidents_df['city'].isna()) & (incidents_df['county'].isna())]

# %%
incidents_df_nan_city = incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].notna()) & (incidents_df['city'].isna())]
print('Number of rows with null values for city, but not for lat/lon and county: ', len(incidents_df_nan_city))
plot_scattermap_plotly(incidents_df_nan_city, 'state', zoom=2, title='Missing city')

# %% [markdown]
# **Final considerations**
# 
# From this analysis we found that:
# - 174,796 entries are fully consistent
# - in 26,635 entries only the city is missing, but it can be inferred from latitude and longitude
# - in 15,000 entries only the county is missing, but it can be inferred from latitude and longitude
# - in 33 entries both the city and county are missing. Even in this group, the missing information can be inferred from latitude and longitude, as they are all closely clustered around Baltimore
# - in 3,116 entries latitude, longitude, and city are missing. They can be inferred (though not perfectly) from the county-state pair
# - in 19,844 entries only the state field is present
# 
# The dataset does not contain any other combinations beyond the ones just mentioned.
# 
# In the following we will recover the missing information using the latitude and longitude values.

# %% [markdown]
# #### City attribute

# %% [markdown]
# To recover missing values for the attribute *city* when *latitude* and *longitude* are available we will compute the closest city centroid and assign the corresponding city if the distance is below a certain threshold.

# %% [markdown]
# First we Compute the centroid for each city and visualize the first 10 centroids in lexicographic order:

# %%
centroids = incidents_df.loc[incidents_df['latitude'].notna() & incidents_df['city'].notna()][[
    'latitude', 'longitude', 'city', 'state', 'county']].groupby(['state', 'county', 'city']).mean()
centroids.head(10)

# %%
print('Number of distinct cities:', len(centroids.index.to_list()))

# %% [markdown]
# For each tuple <state, county, city> in 'centroids', we extract the corresponding latitude and longitude coordinates from the 'clean_geo_data' DataFrame. We then compute the distance between these coordinates and the centroids using the geodesic distance (in kilometers). We also compute percentiles (at 0.05 intervals), maximum, minimum, and average distances of the points in each city.

# %%
info_city = pd.DataFrame(columns=['5', '15', '25', '35', '45', '55', '65', '75', '85', '95',
    'tot_points', 'min', 'max', 'avg', 'centroid_lat', 'centroid_lon'], index=centroids.index)
info_city.head(2)

if LOAD_DATA_FROM_CHECKPOINT: # load data
    info_city = load_checkpoint('checkpoint_cities')
else: # compute data
    for state, county, city in centroids.index:
        dummy = []
        for lat, long in zip(incidents_df.loc[(incidents_df['city'] == city) & 
            (incidents_df['state'] == state) & (incidents_df['county'] == county) & 
            incidents_df['latitude'].notna()]['latitude'], 
            incidents_df.loc[(incidents_df['city'] == city) & 
            (incidents_df['state'] == state) & (incidents_df['county'] == county) & 
            incidents_df['longitude'].notna()]['longitude']):
            dummy.append(geopy_distance.geodesic([lat, long], centroids.loc[state, county, city]).km)
            
        dummy = sorted(dummy)
        pc = np.quantile(dummy, np.arange(0, 1, 0.05))
        for i in range(len(info_city.columns) - 6):
            info_city.loc[state, county, city][i] = pc[i*2 + 1]
        info_city.loc[state, county, city][len(info_city.columns) - 6] = len(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 5] = min(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 4] = max(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 3] = sum(dummy)/len(dummy)
        info_city.loc[state, county, city][len(info_city.columns) - 2] = centroids.loc[state, county, city]['latitude']
        info_city.loc[state, county, city][len(info_city.columns) - 1] = centroids.loc[state, county, city]['longitude']
    save_checkpoint(info_city, 'checkpoint_cities') # save data 

# %% [markdown]
# We display the resulting dataframe and plot it on a map:

# %%
info_city.head()

# %%
info_city.loc[info_city['tot_points'] > 1].info()

# %%
fig = plot_scattermap_plotly(
    info_city,
    size='tot_points',
    x_column='centroid_lat',
    y_column='centroid_lon',
    hover_name=False, zoom=2,
    title='Number of incidents per city'
)
fig.show()
pyo.plot(fig, filename='../html/incidents_per_city.html', auto_open=False);

# %% [markdown]
# We utilize the previously calculated data to infer missing values for the *city* field in entries of the dataset where latitude and longitude are available. The *city* field is assigned if the distance of the entry from the centroid falls within the third quartile of all points assigned to that centroid.

# %%
def substitute_city(row, info_city):
    if pd.isna(row['city']) and not np.isnan(row['latitude']):
        for state, county, city in info_city.index:
            if row['state'] == state and row['county'] == county:
                if info_city.loc[state, county, city]['tot_points'] > 1:
                    max_radius = info_city.loc[state, county, city]['75'] # terzo quantile
                    centroid_coord = [info_city.loc[state, county, city]['centroid_lat'], 
                        info_city.loc[state, county, city]['centroid_lon']]
                    if (geopy_distance.geodesic([row['latitude'], row['longitude']], centroid_coord).km <= 
                        max_radius):
                        row['city'] = city
                        break
                    
    return row

# %%
if LOAD_DATA_FROM_CHECKPOINT:
    new_incidents_df = load_checkpoint('checkpoint_2', date_cols=['date', 'date_original'])
else:
    new_incidents_df = incidents_df.apply(lambda row: substitute_city(row, info_city), axis=1)
    save_checkpoint(incidents_df, 'checkpoint_2')

# %%
n_nan_cities = incidents_df['city'].isnull().sum()
n_non_nan_cities = new_incidents_df['city'].isnull().sum()
print('Number of rows with null values for city before the inference: ', n_nan_cities)
print('Number of rows with null values for city after the inference: ', n_non_nan_cities)
print(f'Number of city recovered: {n_nan_cities - n_non_nan_cities}')

# %%
incidents_df_nan_city = new_incidents_df.loc[(incidents_df['latitude'].notna()) & (incidents_df['county'].notna()) & (incidents_df['city'].isna())]
print('Number of rows with null values for city, but not for lat/lon and county: ', len(incidents_df_nan_city))
plot_scattermap_plotly(incidents_df_nan_city, 'city', zoom=2, title='City inferred')

# %%
incidents_df = new_incidents_df
incidents_df[['latitude', 'county', 'city']].isna().groupby(['latitude', 'county', 'city']).size().to_frame().rename(columns={0:'count'})

# %% [markdown]
# #### Districts attributes

# %% [markdown]
# We check if the attribute `congressional_district` is numbered consistently (with '0' for states with only one congressional district). To do so we use the dataset containing the data about elections in the period of interest (congressional districts are redrawn when (year%10)==0):

# %%
at_large_states = elections_df[
    (elections_df['year'].between(2013, 2018, inclusive="both")) & 
    (elections_df['congressional_district']==0)
    ]['state'].unique()
at_large_states

# %% [markdown]
# Now we check if states with a '0' as congressional district are the same states with only one congressional district in the dataset containing the data about elections:

# %%
zero_congress_states_inc = incidents_df[incidents_df['congressional_district']==0]['state'].unique()
set(zero_congress_states_inc).issubset(set(at_large_states))

# %% [markdown]
# We check if states with a single congressional district are always numbered with '0' in the dataset containing the data about elections:

# %%
incidents_df[(incidents_df['state'] == at_large_states.any()) & (incidents_df['congressional_district']!=0)].size==0

# %% [markdown]
# Since they are not, we fix this issue:

# %%
incidents_df.loc[incidents_df['state'].isin(at_large_states), 'congressional_district'] = 0

# %% [markdown]
# We check if the range of the attribute `congressional_district` is consistent with the number of congressional districts in the dataset containing the data about elections:

# %%
incidents_df['state'] = incidents_df['state'].str.upper()
wrong_congr_states = elections_df.groupby('state')['congressional_district'].max()>=incidents_df.groupby('state')['congressional_district'].max()
for state in wrong_congr_states[wrong_congr_states==False].index:
    print(f"State {state} has more districts in the incidents data than in the elections data")

# %% [markdown]
# We display the rows with inconsistent congressional district in Kentucky:

# %%
incidents_df[
    (incidents_df['state']=='KENTUCKY') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='KENTUCKY') & (elections_df['year']>2012)]['congressional_district'].max())
]

# %% [markdown]
# Searching online we found that Kentucky, in that period, had 6 congressional districts, so we'll set to nan the congressional district for the row above:

# %%
incidents_df.loc[
    (incidents_df['state']=='KENTUCKY') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='KENTUCKY') & (elections_df['year']>2012)]['congressional_district'].max()),
    'congressional_district'] = np.nan

# %% [markdown]
# We display the rows with inconsistent congressional district in Oregon:

# %%
incidents_df[
    (incidents_df['state']=='OREGON') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='OREGON') & (elections_df['year']>2012)]['congressional_district'].max())
]

# %% [markdown]
# Searching online we found that Oregon, in that period, had 5 congressional districts, so we'll set to nan the congressional district for the rows above:

# %%
incidents_df.loc[
    (incidents_df['state']=='OREGON') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='OREGON') & (elections_df['year']>2012)]['congressional_district'].max()),
    'congressional_district'] = np.nan 

# %% [markdown]
# We display the rows with inconsistent congressional district in West Virginia:

# %%
incidents_df[
    (incidents_df['state']=='WEST VIRGINIA') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='WEST VIRGINIA') & (elections_df['year']>2012)]['congressional_district'].max())
]

# %% [markdown]
# Searching online we found that West Virginia, in that period, had 3 congressional districts, so we'll set to nan the congressional district for the row above:

# %%
incidents_df.loc[
    (incidents_df['state']=='WEST VIRGINIA') &
    (incidents_df['congressional_district'] > 
        elections_df[(elections_df['state']=='WEST VIRGINIA') & (elections_df['year']>2012)]['congressional_district'].max()),
    'congressional_district'] = np.nan

# %% [markdown]
# We check whether given a certain value for the attributes `latitude` and `longitude`, the attribute `congressional_district` has always the same value:

# %%
incidents_df[incidents_df['congressional_district'].notnull()].groupby(['latitude', 'longitude'])['congressional_district'].unique()[lambda x: x.str.len() > 1].to_frame().rename(columns={0:'count'}).sample(5, random_state=1)

# %% [markdown]
# All these points are probably errors, due to the fact that they are near the border between two congressional districts. We correct them setting the most frequent value for the attribute `congressional_district` (setting that value also for the entries with missing values):

# %%
corrected_congr_districts = incidents_df[
    ~incidents_df['congressional_district'].isna()
    ].groupby(['latitude', 'longitude'])['congressional_district'].agg(lambda x: x.value_counts().index[0])
incidents_df = incidents_df.merge(corrected_congr_districts, on=['latitude', 'longitude'], how='left')
# where latitude and longitude are null, keep the original value
incidents_df['congressional_district_y'].fillna(incidents_df['congressional_district_x'], inplace=True)
incidents_df.rename(columns={'congressional_district_y':'congressional_district'}, inplace=True)
incidents_df.drop(columns=['congressional_district_x'], inplace=True)

# %% [markdown]
# In the same city or county there could be different values for the attribute `congressional_district` (this is not an error, is actually possible according to the USA law):

# %%
incidents_df[incidents_df['congressional_district'].notna()].groupby(['state', 'city_or_county'])['congressional_district'].unique()[lambda x: x.str.len() > 1].to_frame()

# %% [markdown]
# We print the unique values the attribute `state_house_district` can take on:

# %%
house_districts = incidents_df['state_house_district'].unique()
house_districts.sort()
house_districts

# %% [markdown]
# Also this attribute has some errors because the maximum number of state house districts should be 204 (for New Hampshire, see [here](https://ballotpedia.org/State_Legislative_Districts)). For now we won't correct this error beacuse this attribute is not useful for our analysis.
# 
# We check if given a certain value for the attributes `latitude` and `longitude`, the attribute `state_house_district` has always the same value:

# %%
incidents_df[incidents_df['state_house_district'].notnull()].groupby(
    ['latitude', 'longitude'])['state_house_district'].unique()[lambda x: x.str.len() > 1].to_frame()

# %% [markdown]
# We correct the errors:

# %%
corrected_house_districts = incidents_df[
    incidents_df['state_house_district'].notnull()
    ].groupby(['latitude', 'longitude'])['state_house_district'].agg(lambda x: x.value_counts().index[0])
incidents_df = incidents_df.merge(corrected_house_districts, on=['latitude', 'longitude'], how='left')
incidents_df['state_house_district_y'].fillna(incidents_df['state_house_district_x'], inplace=True)
incidents_df.rename(columns={'state_house_district_y':'state_house_district'}, inplace=True)
incidents_df.drop(columns=['state_house_district_x'], inplace=True)

# %% [markdown]
# We now print the unique values the attribute `state_senate_district` can take on:

# %%
senate_districts = incidents_df['state_senate_district'].unique()
senate_districts.sort()
senate_districts

# %% [markdown]
# And again we notice some errors because the maximum number of state senate districts should be 67 (for Minnesota, see [here](https://ballotpedia.org/State_Legislative_Districts)). For now we won't correct this error beacuse this attribute is not useful for our analysis.
# 
# We correct other possible errors as above:

# %%
corrected_senate_districts = incidents_df[
    incidents_df['state_senate_district'].notnull()
    ].groupby(['latitude', 'longitude'])['state_senate_district'].agg(lambda x: x.value_counts().index[0])
incidents_df = incidents_df.merge(corrected_senate_districts, on=['latitude', 'longitude'], how='left')
incidents_df['state_senate_district_y'].fillna(incidents_df['state_senate_district_x'], inplace=True)
incidents_df.rename(columns={'state_senate_district_y':'state_senate_district'}, inplace=True)
incidents_df.drop(columns=['state_senate_district_x'], inplace=True)

# %% [markdown]
# We check whether given a `state`, `city_or_county` and `state_senate_district`, the value of the attribute `congressional_district` is always the same:

# %%
incidents_df[incidents_df['congressional_district'].notnull()].groupby(
    ['state', 'city_or_county', 'state_senate_district'])['congressional_district'].unique()[lambda x: x.str.len() > 1].shape[0]==0

# %% [markdown]
# Hence we cannot recover the missing values for the attribute `congressional_district` from the values of `state_senate_district`. We check the same for the attribute `state_house_district`:

# %%
incidents_df[incidents_df['congressional_district'].notnull()].groupby(
    ['state', 'city_or_county', 'state_house_district'])['congressional_district'].unique()[lambda x: x.str.len() > 1].shape[0]==0

# %% [markdown]
# We cannot recover the missing values for the attribute `congressional_district` from the values of `state_house_district` either.
# 
# We could, instead, recover the missing values from the entries with "similar" `latitude` and `longitude`. To explore this possibility we first plot on a map the dislocation of the incidents, coloring them according to the value of the attribute `congressional_district`:

# %%
plot_scattermap_plotly(
    incidents_df,
    'congressional_district',
    black_nan=True,
    zoom=2,
    height=800,
    width=800,
    title="USA Congressional districts"
    )

# %% [markdown]
# Many points with missing `congressional_district` are often "surrounded" by points belonging to the same congressional district. We could, therefore, use KNN classifier to recover those values.
# 
# We'll do this first for the state of Alabama, showing the results with some plots. Later we will do the same for all the other states. 
# 
# We plot the distribution of the values of the attribute `congressional_district` for the state of Alabama:

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['state']=='ALABAMA'],
    attribute='congressional_district',
    black_nan=True,
    width=500,
    height=600,
    zoom=5.5,
    title="Alabama incidents by Congressional Districts",
    legend_title="Congressional District"
)

# %% [markdown]
# We define a function to prepare the data for the classification task:

# %%
def build_X_y_for_district_inference(incidents_df):
    X_train = np.concatenate((
        incidents_df[
            (incidents_df['congressional_district'].notna()) &
            (incidents_df['latitude'].notna()) & 
            (incidents_df['longitude'].notna())
            ]['latitude'].values.reshape(-1, 1),
        incidents_df[
            (incidents_df['congressional_district'].notna()) & 
            (incidents_df['latitude'].notna()) & 
            (incidents_df['longitude'].notna())
            ]['longitude'].values.reshape(-1, 1)),
        axis=1
    )
    X_test = np.concatenate((
        incidents_df[
            (incidents_df['congressional_district'].isna()) & 
            (incidents_df['latitude'].notna()) & 
            (incidents_df['longitude'].notna())
            ]['latitude'].values.reshape(-1, 1),
        incidents_df[
            (incidents_df['congressional_district'].isna()) &
            (incidents_df['latitude'].notna()) & 
            (incidents_df['longitude'].notna())
            ]['longitude'].values.reshape(-1, 1)),
        axis=1
    )
    y_train = incidents_df[
        (incidents_df['congressional_district'].notna()) & 
        (incidents_df['latitude'].notna()) & 
        (incidents_df['longitude'].notna())
        ]['congressional_district'].values
    return X_train, X_test, y_train

# %% [markdown]
# We define the function to compute the geodesic distance to pass to the KNN classifier:

# %%
def geodesic_distance(point1, point2):
    return geopy_distance.geodesic(point1, point2).km

# %% [markdown]
# Now we are ready to apply the classifier (using K=1):

# %%
X_train, X_test, y_train = build_X_y_for_district_inference(incidents_df[incidents_df['state']=="ALABAMA"])
knn_clf = KNeighborsClassifier(n_neighbors=1, metric=geodesic_distance)
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
incidents_df['KNN_congressional_district'] = incidents_df['congressional_district']
incidents_df.loc[
    (incidents_df['state']=="ALABAMA") &
    (incidents_df['congressional_district'].isna()) &
    (incidents_df['latitude'].notna()) & 
    (incidents_df['longitude'].notna()),
    'KNN_congressional_district'
    ] = knn_pred

# %% [markdown]
# We plot the results:

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['state']=='ALABAMA'],
    attribute='KNN_congressional_district',
    width=500,
    height=600,
    zoom=5.5,
    title="Alabama incidents by Congressional Districts",
    legend_title="Congressional District"
)

# %% [markdown]
# To improve the visualization, we plot on the map the decision boundaries of the classifier. To do so, we convert latitude and longitude to a 2D space:

# %%
transformer = Transformer.from_crs("EPSG:4326", "EPSG:26929", always_xy=True) # EPSG:26929 identifies the projected coordinate system for Alabama East (had to choose between E,W,N,S)

X_train_converted = []

for i in range(X_train.shape[0]):
    x, y = transformer.transform(X_train[i][1], X_train[i][0])
    X_train_converted.append([x,y])

X_train_converted = np.array(X_train_converted)

# %% [markdown]
# And now we train the classifier using the euclidean distance:

# %%
knn_eu_clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn_eu_clf.fit(X_train_converted, y_train)

# %% [markdown]
# We plot the boundaries of the classifier:

# %%
alabama_color_map = {
    1:'red',
    2:'orange',
    3:'yellow',
    4:'green',
    5:'lightblue',
    6:'blue',
    7:'purple'
}
plot_clf_decision_boundary(knn_eu_clf, X_train_converted, y_train, alabama_color_map, "KNN Alabama borders")

# %% [markdown]
# We can now compare the boundaries built by the classifier with the actual boundaries (this map was taken [here](https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif/lossless-page1-1256px-United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif.png)):
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif/lossless-page1-1256px-United_States_Congressional_Districts_in_Alabama%2C_since_2013.tif.png" alt="Alt Text" width="600"/>

# %% [markdown]
# The result is satisfactory. However, it is important to highlight that if there are no examples available for a specific district, we won't assign the correct label to the points in that districts. We check how many congressional districts have 2 or less examples:

# %%
incidents_df.groupby(['state', 'congressional_district']).size()[lambda x: x <= 2]

# %% [markdown]
# By the way, missclassification can still occurr, depending on the position of the available examples w.r.t the position of the points to classify. Aware of this limitation, we proceed to apply this method to the other states and plot the result:

# %%
if LOAD_DATA_FROM_CHECKPOINT:
    incidents_df = load_checkpoint('checkpoint_3', date_cols=['date', 'date_original'])
else:
    for state in incidents_df['state'].unique():
        if state != "ALABAMA":
            print(f"{state} done.")
            X_train, X_test, y_train = build_X_y_for_district_inference(incidents_df[incidents_df['state']==state])
            if X_test.shape[0] == 0:
                continue
            knn_clf.fit(X_train, y_train)
            knn_pred = knn_clf.predict(X_test)
            incidents_df.loc[
                (incidents_df['state']==state) &
                (incidents_df['congressional_district'].isna()) &
                (incidents_df['latitude'].notna()) & 
                (incidents_df['longitude'].notna()),
                'KNN_congressional_district'
            ] = knn_pred
    incidents_df.drop(columns=['congressional_district'], inplace=True)
    incidents_df.rename(columns={'KNN_congressional_district':'congressional_district'}, inplace=True)
    save_checkpoint(incidents_df, 'checkpoint_3')

plot_scattermap_plotly(
    incidents_df,
    'congressional_district',
    zoom=2,
    height=800,
    width=800,
    title="USA Congressional districts (after inference)"
)

# %% [markdown]
# We now plot on a map the location of the incidents, coloring them according to the value of the attribute `state_senate_district` and `state_house_district`, to assess wheter we can apply the same method to recover missing values:

# %%
plot_scattermap_plotly(
    incidents_df,
    'state_senate_district',
    black_nan=True,
    zoom=2,
    height=800,
    width=800,
    title="USA State senate districts"
    )

plot_scattermap_plotly(
    incidents_df,
    'state_house_district',
    black_nan=True,
    zoom=2,
    height=800,
    width=800,
    title="USA State house districts"
    )

# %% [markdown]
# These attributes have a lot of missing values, sometimes spread over large areas where there are no other points. Given this scarcity of training examples, we cannot apply the same method to recover the missing values.
# 
# Finally we visualize the most frequent addresses:

# %%
incidents_df.groupby(['address']).size().sort_values(ascending=False)[:50].plot(
    kind='bar',
    figsize=(10,6),
    title='Counts of the addresses with the 50 highest number of incidents'
);

# %% [markdown]
# Many of the most frequent addresses are located in airports.

# %% [markdown]
# ### Age, gender and number of participants: exploration and preparation

# %% [markdown]
# We display a concise summary of the attributes related to age, gender and number of participants:

# %%
participants_columns = ['participant_age1', 'participant_age_group1', 'participant_gender1', 
    'min_age_participants', 'avg_age_participants', 'max_age_participants',
    'n_participants_child', 'n_participants_teen', 'n_participants_adult', 
    'n_males', 'n_females',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants']
age_df = incidents_df[participants_columns]
age_df.info()

# %%
age_df['participant_age_group1'].unique()

# %% [markdown]
# We check if we have entries with non-null values for participant_age1 but NaN for participant_age_group1. 

# %%
age_df[age_df['participant_age1'].notna() & age_df['participant_age_group1'].isna()]

# %% [markdown]
# These 126 values can be inferred.

# %% [markdown]
# Below, we provide a brief summary of the operations we performed to correct missing and inconsistent values.

# %% [markdown]
# First of all, we converted all the consistent values to integers. All the out of range values (e.g. nagative values or improbable ages) were set to *NaN*.
# We considered the maximum possible age to be 122 years, as it is the age reached by [Jeanne Louise Calment](https://www.focus.it/scienza/scienze/longevita-vita-umana-limite-biologico#:~:text=Dal%201997%2C%20anno%20in%20cui,ha%20raggiunto%20un%20limite%20biologico), the world's oldest person.

# %% [markdown]
# To identify inconsistencies in the data related to the minimum, maximum, average age of participants, and to the composition of the age groups we checked if:
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
# To identify inconsistencies in the data related to the number of participants we checked if:
# 
# - n_participants $\geq$ 0
# - n_participants $==$ n_males $+$ n_females
# - n_killed $+$ n_injured $\leq$ n_participants
# - n_arrested $\leq$ n_participants
# - n_unharmed $\leq$ n_participants

# %% [markdown]
# Values related to participant_age_group1 and participant_gender1 have been binarized using one-hot encoding, thus creating the boolean features *participant1_child*, *participant1_teen*, *participant1_adult*, *participant1_male*, *participant1_female*.
# 
# To identify other potential inconsistencies we did the following checks:
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
# We kept track of data consistency by using the following variables (the variables were set to *True* if data were consistent, *False* if they were not, or *NaN* when data was missing):
# - *consistency_age*: to track the consistency between the minimum, maximum, and average ages and the number of participants by age groups
# - *consistency_n_participant*: to track the consistency between the number of participants
# - *consistency_gender*: to track the consistency of the gender attribute
# - *consistency_participant1*: to track the consistency of the attributes related to participant1
# - *participant1_age_consistency_wrt_all_data*: to track the consistency between the age of participant1 and the other attributes related to ages
# - *participant1_age_range_consistency_wrt_all_data*: to track the consistency between the age group of participant1 and the other attributes related to age groups
# - *participant1_gender_consistency_wrt_all_data*: to track the consistency between the gender of participant1 and the other attributes related to gender
# - *consistency_participants1_wrt_n_participants*: to track the overall consistency between the data about participant1 and the other data
# 
# The chunck of code below performes the specified checks:

# %%
from TASK_1.data_preparation_utils import check_age_gender_data_consistency

if LOAD_DATA_FROM_CHECKPOINT: # load data
    age_temporary_df = load_checkpoint('checkpoint_tmp')#, date_cols=['date', 'date_original']) #TODO: rimette uniforme ad altri
else: # compute data
    age_temporary_df = age_df.apply(lambda row: check_age_gender_data_consistency(row), axis=1)
    save_checkpoint(age_temporary_df, 'checkpoint_tmp') # save data

# %% [markdown]
# In the following, we will display and visualize statistics about the data consistency.

# %%
age_temporary_df.info()

# %%
age_temporary_df[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].describe()

# %%
print('Number of rows with null values: ', age_temporary_df[age_temporary_df['nan_values'] == True].shape[0])
print('Number of rows with inconsistent values in age data: ', age_temporary_df[age_temporary_df['consistency_age'] == False].shape[0])
print('Number of rows with inconsistent values in number of participants data: ', age_temporary_df[age_temporary_df[
    'consistency_n_participant'] == False].shape[0])
print('Number of rows with inconsistent values in gender data: ', age_temporary_df[age_temporary_df['consistency_gender'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 data: ', age_temporary_df[age_temporary_df[
    'consistency_participant1'] == False].shape[0])
print('Number of rows with inconsistent values for participants1: ', age_temporary_df[age_temporary_df[
    'consistency_participant1'] == False].shape[0])
print('Number of rows with NaN values for participants1: ', age_temporary_df[age_temporary_df[
    'consistency_participant1'] == np.nan].shape[0])
print('Number of rows with inconsistent values in participants1 wrt all other data: ', age_temporary_df[age_temporary_df[
    'consistency_participants1_wrt_n_participants'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age data: ', age_temporary_df[age_temporary_df[
    'participant1_age_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age range data: ', age_temporary_df[age_temporary_df[
    'participant1_age_range_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt gender data: ', age_temporary_df[age_temporary_df[
    'participant1_gender_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with null values in age data: ', age_temporary_df[age_temporary_df['consistency_age'].isna()].shape[0])
print('Number of rows with null values in number of participants data: ', age_temporary_df[age_temporary_df[
    'consistency_n_participant'].isna()].shape[0])
print('Number of rows with null values in gender data: ', age_temporary_df[age_temporary_df['consistency_gender'].isna()].shape[0])
print('Number of rows with null values in participants1 data: ', age_temporary_df[age_temporary_df[
    'consistency_participant1'].isna()].shape[0])
print('Number of rows with all null data: ', age_temporary_df.isnull().all(axis=1).sum())

# %% [markdown]
# We notice that:
# - The data in our dataset related to participant1, excluding the 1099 cases where age and age group data were inconsistent with each other and 190 cases where age range is not consistent, always appear to be consistent with the data in the rest of the dataset and can thus be used to fill in missing values or correct data
# - In the data related to age and gender, some inconsistencies are present, but they account for only 1.88% and 6.01% of the total dataset rows, respectively
# - In 93779 rows, at least one field had a *NaN* value

# %% [markdown]
# We plot the age distribution of participant1 and compare it to the distribution of the minimum and maximum participants' age for each group:

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

ax0.hist(age_temporary_df['participant_age1'], bins=100, edgecolor='black', linewidth=0.8)
ax0.set_xlabel('Age')
ax0.set_ylabel('Frequency')
ax0.set_title('Distribution of age participant1')

ax1.hist(age_temporary_df['min_age_participants'], bins=100, edgecolor='black', linewidth=0.8)
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of min age participants')

ax2.hist(age_temporary_df['max_age_participants'], bins=100, edgecolor='black', linewidth=0.8)
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of max age participants')

plt.show()

# %% [markdown]
# The similar shapes of the distributions provides confirmation that the data pertaining to participant1 is accurate and reliable. Therefore, we can confidently use participant1's data to fill gaps in cases incidents involved a single participant.

# %% [markdown]
# We visualize the number of unique values for the cardinality of participants in each incident and provided a brief summary of this feature:

# %%
print('Values of n_participants: ', age_temporary_df['n_participants'].unique())
age_temporary_df['n_participants'].describe().to_frame()

# %% [markdown]
# We visualize the distribution of the number of participants for each incident using a log scale:

# %%
plt.figure(figsize=(20, 5))
plt.bar(incidents_df.groupby('n_participants')['n_participants'].count().index, incidents_df.groupby('n_participants')['n_participants'].count(),
    alpha=0.8, edgecolor='black', linewidth=0.8)
plt.yscale('log')
plt.xlabel('Number of participants for incidents')
plt.ylabel('Number of incidents')
plt.plot([0.5, 103.5], [1, 1], '--', color='magenta', label='1 incident')
plt.plot([0.5, 103.5], [2, 2], '--', color='red', label='2 incidents')
plt.plot([0.5, 103.5], [10, 10], '--', color='green', label='10 incidents')
plt.plot([0.5, 103.5], [100, 100], '--', color='blue', label='100 incidents')
plt.xticks(range(1, 104, 2), range(1, 104, 2))
plt.legend()
plt.show()

# %% [markdown]
# We will now correct inconsistencies in the following manner:
# - when having the number of males (n_males) and number of females (n_females), we set the total number of participants as n_participants = n_males + n_females
# - when having a single participant and consistent data for *participants1*, we use that data to set the attributes related to age (max, min, average) and gender

# %%
from TASK_1.data_preparation_utils import set_gender_age_consistent_data

if LOAD_DATA_FROM_CHECKPOINT:
    with zipfile.ZipFile('checkpoints/checkpoint_4.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('checkpoints/')
    incidents_df = load_checkpoint('checkpoint_4', date_cols=['date', 'date_original'])
else:
    new_age_df = age_temporary_df.apply(lambda row: set_gender_age_consistent_data(row), axis=1)
    incidents_df[new_age_df.columns] = new_age_df[new_age_df.columns]
    save_checkpoint(incidents_df, 'checkpoint_4')

# %% [markdown]
# We visualize the data after the corrections:

# %%
incidents_df.sample(2, random_state=1)

# %%
incidents_df.info()

# %%
print('Number of rows in which all data are null: ', incidents_df.isnull().all(axis=1).sum())
print('Number of rows with some null data: ', incidents_df.isnull().any(axis=1).sum())
print('Number of rows in which number of participants is null: ', incidents_df[incidents_df['n_participants'].isnull()].shape[0])
print('Number of rows in which number of participants is 0: ', incidents_df[incidents_df['n_participants'] == 0].shape[0])
print('Number of rows in which number of participants is null and n_killed is not null: ', incidents_df[
    incidents_df['n_participants'].isnull() & incidents_df['n_killed'].notnull()].shape[0])
print('Total rows with null value for n_participants: ', incidents_df['n_participants'].isnull().sum())
print('Total rows with null value for n_participants_child: ', incidents_df['n_participants_child'].isnull().sum())
print('Total rows with null value for n_participants_teen: ', incidents_df['n_participants_teen'].isnull().sum())
print('Total rows with null value for n_participants_adult: ', incidents_df['n_participants_adult'].isnull().sum())
print('Total rows with null value for n_males: ', incidents_df['n_males'].isnull().sum())
print('Total rows with null value for n_females: ', incidents_df['n_females'].isnull().sum())

# %%
sns.heatmap(incidents_df.isnull(), cbar=False)

# %% [markdown]
# We recovered all the data related to age and gender. In 98973 entries, at most a value is missing.

# %% [markdown]
# We now explore the distribution of the total number of participants and the number of participants per age group. Once again we use a logaritmic scale in the y-axis:

# %%
def plot_hist(df_column, n_bin=100, density=True, title=None, y_label=None, color=None, y_logscale=False):
    
    def iqr_fence(x):
        q1 = x.quantile(0.25)
        med = x.quantile(0.5)
        q3 = x.quantile(0.75)
        IQR = q3 - q1
        u = x.max()
        l = x.min()
        Lower_Fence = builtins.max(builtins.min(x[x > q1 - (1.5 * IQR)], default=pd.Timestamp.min), l)
        #Lower_Fence = builtins.max(q1 - (1.5 * IQR), l)
        Upper_Fence = builtins.min(builtins.max(x[x < q3 + (1.5 * IQR)], default=pd.Timestamp.max), u)
        #Upper_Fence = builtins.min(q3 + (1.5 * IQR), u)
        return [Lower_Fence, q1, med, q3, Upper_Fence]
    relevant_positions = iqr_fence(df_column)
    n_items = len(df_column.index)

    fig, axs = plt.subplots(3, sharex=True, figsize=(14, 6))
    fig.suptitle(title)

    # fixed bin
    axs[0].hist(df_column, bins=n_bin, density=density, color=color)
    axs[0].set_ylabel(str(n_bin) + ' bin')
    axs[0].grid(axis='y')
    if y_logscale:
        axs[0].set_yscale('log')

    # number of bins computed using Sturge's rule
    n_bin = int(1 + math.log2(n_items))
    axs[1].hist(df_column, bins=n_bin, density=density, color=color)
    axs[1].set_ylabel("Sturge\'s rule binning")
    if y_logscale:
        axs[1].set_yscale('log')
    axs[1].grid(axis='y')

    axs[2].boxplot(x=df_column.dropna().values, labels=[''], vert=False)
    axs[2].set_xlabel(y_label)

    for i in range(2):
        axs[i].axvline(x = relevant_positions[0], color = 'black', linestyle = '--', alpha=0.75)
        axs[i].axvline(x = relevant_positions[1], color = 'black', linestyle = '-.', alpha=0.75)
        axs[i].axvline(x = relevant_positions[2], color = 'black', linestyle = '-.', alpha=0.75)
        axs[i].axvline(x = relevant_positions[3], color = 'black', linestyle = '-.', alpha=0.75)
        axs[i].axvline(x = relevant_positions[4], color = 'black', linestyle = '--', alpha=0.75)
    
    return fig

# %%
plot_hist(incidents_df['n_participants'], title='Distribution of number of participants', n_bin=104, y_label='n_participants', density=False, y_logscale=True);

# %%
plt.figure(figsize=(20, 5))
plt.hist(incidents_df['n_participants'], bins=104, edgecolor='black', linewidth=0.8)
plt.xlabel('Number of participants')
plt.ylabel('Frequency (log scale)')
plt.xticks(np.arange(1, 104, 2))
plt.yscale('log')
plt.title('Distribution of number of participants')
plt.show()

# %%
incidents_df[['n_participants', 'n_participants_child', 'n_participants_teen', 'n_participants_adult']].max().to_frame().rename(columns={0:'max value'})

# %%
incidents_df[incidents_df['n_participants_adult'] > 60][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(20, 12), sharex=True, sharey=True)

ax0.bar(incidents_df['n_participants_child'].value_counts().index, incidents_df['n_participants_child'].value_counts(),
    alpha=0.8, color='magenta', edgecolor='black', linewidth=0.8, label='Children')
ax0.legend()
ax1.bar(incidents_df['n_participants_teen'].value_counts().index, incidents_df['n_participants_teen'].value_counts(),
    alpha=0.8, color='red', edgecolor='black', linewidth=0.8, label='Teen')
ax1.legend()
ax2.bar(incidents_df['n_participants_adult'].value_counts().index, incidents_df['n_participants_adult'].value_counts(),
    color='orange', edgecolor='black', linewidth=0.8, label='Adult')
ax2.legend()

plt.xlim(-1, 64)
plt.xticks(range(0, 64))
plt.yscale('log')
plt.xlabel('Number of participants')
ax0.set_ylabel('Number of incidents')
ax1.set_ylabel('Numer of incidents')
ax2.set_ylabel('Numer of incidents')
ax0.set_title('Number of participants for each incident per age')
plt.show()

# %% [markdown]
# We observe that in incidents involving children and teenagers under the age of 18, the total number of participants is smaller than 7 and 27, respectively. In general, incidents involving a single person are much more frequent than other incidents, and most often, they involve teenagers and children, with a smaller percentage involving adults. On the other hand, incidents with more than one participant mostly consist of adults, and as the number of participants increases, the frequency of such incidents decreases.

# %% [markdown]
# We also plot the distribution of the number of incidents per gender:

# %%
plt.figure(figsize=(20, 5))
plt.bar(incidents_df['n_males'].value_counts().index-0.2, incidents_df['n_males'].value_counts(), 0.4,
    edgecolor='black', linewidth=0.8, label='Males participants')
plt.bar(incidents_df['n_females'].value_counts().index+0.2, incidents_df['n_females'].value_counts(), 0.4,
    edgecolor='black', linewidth=0.8, label='Females participants')
plt.xticks(range(0, 64))
plt.yscale('log')
plt.xlabel('Number of participants')
plt.ylabel('Number of incidents')
plt.legend()
plt.title('Number of participants for each incident per gender')
plt.show()

# %% [markdown]
# Below, we plot the distribution of the average age of participants in each incident.

# %%
plot_hist(incidents_df['avg_age_participants'], y_label='avg_age_participants', density=False);

# %% [markdown]
# ### Incident characteristics features: exploration and preparation

# %% [markdown]
# We use a word cloud to display the most frequent words in the attribut notes:

# %%
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))

word_cloud_all_train = WordCloud(
    width=800,
    height=400,
    stopwords=stopwords_set,
    collocations=False,
    background_color='white'
    ).generate(' '.join(incidents_df[incidents_df['notes'].notna()]['notes'].tolist()));
word_cloud_all_train.to_svg()
plt.figure( figsize=(20,10) )
plt.imshow(word_cloud_all_train)
plt.axis('off');
plt.title('Word cloud of notes\n', fontsize=40, fontweight='bold');
plt.savefig("../html/word_cloud_notes.svg")

# %% [markdown]
# We check if given the first characteristic of a record, the second one is different

# %%
incidents_df[incidents_df['incident_characteristics1']==incidents_df['incident_characteristics2']].shape[0]==0

# %% [markdown]
# We plot the frequency of each characteristic:

# %%
# merge characteristics list
ch1_counts = incidents_df['incident_characteristics1'].value_counts()
ch2_counts = incidents_df['incident_characteristics2'].value_counts()
ch_counts = ch1_counts.add(ch2_counts, fill_value=0).sort_values(ascending=True)
ch_counts.to_frame()

# %%
fig = ch_counts.plot(kind='barh', figsize=(5, 18))
fig.set_xscale("log")
plt.title("Counts of 'incident_characteristics'")
plt.xlabel('Count')
plt.ylabel('Incident characteristics')
plt.tight_layout()

# %%
ch1_females_counts = incidents_df[incidents_df['n_females']>1]['incident_characteristics1'].value_counts()
ch2_females_counts = incidents_df[incidents_df['n_females']>1]['incident_characteristics2'].value_counts()
ch_females_counts = ch1_females_counts.add(ch2_females_counts, fill_value=0).sort_values(ascending=False).plot(
    kind='bar',
    title='Characteristics counts of incidents with females involved',
    figsize=(20,10)
)

# %%
characteristics_count_matrix = pd.crosstab(incidents_df['incident_characteristics2'], incidents_df['incident_characteristics1'])
fig, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(characteristics_count_matrix, cmap='coolwarm', ax=ax, xticklabels=True, yticklabels=True, linewidths=.5)
ax.set_xlabel('incident_characteristics1')
ax.set_ylabel('incident_characteristics2')  
ax.set_title('Counts of incident characteristics')
plt.tight_layout()

# %%
characteristics_count_matrix[["Shot - Dead (murder, accidental, suicide)"]].sort_values(
    by="Shot - Dead (murder, accidental, suicide)",
    inplace=False,
    ascending=False).plot(
        kind='bar',
        figsize=(20,10)
    );

# %% [markdown]
# We defined the following binary tags to categorize the characteristics of each incident:
# - <b>firearm</b>: it tells if firearms were involved in the incident
# - <b>air_gun</b>: it tells if air guns were involved in the incident
# - <b>shots</b>: it tells if the incident involved shots
# - <b>aggression</b>: it tells if there was an aggression (both using a gun or not)
# - <b>suicide</b>: it tells if the incident involved a suicide (attempts are included)
# - <b>injuries</b>: it tells if one ore more subjects got injured
# - <b>death</b>: it tells if one ore more subjects died
# - <b>road</b>: it tells if the incident happened in a road
# - <b>illegal_holding</b>: it tells if the incident involved a stealing act or if a gun was illegaly possessed
# - <b>house</b>: it tells if the incident happened in a house
# - <b>school</b>: it tells if the incident happened next to a school
# - <b>children</b>: it tells if the incident involved one or more children
# - <b>drugs</b>: it tells if the incident involved drugs
# - <b>officers</b>: it tells if one or more officiers were involved in the incident
# - <b>organized</b>: it tells if the incident was planned by an organization or a group
# - <b>social_reasons</b>: it tells if the incident involved social discriminations or terrorism
# - <b>defensive</b>: it tells if there was a defensive use of a gun during the incident
# - <b>workplace</b>: it tells if the incident happened in a workplace
# - <b>abduction</b>: it tells if the incident involved any form of abduction
# - <b>unintentional</b>: it tells if the incident was unintentional
# 
# Each tag was set to True if and only if we had enough information to assume that the incident had that particular characteristic. 

# %% [markdown]
# We set all the tags and check their consistency w.r.t. the other data:

# %%
from TASK_1.data_preparation_utils import add_tags, check_tag_consistency, check_characteristics_consistency, IncidentTag

tags_columns = [tag.name for tag in IncidentTag]
tags_columns.append('tag_consistency')

if LOAD_DATA_FROM_CHECKPOINT:
    with zipfile.ZipFile('checkpoints/checkpoint_5.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('checkpoints/')
    incidents_df = load_checkpoint('checkpoint_5', date_cols=['date', 'date_original'])
else:
    incidents_df = add_tags(incidents_df)
    incidents_df['tag_consistency'] = True
    incidents_df = incidents_df.apply(lambda row: check_tag_consistency(row), axis=1)
    incidents_df = incidents_df.apply(lambda row: check_characteristics_consistency(row), axis=1)
    save_checkpoint(incidents_df, 'checkpoint_5')

# %%
incidents_df['tag_consistency'].value_counts().to_frame()

# %% [markdown]
# We correct the inconsistencies in the tag assuming the numerical data are consistent and we save again the dataset.

# %%
from TASK_1.data_preparation_utils import set_tags_consistent_data

if LOAD_DATA_FROM_CHECKPOINT:
    with zipfile.ZipFile('checkpoints/checkpoint_6.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('checkpoints/')
    incidents_df = load_checkpoint('checkpoint_6', date_cols=['date', 'date_original'])
else:
    for index, record in incidents_df.iterrows():
        if record['tag_consistency'] == False:
            incidents_df.loc[index] = set_tags_consistent_data(record)
            incidents_df.loc[index, 'tag_consistency'] = True # set to true and then check again if it's still not consistent
    incidents_df = incidents_df.apply(lambda row: check_tag_consistency(row), axis=1)
    incidents_df = incidents_df.apply(lambda row: check_characteristics_consistency(row), axis=1)
    save_checkpoint(incidents_df, 'checkpoint_6')

# %%
pd.DataFrame(data=incidents_df.dtypes).T

# %%
incidents_df['tag_consistency'].value_counts().to_frame()

# %% [markdown]
# We display the frequencies of the tags grouping them:

# %%
tags_counts = {}
tags_counts['Murder'] = incidents_df[
    (incidents_df['death']==True) &
    ((incidents_df['aggression']==True) |
    (incidents_df['social_reasons']==True))].shape[0] # not accidental nor defensive
tags_counts['Suicide'] = incidents_df[
    (incidents_df['death']==True) &
    (incidents_df['suicide']==True)].shape[0] # warninig: if murder/suicide is counted twice
tags_counts['Defensive'] = incidents_df[
    (incidents_df['death']==True) &
    (incidents_df['defensive']==True)].shape[0]
tags_counts['Accidental'] = incidents_df[
    (incidents_df['death']==True) &
    (incidents_df['unintentional']==True)].shape[0]
tags_counts['Others or not known'] = incidents_df[
    (incidents_df['death']==True) &
    (incidents_df['aggression']==False) &
    (incidents_df['social_reasons']==False) &
    (incidents_df['unintentional']==False)].shape[0]

fig, ax = plt.subplots()
total = sum(tags_counts.values())
ax.pie(tags_counts.values())
legend_labels = [f'{label}: {(size/total)*100:.1f}%' for label, size in tags_counts.items()]
plt.legend(legend_labels)
plt.title("Gun incidents")
plt.show()
fig.savefig("../html/pie_incident_type.svg")

# %% [markdown]
# Most of the incidents involved Murder. Suicide, Defensive and Accidental are very few compare to murders. The other big slice of the pie belongs to 'Others', showing that there are a lot of different incidents that are less common.

# %% [markdown]
# We show the frequency of the values of each singular tag:

# %%
ax = (incidents_df[tags_columns].apply(lambda col: col.value_counts()).T.sort_values(by=True)/incidents_df.shape[0]*100).plot(kind='barh', stacked=True, alpha=0.8, edgecolor='black')
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=8)
plt.title("Incidents characteristic (%)")

# %% [markdown]
# The most common tags are firearm, shots, aggression and injuries (above 50% of the records), in particular firearm is True for almost every record (97.8 %). On the other hand there are tags (air_gun, school, social_reasons and abduction) that are very rare.

# %% [markdown]
# We check for correlations between accidental incidents and the presence of children.

# %%
incidents_df['unintentional'].corr(incidents_df['n_participants_child']>0)

# %% [markdown]
# The two events are not correlated.

# %% [markdown]
# We display the most common characteristics for incidents involving women.

# %%
ch1_females_counts = incidents_df[incidents_df['n_females']>1]['incident_characteristics1'].value_counts()
ch2_females_counts = incidents_df[incidents_df['n_females']>1]['incident_characteristics2'].value_counts()
ch_females_counts = ch1_females_counts.add(ch2_females_counts, fill_value=0).sort_values(ascending=False).plot(
    kind='bar',
    title='Characteristics counts of incidents with females involved',
    figsize=(20,10)
)

# %% [markdown]
# The distribution is very similar to the one involving both men and women. Some of the main differences are that, for women, the frequency of suicides is higher, while the involvemnte of officiers is lower.
# 
# We plot on a map the location of mass shootings, incidents involving children and suicides:

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['n_killed']>=4],
        zoom=2,
        title='Mass shootings'
)

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['children']==True],
    zoom=2,
    title='Incidents involving children'
)

# %%
plot_scattermap_plotly(
    incidents_df[incidents_df['suicide']==True],
    zoom=2,
    title='Suicides'
)

# %% [markdown]
# The dislocation of these kind of incidents is similar to the one of the whole dataset.

# %% [markdown]
# ## Joint analysis of the datasets

# %% [markdown]
# We join the poverty data with the incidents data:

# %%
poverty_df['state'] = poverty_df['state'].str.upper()
incidents_df = incidents_df.merge(poverty_df, on=['state', 'year'], how='left', validate="m:1")
incidents_df.head()

# %% [markdown]
# We join the elections data with the incidents data:

# %%
elections_df_copy = elections_df.copy()
elections_df_copy['year'] = elections_df_copy['year'] + 1
elections_df = pd.concat([elections_df, elections_df_copy], ignore_index=True)
incidents_df = incidents_df.merge(elections_df, on=['state', 'year', 'congressional_district'], how='left')
incidents_df.head()

# %% [markdown]
# We read and join the data about the USA population from the 2010 census downloaded from [Wikipedia](https://en.wikipedia.org/wiki/2010_United_States_census).

# %%
usa_population_df = pd.read_csv(DATA_FOLDER_PATH + 'external_data/2010_United_States_census.csv')

# %%
usa_population_df.info()

# %%
usa_population_df.drop(columns=['Population as of 2000 census', 'Change', 'Percent change', 'Rank'], inplace=True)
usa_population_df.rename(columns={'Population as of 2010 census':'population_state_2010', 'State': 'state'}, inplace=True)
usa_population_df['state'] = usa_population_df['state'].str.upper()
usa_population_df['population_state_2010'] = usa_population_df['population_state_2010'].str.replace(',', '').astype('int64')
incidents_df = incidents_df.merge(usa_population_df, on=['state'], how='left')
incidents_df.sample(5, random_state=1)

# %% [markdown]
# We plot the number of incidents per state per 100k inhabitants:

# %%
incidents_per_state = incidents_df[incidents_df['year']<=2020].groupby(['state', 'population_state_2010']).size()
incidents_per_state = ((incidents_per_state / incidents_per_state.index.get_level_values('population_state_2010'))*100000).to_frame(name='incidents_per_100k_inhabitants').sort_values(by='incidents_per_100k_inhabitants', ascending=True)
incidents_per_state.reset_index(inplace=True)
incidents_per_state.plot(
    kind='barh',
    x='state',
    y='incidents_per_100k_inhabitants',
    figsize=(15, 10),
    ylabel='State',
    xlabel='Incidents per 100k inhabitants',
    title='Incidents per 100k inhabitants per state'
)

# %% [markdown]
# District of Columbia has the highest number of incidents per 100k inhabitants.

# %% [markdown]
# We display the tag frequency for the District of Columbia:

# %%
incidents_df[incidents_df['state']=='DISTRICT OF COLUMBIA']['incident_characteristics1'].value_counts().plot(kind='barh', figsize=(20, 10))

# %% [markdown]
# We visualize the number of incidents happened in each state every month:

# %%
incidents_df['year'] = incidents_df['year'].astype('UInt64')
incidents_per_month_per_state = incidents_df.groupby(['state', 'month_name', 'year']).size()
incidents_per_month_per_state = incidents_per_month_per_state.to_frame(name='incidents').reset_index()
incidents_per_month_per_state = incidents_per_month_per_state.sort_values(by=['year', 'month_name', 'state'], ignore_index=True)
incidents_per_month_per_state['incidents_per_100k_inhabitants'] = incidents_per_month_per_state.apply(
    lambda row: (row['incidents'] / usa_population_df[usa_population_df['state']==row['state']]['population_state_2010'].iloc[0])*100000,
    axis=1
)
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(
    incidents_per_month_per_state[incidents_per_month_per_state.year<=2020].pivot(
        index='state',
        columns=['year', 'month_name'],
        values='incidents_per_100k_inhabitants'
    ).fillna(0),
    cmap='coolwarm',
    ax=ax,
    xticklabels=True,
    yticklabels=True,
    linewidths=.5
)
ax.set_xlabel('Month-Year')
ax.set_ylabel('State')
ax.set_title('Number of incidents per 100k inhabitants')

plt.xticks(rotation=90)
plt.tight_layout()

# %%
incidents_per_month_per_state = incidents_df[incidents_df['incident_characteristics1']!='Non-Shooting Incident'].groupby(['state', 'month_name', 'year']).size()
incidents_per_month_per_state = incidents_per_month_per_state.to_frame(name='incidents').reset_index()
incidents_per_month_per_state = incidents_per_month_per_state.sort_values(by=['year', 'month_name', 'state'], ignore_index=True)
incidents_per_month_per_state['incidents_per_100k_inhabitants'] = incidents_per_month_per_state.apply(
    lambda row: (row['incidents'] / usa_population_df[usa_population_df['state']==row['state']]['population_state_2010'].iloc[0])*100000,
    axis=1
)
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(
    incidents_per_month_per_state[incidents_per_month_per_state.year<=2020].pivot(
        index='state',
        columns=['year', 'month_name'],
        values='incidents_per_100k_inhabitants'
    ).fillna(0),
    cmap='coolwarm',
    ax=ax,
    xticklabels=True,
    yticklabels=True,
    linewidths=.5
)
ax.set_xlabel('Month-Year')
ax.set_ylabel('State')
ax.set_title('Number of incidents per 100k inhabitants (excluding non-shooting incidents)')

plt.xticks(rotation=90)
plt.tight_layout()

# %% [markdown]
# We exclude District of Columbia:

# %%
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(
    incidents_per_month_per_state[(incidents_per_month_per_state.year<=2020) & (incidents_per_month_per_state['state']!='DISTRICT OF COLUMBIA')].pivot(
        index='state',
        columns=['year', 'month_name'],
        values='incidents_per_100k_inhabitants'
    ).fillna(0),
    cmap='coolwarm',
    ax=ax,
    xticklabels=True,
    yticklabels=True,
    linewidths=.5
)
ax.set_xlabel('Month-Year')
ax.set_ylabel('State')
ax.set_title('Number of incidents per 100k inhabitants')


plt.xticks(rotation=90)
plt.tight_layout()

fig.savefig("../html/heatmap_incidents_months.svg")

# %% [markdown]
# We plot the correlations between the number of incidents and the poverty percentage in each state coloring the points according to the party that got the majority of votes in that state:

# %%
winning_party_per_state = pd.read_csv(DATA_FOLDER_PATH + '../data/winning_party_per_state.csv')
winning_party_per_state_copy = winning_party_per_state.copy()
winning_party_per_state_copy['year'] = winning_party_per_state['year'] + 1
winning_party_per_state = pd.concat([winning_party_per_state, winning_party_per_state_copy], ignore_index=True)
incidents_df = incidents_df[incidents_df['year'].notna()].merge(winning_party_per_state[['state', 'year', 'majority_state_party']], on=['state', 'year'], how='left')

incidents_per_state_2016 = incidents_df[(incidents_df['n_killed']>0)].groupby(['state', 'year', 'population_state_2010', 'povertyPercentage', 'majority_state_party']).size()
incidents_per_state_2016 = incidents_per_state_2016.to_frame(name='incidents').reset_index()
incidents_per_state_2016['incidents_per_100k_inhabitants'] = (incidents_per_state_2016['incidents'] / incidents_per_state_2016['population_state_2010'])*100000
fig = px.scatter(
    incidents_per_state_2016,
    x='povertyPercentage',
    y='incidents_per_100k_inhabitants',
    color='majority_state_party',
    hover_name='state',
    hover_data={'povertyPercentage': True, 'incidents_per_100k_inhabitants': True},
    title='Mortal gun incidents in the USA',
    facet_col="year",
    facet_col_wrap=3,
    height=800
)
pyo.plot(fig, filename='../html/scatter_poverty.html', auto_open=False)
fig.show()

# %% [markdown]
# The two attributes are slightly correlated.

# %% [markdown]
# We plot the correlations between each attribute:

# %%
numerical_columns = incidents_df.select_dtypes(include=['float64', 'int64']).columns
fig = plt.figure(figsize=(15, 12))
corr_matrix = incidents_df[numerical_columns].corr(method='pearson')
sns.heatmap(corr_matrix, mask=np.triu(corr_matrix));
plt.title('Correlation matrix (Pearson)');
fig.savefig("../html/attributes_correlation_matrix.svg")

# %% [markdown]
# The attributes min_age_participants, avg_age_participants and max_age_participants are positively correlated with each other (probably because there are many incidents involving a single person). n_participants is positively correlated with n_participants_adults and n_males: most of the incidents involve adult males. povertyPercentage is inversly correlated with latitude: southern states are poorer.

# %% [markdown]
# We re-order the columns and we save the cleaned dataset:

# %%
time_columns = ['date', 'date_original', 'year', 'month', 'day', 'day_of_week']

geo_columns = ['state', 'address', 'latitude', 'longitude',
               'county', 'city', 'location_importance', 'address_type',
               'congressional_district', 'state_house_district', 'state_senate_district',
               'px_code']

participants_columns = ['participant_age1', 'participant1_child',
       'participant1_teen', 'participant1_adult', 'participant1_male',
       'participant1_female', 'min_age_participants', 'avg_age_participants', 'max_age_participants',
       'n_participants_child', 'n_participants_teen', 'n_participants_adult',
       'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested',
       'n_unharmed', 'n_participants']

characteristic_columns = ['notes', 'incident_characteristics1', 'incident_characteristics2', 
    'firearm', 'air_gun', 'shots', 'aggression', 'suicide', 'injuries',
    'death', 'road', 'illegal_holding', 'house', 'school', 'children',
    'drugs', 'officers', 'organized', 'social_reasons', 'defensive',
    'workplace', 'abduction', 'unintentional']

external_columns = ['povertyPercentage', 'party', 'candidatevotes', 'totalvotes', 'candidateperc', 'population_state_2010']

incidents_df = incidents_df[time_columns + geo_columns + participants_columns + characteristic_columns + external_columns]
incidents_df = incidents_df.rename(
    columns={
        'povertyPercentage': 'poverty_perc',
        'candidatevotes': 'candidate_votes',
        'totalvotes': 'total_votes',
        'candidateperc': 'candidate_perc'
    }
)

incidents_df.to_csv(DATA_FOLDER_PATH +'incidents_cleaned.csv', index=True)


