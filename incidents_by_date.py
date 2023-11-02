# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import hist_box_plot

# %%
incidents_df = pd.read_csv('./data/final.csv', index_col=0)
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')


# %%
def compute_ratio_indicator(df, gby, num, den, suffix, agg_fun):
    grouped_df = df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    df[num+'_'+den+suffix+'_ratio'] = df[num] / df[den+suffix]
    df.drop(columns=[den+suffix], inplace=True)
    return df


# %% [markdown]
# ## Visualize data

# %% [markdown]
# ### Entries without city

# %%
incidents_df[incidents_df['city'].isna()].shape[0]

# %%
incidents_df[incidents_df['city'].isna()].groupby('state')['state'].count()

# %%
fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(incidents_df[incidents_df['city'].isna()]['state'], bins=52, label='#Missing city', edgecolor='black', linewidth=0.8)
ax.hist(incidents_df['state'], bins=52, label='#Total', density=False, histtype='step', linewidth=1)
ax.set_xlabel('State')
#ax.set_yscale('log')
ax.set_ylabel('Number of incidents')
ax.legend()
ax.xaxis.set_tick_params(rotation=90)
plt.show()

# %% [markdown]
# ### Incidents per day, during years

# %%
incidents_df.groupby(incidents_df['date'])['date'].count().describe()

# %%
incidents_df['date'].groupby([incidents_df['date'].dt.month, incidents_df['date'].dt.day]).count()

# %%
incidents_df['date'].groupby([incidents_df['date'].dt.month, incidents_df['date'].dt.day]).describe(datetime_is_numeric=True)

# %%
incidents_df['date'].groupby([incidents_df['date'].dt.month, incidents_df['date'].dt.day]).count().describe()

# %%
date_list = []
for month, day in np.array(incidents_df['date'].groupby([incidents_df['date'].dt.month, incidents_df['date'].dt.day]).count().index):
    if month==1: string_month='January'
    elif month==2: string_month='February'
    elif month==3: string_month='March'
    elif month==4: string_month='April'
    elif month==5: string_month='May'
    elif month==6: string_month='June'
    elif month==7: string_month='July'
    elif month==8: string_month='August'
    elif month==9: string_month='September'
    elif month==10: string_month='October'
    elif month==11: string_month='November'
    elif month==12: string_month='December'
    date_list.append(string_month+' '+str(day))


# %%
frequency = np.array(incidents_df['date'].groupby([incidents_df['date'].dt.month, incidents_df['date'].dt.day]).count())

# %%
from matplotlib.widgets import Cursor

# %%
plt.figure(figsize=(20, 5))
plt.plot(frequency, '.', label='Primo quantile')
plt.plot(np.where(frequency>611)[0], frequency[frequency>611], '.', label='Secondo quantile')
plt.plot(np.where(frequency>647)[0], frequency[frequency>647], '.', label='Terzo quantile')
plt.plot(np.where(frequency>690)[0], frequency[frequency>690], '.', label='Quarto quantile')
plt.xticks(range(0, len(date_list), 5), date_list[::5], rotation=90, fontsize=12)
plt.xlim(-1, len(date_list))
plt.legend()
plt.ylabel('Number of incidents')
plt.xlabel('Date')
cursor = Cursor(plt.gca(), useblit=True, color='red', linewidth=1)
plt.show()

# %%
len(np.where(frequency>690)[0])

# %%
for i in np.where(frequency>800)[0]:
    print(date_list[i])
    print(frequency[i])

# %%
for i in np.where(frequency>750)[0]:
    print(date_list[i])

# %%
plt.figure(figsize=(20, 5))
plt.stairs(frequency, fill=True, color='red', label='Quarto quantile')
plt.stairs([frequency if frequency<690 else 0 for frequency in frequency], fill=True, color='orange', label='Terzo quantile')
plt.stairs([frequency if frequency<647 else 0 for frequency in frequency], fill=True, color='yellow', label='Secondo quantile')
plt.stairs([frequency if frequency<611 else 0 for frequency in frequency], fill=True, color='magenta', label='Primo quantile')
plt.legend()
plt.xticks(range(0, len(date_list), 5), date_list[::5], rotation=90, fontsize=12)
plt.xlim(0, len(date_list))
plt.ylim(140, 1250)
plt.ylabel('Number of incidents')
plt.xlabel('Date')
plt.show()

# %%
# incidents first january
incidents_df[(incidents_df['date'].dt.month==1) & (incidents_df['date'].dt.day==1)].groupby('year')['year'].count()

# %%
incidents_df[(incidents_df['date'].dt.month==7) & (incidents_df['date'].dt.day==4)].groupby('year')['year'].count()

# %% [markdown]
# [Federal Holiday calendar in USA](https://www.commerce.gov/hr/employees/leave/holidays)
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

# %%
incidents_df['date'][0].day_name()

# %%
incidents_df['date'][0].month_name()

# %%
incidents_df['date'][0].is_leap_year

# %%
str(incidents_df['date'][0].year)+'-'+str(incidents_df['date'][0].)+'-'+str(incidents_df['date'][0].day)

# %%
# number of incidents during black friday, the day after thanksgiving
incidents_df.groupby(incidents_df['date'].isin(['2013-11-29', '2014-11-28', '2015-11-27', '2016-11-25', '2017-11-24', 
    '2018-11-23'])==True).count()

# %% [markdown]
# 505 incidenti durante il black friday (primo quantile), ma comunque Novembre ha meno incidenti rispetto agli altri mesi

# %% [markdown]
# Thanksgiving Day è il giorno con meno incidenti in assoluto
#
# Capodanno quello con più incidenti
#
# 29 febbraio non lo considero
#
# Natale, Columnbus Day, Juneteenth National Independence Day, Thanksgiving Day, Veterans Day sono nel primo quantile. \
# Natale e Ringraziamento stanno a casa a festeggiare. \
# Durante Columnbus Day, Juneteenth National Independence Day, Veterans Day vengono organizzate parate e cose pubbliche. \
# Juneteenth National Independence Day: celebra la liberazione degli schiavi in ​​Texas il 19 giugno 1865.           
#
# A marzo molti incidenti \
# Altre cose da considerare: spring break, san Patrick (17 marzo), pasqua (la festeggiano e ci sono anche eventi religiosi tipo processioni)

# %%
holiday_dict = {'New Year': ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01'],
    'Martin Luther King Day': ['2013-01-21', '2014-01-20', '2015-01-19', '2016-01-18', '2017-01-16', '2018-01-15'],
    'Washington Birthday': ['2013-02-18', '2014-02-17', '2015-02-16', '2016-02-15', '2017-02-20', '2018-02-19'],
    'Sant Patrick Day': ['2013-03-17', '2014-03-17', '2015-03-17', '2016-03-17', '2017-03-17', '2018-03-17'],
    'Easter': ['2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27', '2017-04-16', '2018-04-01'],
    'Easter Monday': ['2013-04-01', '2014-04-21', '2015-04-06', '2016-03-28', '2017-04-17', '2018-04-02'],
    'Memorial Day': ['2013-05-27', '2014-05-26', '2015-05-25', '2016-05-30', '2017-05-29', '2018-05-28'],
    'Juneteenth National Independence Day': ['2013-06-19', '2014-06-19', '2015-06-19', '2016-06-19', '2017-06-19', '2018-06-19'],
    'Independence Day': ['2013-07-04', '2014-07-04', '2015-07-03', '2016-07-04', '2017-07-04', '2018-07-04'],
    'Labor Day': ['2013-09-02', '2014-09-01', '2015-09-07', '2016-09-05', '2017-09-04', '2018-09-03'],
    'Columbus Day': ['2013-10-14', '2014-10-13', '2015-10-12', '2016-10-10', '2017-10-09', '2018-10-08'],
    'Veterans Day': ['2013-11-11', '2014-11-11', '2015-11-11', '2016-11-11', '2017-11-11', '2018-11-11'],
    'Thanksgiving Day': ['2013-11-28', '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23', '2018-11-22'],
    'Black Friday': ['2013-11-29', '2014-11-28', '2015-11-27', '2016-11-25', '2017-11-24', '2018-11-23'],
    'Christmas Day': ['2013-12-25', '2014-12-25', '2015-12-25', '2016-12-26', '2017-12-25', '2018-12-25']}

# %%
# number of incidents in each holiday
incidents_df['holiday'] = 0
for holiday in holiday_dict.keys():
    incidents_df.loc[incidents_df['date'].isin(holiday_dict[holiday]), 'holiday'] = holiday
incidents_df.groupby('holiday')['holiday'].count()

# %%
print('Number of days with more than 450 incidents: ', len(np.where(frequency>450)[0]))

# %%
for i in np.where(frequency<451)[0]:
    print(date_list[i])
    print(frequency[i])
