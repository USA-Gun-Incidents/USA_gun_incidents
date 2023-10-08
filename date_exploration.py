# -*- coding: utf-8 -*-
# %% [markdown]
# # Simple data preprocessing and cleaning

# %%
# %matplotlib inline

#lib
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

# %% [markdown]
# removing duplicated rows

# %%
#the dataset is probably read right
inc = pd.read_csv('data/incidents.csv', sep=',') 
inc.drop_duplicates(inplace=True)
inc.info()
inc.head()

# %%
for col in inc:
    dummy = inc[col].unique()
    print( [ col, dummy, len(dummy)] )

# %%
inc.drop('notes', axis=1, inplace=True)
inc['date'] = inc.apply(lambda row : pd.to_datetime(row['date'], format="%Y-%m-%d"), axis = 1)

# %% [markdown]
# # Checking semantic and syntactic concistency
#

# %%
print(type(inc['date'][0]))
inc.head(10)

# %% [markdown]
# Date distribution before error correction

# %%
# plot range data
tot_row = len(inc.index)

# one binth for every month in the range
min_date = inc['date'].min()
max_date = inc['date'].max()
n_bin = int((max_date - min_date).days / 30) 
n_bin_2 = int(1 + math.log2(tot_row)) #Sturge's rule

equal_freq_bins=inc['date'].sort_values().quantile(np.arange(0,1, 1/n_bin)).to_list()
equal_freq_bins2=inc['date'].sort_values().quantile(np.arange(0,1, 1/n_bin_2)).to_list()

fig, axs = plt.subplots(4, sharex=True, sharey=True)
fig.set_figwidth(14)
fig.set_figheight(6)
fig.suptitle('Dates distribution')

colors_palette = iter(mcolors.TABLEAU_COLORS)
bins = [n_bin, n_bin_2, equal_freq_bins, equal_freq_bins2]
ylabels = ['fixed #bin', 'Sturge\'s rule', 'f. #bin density', 'S. rule density']
for i, ax in enumerate(axs):
    ax.hist(inc['date'], bins=bins[i], color=next(colors_palette), density=True)

    ax.set_ylabel(ylabels[i])
    ax.grid(axis='y')
    ax.axvline(min_date, color='k', linestyle='dashed', linewidth=1)
    ax.axvline(max_date, color='k', linestyle='dashed', linewidth=1)
axs[3].set_xlabel('dates')


print('Range data: ', inc['date'].min(), ' - ', inc['date'].max())

# %%
import matplotlib.dates as mdates

def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = mdates.num2date(bp['whiskers'][i*2].get_ydata()[1])
        dict1['lower_quartile'] = mdates.num2date(bp['boxes'][i].get_ydata()[1])
        dict1['median'] = mdates.num2date(bp['medians'][i].get_ydata()[1])
        dict1['upper_quartile'] = mdates.num2date(bp['boxes'][i].get_ydata()[2])
        dict1['upper_whisker'] = mdates.num2date(bp['whiskers'][(i*2)+1].get_ydata()[1])
        
        dict1['fliers'] = len(bp['fliers'][i].get_ydata())
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)


# %%
ticks = []
labels = []
for i in range(2012, 2032):
    ticks.append(mdates.date2num(pd.to_datetime(str(i) + '-01-01', format="%Y-%m-%d")))
    labels.append(str(i))

boxplot = plt.boxplot(x=mdates.date2num(inc['date']), labels=['dates'])
print()
plt.yticks(ticks, labels)
plt.grid()
dates_data = get_box_plot_data(['dates'], boxplot)
dates_data


# %%
print(type(inc['date']))

# %%
#inc_cleaned = inc.dropna(axis=0).copy()
inc_cleaned = inc.copy()
t = inc_cleaned.dtypes
for i, c in enumerate(inc_cleaned.columns):
    if t[i] == 'object':
        inc_cleaned.insert(len(inc_cleaned.columns), value=inc_cleaned[c].astype("category").cat.codes, column=c + ' codes')
        inc_cleaned.drop(c, axis=1, inplace=True)
inc_cleaned['date'] = mdates.date2num(inc_cleaned['date'])

inc_cleaned.info()

# %% [markdown]
# in order to calculate the correlation I convert all the data stored as objects into categories, and replace the value with the associated code (it would be better not to do this for numbers but there are errors)

# %%
corr = inc_cleaned.corr()
corr.style.background_gradient(cmap='coolwarm')

# %% [markdown]
# computing the scatter matrix among the features most correlated with the date

# %%
corr['date']=corr['date'].apply(lambda x: abs(x))
corr_dummy = corr.sort_values('date', ascending=False, axis=0).head(6)
inc_cleaned[corr_dummy.index]
scatter_axes = pd.plotting.scatter_matrix(inc_cleaned[corr_dummy.index], figsize=(10,10), alpha=0.2)
for ax in scatter_axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

# %% [markdown]
# All the dates are correct, both from a syntactic point of view, in fact they do not present null values ​​or illegible values. Looking at the graph we notice that there are incorrect values, greater than the maximum
#
#
# Finally, we find no correlation of any kind between data and other values ​​in the dataset

# %% [markdown]
# # Error correction and distribution analysis

# %%
#let's try to remove 10 years from the wrong dates, considering the error, a typo
actual_index = inc.index.tolist()

def subtract_ten_if(x):
        if x['date'] > dates_data['upper_whisker'][0].to_datetime64(): 
                return x['date'] - pd.DateOffset(years=10)
        else: return x['date']

def replace_with_random(x):
        ret = x['date']
        while ret > dates_data['upper_whisker'][0].to_datetime64(): 
                ret = inc['date'][random.choice(actual_index)]
        return ret

def replace_with_median(x):
        ret = x['date']
        while ret > dates_data['upper_whisker'][0].to_datetime64(): 
                ret = dates_data['median'][0].to_datetime64()
        return ret

mod1 = inc.apply(lambda row : subtract_ten_if(row), axis = 1)
mod2 = inc.apply(lambda row : replace_with_random(row), axis = 1)
mod3 = inc.apply(lambda row : replace_with_median(row), axis = 1)

# %%
print(len(mod1))

# %%
# one binth for every month in the range
fixed_bin = int((dates_data['upper_whisker'][0] - dates_data['lower_whisker'][0]).days / 30)

prop_bin = []
prop_bin.append(inc['date'].sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list())
prop_bin.append(mod1.sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list())
prop_bin.append(mod2.sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list())
prop_bin.append(mod3.sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list())


fig, axs = plt.subplots(4, 2, sharex=True, sharey=False)
fig.set_figwidth(14)
fig.set_figheight(5)
fig.suptitle('Dates distribution')

colors_palette = iter(mcolors.TABLEAU_COLORS)
bins = [n_bin, equal_freq_bins]
ylabels = ['original', 'mod 1', 'mod2', 'mod3']
dates = [inc['date'],  mod1, mod2, mod3]

for i, ax in enumerate(axs):
    for el in dates_data.loc[0][1:]:
        ax[0].axvline(el, color='k', linestyle='dashed', linewidth=1, alpha=0.4)
        ax[1].axvline(el, color='k', linestyle='dashed', linewidth=1, alpha=0.4)
        
    c = next(colors_palette)
    ax[0].hist(dates[i], bins=fixed_bin, color=c, density=False)
    ax[1].hist(dates[i], bins=prop_bin[i], color=c, density=True)
    

    ax[0].set_ylabel(ylabels[i])
    ax[0].grid(axis='y')
    ax[1].grid(axis='y')
    
    


# %%

dates_num = []
for i in dates:
    dates_num.append(mdates.date2num(i))

boxplot = plt.boxplot(x=dates_num, labels=ylabels)
plt.yticks(ticks, labels)
plt.grid()

dates_data = get_box_plot_data(ylabels, boxplot)


# %% [markdown]
# Of the three methods used for error correction, only the third is satisfactory for two reasons:
#
# -Is the one that preserves the distribution the most thank's to the rendom sampling approach, and since the values ​​are more than 23K they cannot be replaced with equal values
#
# -Since the date of the incidents is not related in any way to other fields of the dataset, and failing to identify the cause that generated the errors, replacing the dates with random ones from the dataset does not introduce particular inconsistencies



# %%
dates_data
