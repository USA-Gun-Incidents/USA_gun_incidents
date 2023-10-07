# %% [markdown]
# Data Understanding con i Bro

# %% [markdown]
# Semantic and syntactic checking of the data field, followed by error correction

# %%
%matplotlib inline

#lib
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# %%
#the dataset is probably read right
inc = pd.read_csv('data/incidents.csv', sep=',') 
inc.drop_duplicates()
inc.info()

# %%
for col in inc:
    dummy = inc[col].unique()
    print( [ col, dummy, len(dummy)] )

# %% [markdown]
# Syntax and data semantics check 

# %%
# corventing dates in a correct format
inc['date'] = inc.apply(lambda row : pd.to_datetime(row['date'], format="%Y-%m-%d"), axis = 1)


# %%
inc_coded = {}
inc_coded['date'] = inc['date']

inc_coded = pd.DataFrame(inc_coded)
inc_coded.head()

# %% [markdown]
# Date distribution before error correction

# %%
# plot range data
tot_row = len(inc.index)

# one binth for every month in the range
min_date = inc['date'].min()
max_date = inc['date'].max()
n_bin = int((max_date - min_date).days / 30)
n_bin_2 = int(1 + (10/3)*math.log10(tot_row))

freq = int(tot_row/n_bin)
equal_freq_bins=inc['date'].sort_values().quantile(np.arange(0,1, 1/n_bin)).to_list()
equal_freq_bins2=inc['date'].sort_values().quantile(np.arange(0,1, 1/n_bin_2)).to_list()

fig, axs = plt.subplots(4, sharex=True, sharey=True,)
fig.set_figwidth(14)
fig.suptitle('Sharing both axes')

colors_palette = iter(mcolors.TABLEAU_COLORS)
axs[0].hist(inc['date'], bins=n_bin, color=next(colors_palette), density=True)
axs[1].hist(inc['date'], bins=n_bin_2, color=next(colors_palette), density=True)
axs[2].hist(inc['date'], bins=equal_freq_bins, color=next(colors_palette),  density=True)
axs[3].hist(inc['date'], bins=equal_freq_bins2, color=next(colors_palette),  density=True)
for ax in axs:
    ax.axvline(min_date, color='k', linestyle='dashed', linewidth=1)
    ax.axvline(max_date, color='k', linestyle='dashed', linewidth=1)



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
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)


ticks = []
labels = []
for i in range(2012, 2032):
    ticks.append(mdates.date2num(pd.to_datetime(str(i) + '-01-01', format="%Y-%m-%d")))
    labels.append(str(i))

boxplot = plt.boxplot(x=mdates.date2num(inc['date']))
plt.yticks(ticks, labels)
dates_data = get_box_plot_data('a', boxplot)


# %%
factors_paired = [(i,j) for i in inc.columns.values for j in inc.columns.values] 

chi2, p_values =[], []

for f in factors_paired:
    if f[0] != f[1]:
        chitest = chi2_contingency(pd.crosstab(df[f[0]], df[f[1]]))   
        chi2.append(chitest[0])
        p_values.append(chitest[1])
    else:      # for same factor pair
        chi2.append(0)
        p_values.append(0)

chi2 = np.array(chi2).reshape((23,23)) # shape it as a matrix
chi2 = pd.DataFrame(chi2, index=df.columns.values, columns=df.columns.values) # then a df for convenience

# %% [markdown]
# Syntax and semantic check after applying different techniques

# %%

#let's try to remove 10 years from the wrong dates, considering the error, a typo

def subtract_ten_if(x):
        if x['date'] > dates_data['upper_whisker'][0].to_datetime64(): 
                return x['date'] - pd.DateOffset(years=10)
        else: return x['date']


new_dates = inc.apply(lambda row : subtract_ten_if(row), axis = 1)

# %%
# plot range data
tot_row = len(inc.index)

# one binth for every month in the range
min_date = new_dates.min()
max_date = new_dates.max()
n_bin = int((max_date - min_date).days / 30)
n_bin_2 = int(1 + (10/3)*math.log10(tot_row))

freq = int(tot_row/n_bin)
equal_freq_bins=new_dates.sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin)).to_list()
equal_freq_bins2=new_dates.sort_values(ascending=False).quantile(np.arange(0,1, 1/n_bin_2)).to_list()

fig, axs = plt.subplots(4, sharex=True, sharey=True,)
fig.set_figwidth(14)
fig.suptitle('Sharing both axes')

colors_palette = iter(mcolors.TABLEAU_COLORS)
axs[0].hist(new_dates, bins=n_bin, color=next(colors_palette), density=True)
axs[1].hist(new_dates, bins=n_bin_2, color=next(colors_palette), density=True)
axs[2].hist(new_dates, bins=equal_freq_bins, color=next(colors_palette),  density=True)
axs[3].hist(new_dates, bins=equal_freq_bins2, color=next(colors_palette),  density=True)
for ax in axs:
    ax.axvline(min_date, color='k', linestyle='dashed', linewidth=1)
    ax.axvline(max_date, color='k', linestyle='dashed', linewidth=1)



print('Range data: ', min_date, ' - ', max_date)

# %%
def replace_with_(x):
        if x['date'] > dates_data['upper_whisker'][0].to_datetime64(): 
                return x['date'] - pd.DateOffset(years=10)
        else: return x['date']


new_dates_2 = inc.apply(lambda row : subtract_ten_if(row), axis = 1)


