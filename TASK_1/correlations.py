# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
indicators_df = pd.read_csv('../data/all_indicators.csv', index_col=0)

# %%
indicators_df.head(2)

# %% [markdown]
# 
# **Geo, data, tags and mixed information indicators**:
# 
# 'location_importance', 'city_entropy', 'address_entropy',  
# 
# 'tags_entropy'
# 
# entropy_month_fixing_year_sem_state_congd
# entropy_month_fixing_year_state
# entropy_day_fixing_year_sem_state_congd
# entropy_day_fixing_year_state
# 
# entropy_address_type_fixing_year_sem_state_congd
# entropy_address_type_fixing_year_state
# 
# entropy_month_day_fixing_year_sem_state_congd
# 
# entropy_tag_congd
# 
# severity
# 
# **Number of participants indicators**:
# 
# 'n_participants'
# 
# n_participants__sum_sem_congd_ratio
# n_participants__sum_year_state_ratio
# n_participants__mean_sem_congd_ratio
# n_participants__mean_year_state_ratio
# 
# log_n_participants__mean_sem_congd_ratio
# log_n_participants__mean_year_state_ratio
# 
# log_n_participants__sum_sem_congd_ratio
# log_n_participants__sum_year_state_ratio
# 
# **Gender participants indicators**:
# 
# 'n_males', 'n_females', 'n_males_pr'
# 
# 'log_n_males_n_males_mean_semest_congd_ratio',
# 
# n_males__sum_sem_congd_ratio
# n_males__sum_year_state_ratio
# n_males__mean_sem_congd_ratio
# n_males__mean_year_state_ratio
# 
# n_females__sum_sem_congd_ratio
# n_females__sum_year_state_ratio
# n_females__mean_sem_congd_ratio
# n_females__mean_year_state_ratio
# 
# log_n_males__mean_sem_congd_ratio
# log_n_males__mean_year_state_ratio
# log_n_females__mean_sem_congd_ratio
# log_n_females__mean_year_state_ratio
# 
# log_n_males__sum_sem_congd_ratio
# log_n_males__sum_year_state_ratio
# log_n_females__sum_sem_congd_ratio
# log_n_females__sum_year_state_ratio
# 
# **Killed/injured indicators**:
# 
# 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed'
# 
# 'n_killed_pr', 'n_injured_pr', 'n_arrested_pr', 'n_unharmed_pr'
# 
# n_killed__sum_sem_congd_ratio
# n_killed__sum_year_state_ratio
# n_killed__mean_sem_congd_ratio
# n_killed__mean_year_state_ratio
# 
# n_injured__sum_sem_congd_ratio
# n_injured__sum_year_state_ratio
# n_injured__mean_sem_congd_ratio
# n_injured__mean_year_state_ratio
# 
# n_arrested__sum_sem_congd_ratio
# n_arrested__sum_year_state_ratio
# n_arrested__mean_sem_congd_ratio
# n_arrested__mean_year_state_ratio
# 
# n_unharmed__sum_sem_congd_ratio
# n_unharmed__sum_year_state_ratio
# n_unharmed__mean_sem_congd_ratio
# n_unharmed__mean_year_state_ratio
# 
# log_n_killed__mean_sem_congd_ratio
# log_n_killed__mean_year_state_ratio
# log_n_injured__mean_sem_congd_ratio
# log_n_injured__mean_year_state_ratio
# log_n_arrested__mean_sem_congd_ratio
# log_n_arrested__mean_year_state_ratio
# log_n_unharmed__mean_sem_congd_ratio
# log_n_unharmed__mean_year_state_ratio
# 
# log_n_killed__sum_sem_congd_ratio
# log_n_killed__sum_year_state_ratio
# log_n_injured__sum_sem_congd_ratio
# log_n_injured__sum_year_state_ratio
# log_n_arrested__sum_sem_congd_ratio
# log_n_arrested__sum_year_state_ratio
# log_n_unharmed__sum_sem_congd_ratio
# log_n_unharmed__sum_year_state_ratio
# 
# 
# **Age indicators**:
# 
# 'min_age', 'avg_age', 'max_age', 'age_range', 
# 
# 'avg_age_entropy'
# 
# 'n_child', 'n_teen', 'n_adult',
# 
# 'n_participants_child_prop', 'n_participants_teen_prop',
# 'n_adults_entropy',
# 
# avg_age__sum_sem_congd_ratio
# avg_age__sum_year_state_ratio
# avg_age__mean_sem_congd_ratio
# avg_age__mean_year_state_ratio
# 
# max_age__sum_sem_congd_ratio
# max_age__sum_year_state_ratio
# max_age__mean_sem_congd_ratio
# max_age__mean_year_state_ratio
# 
# min_age__sum_sem_congd_ratio
# min_age__sum_year_state_ratio
# min_age__mean_sem_congd_ratio
# min_age__mean_year_state_ratio
# 
# log_avg_age__mean_sem_congd_ratio
# log_avg_age__mean_year_state_ratio
# log_max_age__mean_sem_congd_ratio
# log_max_age__mean_year_state_ratio
# log_min_age__mean_sem_congd_ratio
# log_min_age__mean_year_state_ratio
# 
# n_adult__sum_sem_congd_ratio
# n_adult__sum_year_state_ratio
# n_adult__mean_sem_congd_ratio
# n_adult__mean_year_state_ratio
# 
# n_teen__sum_sem_congd_ratio
# n_teen__sum_year_state_ratio
# n_teen__mean_sem_congd_ratio
# n_teen__mean_year_state_ratio
# 
# n_child__sum_sem_congd_ratio
# n_child__sum_year_state_ratio
# n_child__mean_sem_congd_ratio
# n_child__mean_year_state_ratio
# 
# log_n_adult__mean_sem_congd_ratio
# log_n_adult__mean_year_state_ratio
# 
# log_n_teen__mean_sem_congd_ratio
# log_n_teen__mean_year_state_ratio
# 
# log_n_child__mean_sem_congd_ratio
# log_n_child__mean_year_state_ratio
# 
# log_avg_age__sum_sem_congd_ratio
# log_avg_age__sum_year_state_ratio
# log_max_age__sum_sem_congd_ratio
# log_max_age__sum_year_state_ratio
# log_min_age__sum_sem_congd_ratio
# log_min_age__sum_year_state_ratio
# 
# log_n_adult__sum_sem_congd_ratio
# log_n_adult__sum_year_state_ratio
# log_n_teen__sum_sem_congd_ratio
# log_n_teen__sum_year_state_ratio
# log_n_child__sum_sem_congd_ratio
# log_n_child__sum_year_state_ratio
# 
# entropy_n_child_fixing_year_sem_state_congd
# entropy_n_child_fixing_year_state
# entropy_n_teen_fixing_year_sem_state_congd
# entropy_n_teen_fixing_year_state
# entropy_n_adult_fixing_year_sem_state_congd
# entropy_n_adult_fixing_year_state
# 
# entropy_min_age_fixing_year_sem_state_congd
# entropy_min_age_fixing_year_state
# entropy_avg_age_fixing_year_sem_state_congd
# entropy_avg_age_fixing_year_state
# entropy_max_age_fixing_year_sem_state_congd
# entropy_max_age_fixing_year_stat
# 
# 
# **Group participants on total participants number indicators**:
# 
# n_males_n_participants_ratio
# n_females_n_participants_ratio
# 
# n_killed_n_participants_ratio
# n_injured_n_participants_ratio
# n_arrested_n_participants_ratio
# n_unharmed_n_participants_ratio
# 
# n_adult_n_participants_ratio
# n_teen_n_participants_ratio
# n_child_n_participants_ratio
# 
# log_n_males_n_participants_ratio
# log_n_females_n_participants_ratio
# log_n_killed_n_participants_ratio
# log_n_injured_n_participants_ratio
# log_n_arrested_n_participants_ratio
# log_n_unharmed_n_participants_ratio
# log_n_adult_n_participants_ratio
# log_n_teen_n_participants_ratio
# log_n_child_n_participants_ratio
# 
# entropy_n_child_n_teen_n_adult_fixing_year_sem_state_congd

# %% [markdown]
# ## Utilities

# %% [markdown]
# ### Visualize Features Distribuited

# %%
def plot_distribuition(df, attribute_list, log_scale=False):
    n_rows = int(np.ceil(len(attribute_list)/3))
    fig, ax = plt.subplots(n_rows, 3, figsize=(20, 5*n_rows))

    for i, attribute in enumerate(attribute_list):
        # plot normal distribution with the same mean and standard deviation than the attribute
        y = np.random.normal(
            df[attribute].mean(), 
            df[attribute].std(),
            len(df[attribute]))
        ax[i//3, i%3].hist(y, bins=len(df[attribute].unique()), 
            color='red', alpha=0.8, label='Normal distribution')

        # plot attribute distribuition
        ax[i//3, i%3].hist(df[attribute], bins=len(df[attribute].unique()),
            color='blue', alpha=0.7, label=attribute)
        
        ax[i//3, i%3].set_title(f'{attribute} distribuition')
        ax[i//3, i%3].legend(loc='upper right')
        ax[i//3, i%3].set_xlim(df[attribute].min(), df[attribute].max())
        if log_scale:
            ax[i//3, i%3].set_yscale('log')

    plt.suptitle('Distribuition of the attributes', fontsize=20)
    plt.show()

# %% [markdown]
# ### Visualize Distribuition and Outliers

# %%
def plot_violinplot(df, attribute_list, log_scale=False):
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.violinplot(data=df[attribute_list], ax=ax)
    plt.xticks(rotation=90, ha='right')
    if log_scale:
        plt.yscale('log')
    plt.show

def plot_boxplot(df, attribute_list, log_scale=False):
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=df[attribute_list], ax=ax)
    plt.xticks(rotation=90, ha='right')
    if log_scale:
        plt.yscale('log')
    plt.show

# %% [markdown]
# ### Correlation Coefficient
# 
# 1. **Pearson Correlation Coefficient:**
#    - Measures the linear relationship between two continuous variables.
#    - Assumes that the variables are approximately normally distributed.
#    - Sensitive to outliers.
#    - Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.
# 
# 2. **Spearman Rank Correlation Coefficient:**
#    - Measures the strength and direction of the monotonic relationship between two variables.
#    - Does not assume that the variables are normally distributed.
#    - Based on the ranks of the data rather than the actual values.
#    - Less sensitive to outliers than the Pearson correlation.
#    - Values range from -1 (perfect inverse monotonic relationship) to 1 (perfect monotonic relationship), with 0 indicating no monotonic correlation.
# 
# 3. **Kendall Tau Rank Correlation Coefficient:**
#    - Also measures the strength and direction of the monotonic relationship between two variables.
#    - Like Spearman, it does not assume that the variables are normally distributed.
#    - Based on the number of concordant and discordant pairs of observations.
#    - Less affected by outliers.
#    - Values range from -1 to 1, with 0 indicating no correlation. The interpretation is similar to Spearman.
# 
# **Summary:**
# - Use Pearson when the relationship between variables is expected to be linear and the data is approximately normally distributed.
# - Use Spearman or Kendall when the relationship is expected to be monotonic (but not necessarily linear) or when the data is not normally distributed.
# - Spearman and Kendall are often more robust in the presence of outliers.

# %%
def plot_correlation_heatmap(df, attribute_list, method=['pearson', 'kendall', 'spearman']):
    fig, axs = plt.subplots(nrows=len(method), ncols=1, figsize=(10, 5*len(method)))

    for i, m in enumerate(method):
        sns.heatmap(df[attribute_list].corr(m), ax=axs[i], annot=True, cmap='coolwarm',
            mask=np.triu(np.ones_like(df[attribute_list].corr(m), dtype=bool)))
        axs[i].set_title(f'{m.capitalize()} Correlation Heatmap')
        # add space between subplots
    fig.tight_layout()

# %% [markdown]
# ## Number of participants indicators

# %% [markdown]
# 'n_participants'
# 
# n_participants__sum_sem_congd_ratio
# n_participants__sum_year_state_ratio
# n_participants__mean_sem_congd_ratio
# n_participants__mean_year_state_ratio
# 
# log_n_participants__mean_sem_congd_ratio
# log_n_participants__mean_year_state_ratio
# 
# log_n_participants__sum_sem_congd_ratio
# log_n_participants__sum_year_state_ratio

# %%
attribute_list = [
    'n_participants',
    'n_participants__sum_sem_congd_ratio',
    'n_participants__sum_year_state_ratio',
    'n_participants__mean_sem_congd_ratio',
    'n_participants__mean_year_state_ratio',
    'log_n_participants__mean_sem_congd_ratio',
    'log_n_participants__mean_year_state_ratio',
    'log_n_participants__sum_sem_congd_ratio',
    'log_n_participants__sum_year_state_ratio']

# %%
plot_distribuition(df=indicators_df, attribute_list=attribute_list)

# %%
plot_violinplot(df=indicators_df, attribute_list=attribute_list[1:])
plot_boxplot(df=indicators_df, attribute_list=attribute_list[1:])

# %%
plot_correlation_heatmap(df=indicators_df, attribute_list=attribute_list)

# %% [markdown]
# Person: se ho distribuzione simile a normale
# 
# Kendall: se ci sono molti outliers
# 
# Spearman: se non faccio assunzioni sulla distribuzione dei dati, ma è sensibile agli outlier

# %% [markdown]
# 'n_participants__mean_year_state_ratio': alta correlazione con gli altri indici non scelti, ha molti outlier
# 
# 'log_n_participants__sum_sem_congd_ratio': scorrelato da quello sopra, carina la distribuzione

# %%
final_attribute_list = ['n_participants__mean_year_state_ratio', 'log_n_participants__sum_sem_congd_ratio']

# %% [markdown]
# ## Gender participants indicators

# %% [markdown]
# 'n_males', 'n_females', 'n_males_pr'
# 
# 'log_n_males_n_males_mean_semest_congd_ratio',
# 
# n_males__sum_sem_congd_ratio
# n_males__sum_year_state_ratio
# n_males__mean_sem_congd_ratio
# n_males__mean_year_state_ratio
# 
# n_females__sum_sem_congd_ratio
# n_females__sum_year_state_ratio
# n_females__mean_sem_congd_ratio
# n_females__mean_year_state_ratio
# 
# log_n_males__mean_sem_congd_ratio
# log_n_males__mean_year_state_ratio
# log_n_females__mean_sem_congd_ratio
# log_n_females__mean_year_state_ratio
# 
# log_n_males__sum_sem_congd_ratio
# log_n_males__sum_year_state_ratio
# log_n_females__sum_sem_congd_ratio
# log_n_females__sum_year_state_ratio

# %%
attribute_list = [
    'n_males', 'n_females', 'n_males_pr',
    'log_n_males_n_males_mean_semest_congd_ratio',

    'n_males__sum_sem_congd_ratio',
    'n_males__sum_year_state_ratio',
    'n_males__mean_sem_congd_ratio',
    'n_males__mean_year_state_ratio',

    'n_females__sum_sem_congd_ratio',
    'n_females__sum_year_state_ratio',
    'n_females__mean_sem_congd_ratio',
    'n_females__mean_year_state_ratio',

    'log_n_males__mean_sem_congd_ratio',
    'log_n_males__mean_year_state_ratio',
    'log_n_females__mean_sem_congd_ratio',
    'log_n_females__mean_year_state_ratio',

    'log_n_males__sum_sem_congd_ratio',
    'log_n_males__sum_year_state_ratio',
    'log_n_females__sum_sem_congd_ratio',
    'log_n_females__sum_year_state_ratio'
    ]

# %%
plot_distribuition(df=indicators_df, attribute_list=attribute_list)

# %%
attribute_list = [
    'n_males', 'n_females', 'n_males_pr',
    'log_n_males_n_males_mean_semest_congd_ratio',
    'n_males__sum_sem_congd_ratio',
    'n_males__mean_sem_congd_ratio',
    'n_males__mean_year_state_ratio',
    'log_n_males__mean_sem_congd_ratio',
    'log_n_males__mean_year_state_ratio',
    'log_n_males__sum_sem_congd_ratio',
    'log_n_males__sum_year_state_ratio',
    ]

# %%
#plot_violinplot(df=indicators_df, attribute_list=attribute_list)
#plot_boxplot(df=indicators_df, attribute_list=attribute_list)

# %%
attribute_list = [
    'n_males', 'n_males_pr',
    'log_n_males_n_males_mean_semest_congd_ratio',
    'n_males__sum_sem_congd_ratio',
    'n_males__mean_sem_congd_ratio',
    'log_n_males__mean_sem_congd_ratio',
    'log_n_males__sum_sem_congd_ratio',
    'log_n_males__sum_year_state_ratio',
    ]

# %%
plot_correlation_heatmap(df=indicators_df, attribute_list=attribute_list)

# %%
final_attribute_list.append('n_males_pr')
final_attribute_list.append('n_males__sum_sem_congd_ratio')
final_attribute_list.append('n_males__mean_sem_congd_ratio')

# %% [markdown]
# ## Killed/injured indicators

# %% [markdown]
# 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed'
# 
# 'n_killed_pr', 'n_injured_pr', 'n_arrested_pr', 'n_unharmed_pr'
# 
# n_killed__sum_sem_congd_ratio
# n_killed__sum_year_state_ratio
# n_killed__mean_sem_congd_ratio
# n_killed__mean_year_state_ratio
# 
# n_injured__sum_sem_congd_ratio
# n_injured__sum_year_state_ratio
# n_injured__mean_sem_congd_ratio
# n_injured__mean_year_state_ratio
# 
# n_arrested__sum_sem_congd_ratio
# n_arrested__sum_year_state_ratio
# n_arrested__mean_sem_congd_ratio
# n_arrested__mean_year_state_ratio
# 
# n_unharmed__sum_sem_congd_ratio
# n_unharmed__sum_year_state_ratio
# n_unharmed__mean_sem_congd_ratio
# n_unharmed__mean_year_state_ratio
# 
# log_n_killed__mean_sem_congd_ratio
# log_n_killed__mean_year_state_ratio
# log_n_injured__mean_sem_congd_ratio
# log_n_injured__mean_year_state_ratio
# log_n_arrested__mean_sem_congd_ratio
# log_n_arrested__mean_year_state_ratio
# log_n_unharmed__mean_sem_congd_ratio
# log_n_unharmed__mean_year_state_ratio
# 
# log_n_killed__sum_sem_congd_ratio
# log_n_killed__sum_year_state_ratio
# log_n_injured__sum_sem_congd_ratio
# log_n_injured__sum_year_state_ratio
# log_n_arrested__sum_sem_congd_ratio
# log_n_arrested__sum_year_state_ratio
# log_n_unharmed__sum_sem_congd_ratio
# log_n_unharmed__sum_year_state_ratio

# %%
attribute_list = ['n_killed', 'n_injured', 'n_arrested', 'n_unharmed',
'n_killed_pr', 'n_injured_pr', 'n_arrested_pr', 'n_unharmed_pr']

# %%
#plot_distribuition(df=indicators_df, attribute_list=attribute_list)
#plot_boxplot(df=indicators_df, attribute_list=attribute_list)
plot_correlation_heatmap(df=indicators_df, attribute_list=attribute_list)

# %% [markdown]
# Qui terrei le proporzioni

# %%
attribute_list = [
    'n_killed__sum_sem_congd_ratio',
    'n_killed__sum_year_state_ratio',
    'n_killed__mean_sem_congd_ratio',
    'n_killed__mean_year_state_ratio',

    'n_injured__sum_sem_congd_ratio',
    'n_injured__sum_year_state_ratio',
    'n_injured__mean_sem_congd_ratio',
    'n_injured__mean_year_state_ratio',

    'n_arrested__sum_sem_congd_ratio',
    'n_arrested__sum_year_state_ratio',
    'n_arrested__mean_sem_congd_ratio',
    'n_arrested__mean_year_state_ratio',

    'n_unharmed__sum_sem_congd_ratio',
    'n_unharmed__sum_year_state_ratio',
    'n_unharmed__mean_sem_congd_ratio',
    'n_unharmed__mean_year_state_ratio']

# %%
#plot_distribuition(df=indicators_df, attribute_list=attribute_list)
#plot_boxplot(df=indicators_df, attribute_list=attribute_list)
plot_correlation_heatmap(df=indicators_df, attribute_list=attribute_list)

# %%
attribute_list = [
    'log_n_killed__mean_sem_congd_ratio',
    'log_n_killed__mean_year_state_ratio',
    'log_n_injured__mean_sem_congd_ratio',
    'log_n_injured__mean_year_state_ratio',
    'log_n_arrested__mean_sem_congd_ratio',
    'log_n_arrested__mean_year_state_ratio',
    'log_n_unharmed__mean_sem_congd_ratio',
    'log_n_unharmed__mean_year_state_ratio',

    'log_n_killed__sum_sem_congd_ratio',
    'log_n_killed__sum_year_state_ratio',
    'log_n_injured__sum_sem_congd_ratio',
    'log_n_injured__sum_year_state_ratio',
    'log_n_arrested__sum_sem_congd_ratio',
    'log_n_arrested__sum_year_state_ratio',
    'log_n_unharmed__sum_sem_congd_ratio',
    'log_n_unharmed__sum_year_state_ratio']

# %%
#plot_distribuition(df=indicators_df, attribute_list=attribute_list)
#plot_boxplot(df=indicators_df, attribute_list=attribute_list)
#plot_correlation_heatmap(df=indicators_df, attribute_list=attribute_list)

# %%
final_attribute_list += ['n_killed_pr', 'n_injured_pr', 'n_arrested_pr', 'n_unharmed_pr',
 'n_killed__mean_sem_congd_ratio', 'n_killed__sum_sem_congd_ratio',
 'n_injured__mean_sem_congd_ratio', 'n_injured__sum_sem_congd_ratio',
 'n_arrested__mean_sem_congd_ratio', 'n_arrested__sum_sem_congd_ratio',
 'n_unharmed__mean_sem_congd_ratio', 'n_unharmed__sum_sem_congd_ratio']

# %% [markdown]
# ## Others indicators

# %%
final_attribute_list += ['n_participants_child_prop', 'n_participants_teen_prop', 'n_adults_entropy', 'age_range', 'min_age']

# %%
final_attribute_list += [
    'location_importance', 'city_entropy', 'address_entropy',  
    'lat_proj_x', 'lon_proj_x',
    'entropy_n_child_n_teen_n_adult_fixing_year_sem_state_congd',
    'tags_entropy', 'severity']

# %% [markdown]
# ## Final

# %%
#plot_boxplot(df=indicators_df, attribute_list=final_attribute_list)

# %%
final_indicators_list = [
    #'log_n_participants__sum_sem_congd_ratio',
    'n_males_pr',
    'n_males__sum_sem_congd_ratio',
    'n_killed_pr', # correlato con severity, non so se lasciare quello e togliere le proporzioni
    'n_injured_pr',
    'n_arrested_pr',
    'n_unharmed_pr',
    #'n_killed__sum_sem_congd_ratio', # correlato con n_killed_pr
    #'n_injured__sum_sem_congd_ratio',
    #'n_arrested__sum_sem_congd_ratio',
    #'n_unharmed__sum_sem_congd_ratio',
    'n_participants_child_prop',
    'n_participants_teen_prop',
    #'n_adults_entropy', # correlato con entropy_n_child_n_teen_n_adult_fixing_year_sem_state_congd
    'age_range',
    'min_age',
    'location_importance',
    'city_entropy',
    'address_entropy',
    #'lat_proj_x', # non li ho plottati perchè fuori range rispetto agli altri
    #'lon_proj_x',
    'entropy_n_child_n_teen_n_adult_fixing_year_sem_state_congd',
    'tags_entropy',
    'severity']

# %%
plot_boxplot(df=indicators_df, attribute_list=final_indicators_list)

# %%
plt.figure(figsize=(20, 8))
sns.heatmap(indicators_df[final_indicators_list].corr('kendall'), annot=True, cmap='coolwarm',
    mask=np.triu(np.ones_like(indicators_df[final_indicators_list].corr('kendall'), dtype=bool)),
    xticklabels=True, yticklabels=False)
plt.show()


