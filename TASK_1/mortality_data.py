# %% [markdown]
# # Preparation of Mortality Data
#
# Mortality data was downloaded from [CDC WONDER](https://wonder.cdc.gov) through the request form. Since the number of records we wanted to dowload exceeded the maximum number of records that can be downloaded at once, we had to do two queries, one for males and one for females. This notebook combines the two datasets and prepares the data for the analysis.
#
# More information about the data can be found at the following URL: [Multiple Cause of Death 1999 - 2020](https://wonder.cdc.gov/wonder/help/mcd.html).

# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# notebook settings
pd.set_option('display.max_colwidth', -1)

# %%
# display metadata
males_metadata = pd.read_fwf('../data/deaths_males_metadata.txt', header=None)
males_metadata

# %%
males_deaths = pd.read_csv('../data/deaths_males.txt', sep='\t')
males_deaths.head(n=5)

# %%
males_deaths.info()

# %%
ages = males_deaths['Single-Year Ages'].unique()
ages

# %% [markdown]
# We notice there are also "Not Stated" ages.

# %%
males_deaths[males_deaths['Single-Year Ages']=='Not Stated']

# %%
child_ages = [ages[i] for i in range(12)]
teen_ages = [ages[i] for i in range(12, 18)]
adult_ages = [ages[i] for i in range(18, len(ages)-1)]

# %%
males_deaths['Gender'].unique()

# %%
males_deaths['Year'].unique()

# %%
males_deaths['State'].unique()

# %% [markdown]
# For privacy reasons, deaths are suppressed when the count is between 1 and 9.

# %%
males_deaths[(males_deaths['Deaths']=='Suppressed')]

# %%
males_deaths['age_group'] = males_deaths['Single-Year Ages'].apply(
    lambda x: 'male_child' if x in child_ages else ('male_teen' if x in teen_ages else ('male_not_stated' if x=='Not Stated' else 'male_adult')))
males_deaths['Deaths_temp'] = pd.to_numeric(males_deaths['Deaths'], errors='coerce') # temporary discard suppressed values
males_deaths_grouped = males_deaths.groupby(['State', 'Year', 'age_group'])['Deaths_temp'].sum().unstack()
males_deaths_grouped['male_total'] = males_deaths_grouped.sum(axis=1)
males_deaths_grouped['perc_male_child'] = (males_deaths_grouped['male_child']/males_deaths_grouped['male_total'])*100
males_deaths_grouped['perc_male_teen'] = (males_deaths_grouped['male_teen']/males_deaths_grouped['male_total'])*100
males_deaths_grouped['perc_male_adult'] = (males_deaths_grouped['male_adult']/males_deaths_grouped['male_total'])*100
males_deaths_grouped.describe()

# %% [markdown]
# We notice that the maximum values of perc_child and perc_teen is negligible w.r.t perc_adult. Therefore, when age is not stated we can assume it is adult:

# %%
males_deaths['age_group'] = males_deaths['Single-Year Ages'].apply(
    lambda x: 'male_child' if x in child_ages else ('male_teen' if x in teen_ages else 'male_adult'))

# %% [markdown]
# Also, for each state, we replace suppressed values with the mean number of deaths for that age if it is smaller or equal than 9; oterwise we replace it with 9 (the maximum for a suppressed value). If for a given age the number of deaths is always suppressed, we replace suppresed values with 1.

# %%
males_deaths['Deaths_temp'] = males_deaths['Deaths']
males_deaths['Deaths'] = males_deaths['Deaths'].apply(lambda x: 1 if x=='Suppressed' else int(x))
mean_deaths_per_state = males_deaths[males_deaths['Deaths_temp']!='Suppressed'].groupby(['State', 'Single-Year Ages'])['Deaths'].mean()
mean_deaths_per_state[mean_deaths_per_state>9] = 9
mean_deaths_per_state = mean_deaths_per_state.astype(int)
males_deaths['Deaths'] = males_deaths.apply(lambda x: mean_deaths_per_state.loc[x['State'], x['Single-Year Ages']] if x['Deaths_temp']=='Suppressed' and (x['State'],x['Single-Year Ages']) in mean_deaths_per_state.index else x['Deaths'], axis=1)
males_deaths.drop(columns=['Deaths_temp'], inplace=True)
males_deaths_grouped = males_deaths.groupby(['State', 'Year', 'age_group'])['Deaths'].sum().unstack()
males_deaths_grouped.reset_index(inplace=True)
males_deaths_grouped.rename(columns={'State': 'state', 'Year': 'year'}, inplace=True)
males_deaths_grouped

# %% [markdown]
# Now we explore and fix females deaths:

# %%
# display metadata
females_metadata = pd.read_fwf('../data/deaths_females_metadata.txt', header=None)
females_metadata

# %%
females_deaths = pd.read_csv('../data/deaths_females.txt', sep='\t')
females_deaths.head(n=5)

# %%
females_deaths[(females_deaths['Deaths']=='Suppressed')]

# %% [markdown]
# We notice that suppressed values are different from those of males. Hence, the information about the gender was not suppressed and we do not need to care about counting deaths twice.

# %%
females_deaths['age_group'] = females_deaths['Single-Year Ages'].apply(
    lambda x: 'female_child' if x in child_ages else ('female_teen' if x in teen_ages else ('female_not_stated' if x=='Not Stated' else 'female_adult')))
females_deaths['Deaths_temp'] = pd.to_numeric(females_deaths['Deaths'], errors='coerce') # temporary discard suppressed values
females_deaths_grouped = females_deaths.groupby(['State', 'Year', 'age_group'])['Deaths_temp'].sum().unstack()
females_deaths_grouped['female_total'] = females_deaths_grouped.sum(axis=1)
females_deaths_grouped['perc_female_child'] = (females_deaths_grouped['female_child']/females_deaths_grouped['female_total'])*100
females_deaths_grouped['perc_female_teen'] = (females_deaths_grouped['female_teen']/females_deaths_grouped['female_total'])*100
females_deaths_grouped['perc_female_adult'] = (females_deaths_grouped['female_adult']/females_deaths_grouped['female_total'])*100
females_deaths_grouped.describe()

# %% [markdown]
# Even for females, the maximum values of perc_child and perc_teen is negligible w.r.t perc_adult. When age is not stated we can again assume it is adult:

# %%
females_deaths['age_group'] = females_deaths['Single-Year Ages'].apply(
    lambda x: 'female_child' if x in child_ages else ('female_teen' if x in teen_ages else 'female_adult'))

# %% [markdown]
# We replace suppressed values as we did for males:

# %%
females_deaths['Deaths_temp'] = females_deaths['Deaths']
females_deaths['Deaths'] = females_deaths['Deaths'].apply(lambda x: 1 if x=='Suppressed' else int(x))
mean_deaths_per_state = females_deaths[females_deaths['Deaths_temp']!='Suppressed'].groupby(['State', 'Single-Year Ages'])['Deaths'].mean()
mean_deaths_per_state[mean_deaths_per_state>9] = 9
mean_deaths_per_state = mean_deaths_per_state.astype(int)
females_deaths['Deaths'] = females_deaths.apply(lambda x: mean_deaths_per_state.loc[x['State'], x['Single-Year Ages']] if x['Deaths_temp']=='Suppressed' and (x['State'],x['Single-Year Ages']) in mean_deaths_per_state.index else x['Deaths'], axis=1)
females_deaths.drop(columns=['Deaths_temp'], inplace=True)
females_deaths_grouped = females_deaths.groupby(['State', 'Year', 'age_group'])['Deaths'].sum().unstack()
females_deaths_grouped.reset_index(inplace=True)
females_deaths_grouped.rename(columns={'State': 'state', 'Year': 'year'}, inplace=True)
females_deaths_grouped

# %% [markdown]
# We merge the two datasets:

# %%
deaths = females_deaths_grouped.merge(males_deaths_grouped, on=['state', 'year'], how='left')
deaths = deaths[['state', 'year', 'male_child', 'male_teen', 'male_adult', 'female_child', 'female_teen', 'female_adult']]
deaths

# %% [markdown]
# We save the result:

# %%
deaths.to_csv('../data/deaths.csv', index=False)
