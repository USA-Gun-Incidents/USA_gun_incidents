# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
males_deaths = pd.read_csv('data/deaths_males.txt', sep='\t')

# %%
males_deaths.head(n=40)

# %%
males_deaths['Single-Year Ages'].unique()

# %%
males_deaths['Gender'].unique()

# %%
males_deaths['Year'].unique()

# %%
males_deaths['State'].unique()

# %%
males_deaths[males_deaths['Single-Year Ages']=='Not Stated']

# %%
males_deaths[(males_deaths['Deaths']=='0') & (males_deaths['Single-Year Ages']=='Not Stated')]

# %%
males_deaths[(males_deaths['Deaths']=='Suppressed') & (males_deaths['Single-Year Ages']=='Not Stated')]

# %%
males_deaths[(males_deaths['Deaths']!='0') & (males_deaths['Deaths']!='Suppressed') & (males_deaths['Single-Year Ages']=='Not Stated')]

# %%
males_deaths[males_deaths['Deaths']=='Suppressed']

# %%
ages = males_deaths['Single-Year Ages'].unique()
child_ages = [ages[i] for i in range(12)]
teen_ages = [ages[i] for i in range(12, 18)]
adult_ages = [ages[i] for i in range(18, len(ages)-1)]

# %%
males_deaths['age_group'] = males_deaths['Single-Year Ages'].apply(
    lambda x: 'child' if x in child_ages else ('teen' if x in teen_ages else ('Unknown' if x=='Not Stated' else 'adult')))
males_deaths['deaths_temp'] = males_deaths['Deaths'].apply(lambda x: 1 if x=='Suppressed' else int(x))
males_deaths_grouped = males_deaths.groupby(['State', 'Year', 'age_group']).sum()['deaths_temp'].unstack()
males_deaths_grouped

# %%
males_deaths_grouped['total'] = males_deaths_grouped.sum(axis=1)
males_deaths_grouped['prop_child'] = males_deaths_grouped['child']/males_deaths_grouped['total']
males_deaths_grouped['prop_teen'] = males_deaths_grouped['teen']/males_deaths_grouped['total']
males_deaths_grouped['prop_adult'] = males_deaths_grouped['adult']/males_deaths_grouped['total']
males_deaths_grouped

# %%
males_deaths_grouped.describe()

# %%
males_deaths['Deaths'] = males_deaths['Deaths'].apply(lambda x: 1 if x=='Suppressed' else int(x)) # se <18 1, se >= 18 5?
males_deaths['age_group'] = males_deaths['Single-Year Ages'].apply(
    lambda x: 'child' if x in child_ages else ('teen' if x in teen_ages else 'adult'))
males_deaths_grouped = males_deaths.groupby(['State', 'Year', 'age_group']).sum()['Deaths'].unstack()
males_deaths_grouped['total'] = males_deaths_grouped.sum(axis=1)
males_deaths_grouped['prop_child'] = males_deaths_grouped['child']/males_deaths_grouped['total']
males_deaths_grouped['prop_teen'] = males_deaths_grouped['teen']/males_deaths_grouped['total']
males_deaths_grouped['prop_adult'] = males_deaths_grouped['adult']/males_deaths_grouped['total']
males_deaths_grouped.describe()

# %%
females_deaths = pd.read_csv('data/deaths_females.txt', sep='\t')

# %%
females_deaths[females_deaths['Deaths']=='Suppressed'] # sono diversi da quelli sopra

# %%
females_deaths['age_group'] = females_deaths['Single-Year Ages'].apply(
    lambda x: 'child' if x in child_ages else ('teen' if x in teen_ages else ('Unknown' if x=='Not Stated' else 'adult')))
females_deaths['deaths_temp'] = females_deaths['Deaths'].apply(lambda x: 1 if x=='Suppressed' else int(x))
females_deaths_grouped = females_deaths.groupby(['State', 'Year', 'age_group']).sum()['deaths_temp'].unstack()
females_deaths_grouped['total'] = females_deaths_grouped.sum(axis=1)
females_deaths_grouped['prop_child'] = females_deaths_grouped['child']/females_deaths_grouped['total']
females_deaths_grouped['prop_teen'] = females_deaths_grouped['teen']/females_deaths_grouped['total']
females_deaths_grouped['prop_adult'] = females_deaths_grouped['adult']/females_deaths_grouped['total']
females_deaths_grouped.describe()

# %%
females_deaths['Deaths'] = females_deaths['Deaths'].apply(lambda x: 1 if x=='Suppressed' else int(x)) # se <18 1, se >= 18 5?
females_deaths['age_group'] = females_deaths['Single-Year Ages'].apply(
    lambda x: 'child' if x in child_ages else ('teen' if x in teen_ages else 'adult'))
females_deaths_grouped = females_deaths.groupby(['State', 'Year', 'age_group']).sum()['Deaths'].unstack()
females_deaths_grouped['total'] = females_deaths_grouped.sum(axis=1)
females_deaths_grouped['prop_child'] = females_deaths_grouped['child']/females_deaths_grouped['total']
females_deaths_grouped['prop_teen'] = females_deaths_grouped['teen']/females_deaths_grouped['total']
females_deaths_grouped['prop_adult'] = females_deaths_grouped['adult']/females_deaths_grouped['total']
females_deaths_grouped.describe()

# %%
males_deaths_grouped.reset_index(inplace=True)
males_deaths_grouped.drop(columns=['total', 'prop_child', 'prop_teen', 'prop_adult'], inplace=True)
males_deaths_grouped.rename(columns={'State': 'state', 'Year': 'year', 'child': 'male_child', 'teen': 'male_teen', 'adult': 'male_adult'}, inplace=True)

females_deaths_grouped.reset_index(inplace=True)
females_deaths_grouped.drop(columns=['total', 'prop_child', 'prop_teen', 'prop_adult'], inplace=True)
females_deaths_grouped.rename(columns={'State': 'state', 'Year': 'year', 'child': 'female_child', 'teen': 'female_teen', 'adult': 'female_adult'}, inplace=True)

deaths = females_deaths_grouped.merge(males_deaths_grouped, on=['state', 'year'], how='left')
deaths = deaths[['state', 'year', 'male_child', 'male_teen', 'male_adult', 'female_child', 'female_teen', 'female_adult']]
deaths.to_csv('data/deaths.csv', index=False)


