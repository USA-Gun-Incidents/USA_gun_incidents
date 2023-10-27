# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv('data/deaths_males.txt', sep='\t')

# %%
data.head(n=40)

# %%
data['Single-Year Ages'].unique()

# %%
data[data['Single-Year Ages']=='Not Stated']

# %%
data[(data['Deaths']=='0') & (data['Single-Year Ages']=='Not Stated')]

# %%
data[(data['Deaths']=='Suppressed') & (data['Single-Year Ages']=='Not Stated')]

# %%
data[(data['Deaths']!='0') & (data['Deaths']!='Suppressed') & (data['Single-Year Ages']=='Not Stated')]

# %%
data['Gender'].unique()

# %%
data['Year'].unique()

# %%
data['State'].unique()

# %%
data['Single-Year Ages'].unique()

# %%
ages = data['Single-Year Ages'].unique()
child_ages = [ages[i] for i in range(12)]
teen_ages = [ages[i] for i in range(12, 18)]
adult_ages = [ages[i] for i in range(18, len(ages)-1)]

# %%
data['age_group'] = data['Single-Year Ages'].apply(
    lambda x: 'Child 0-11' if x in child_ages else ('Teen 12-17' if x in teen_ages else ('Unknown' if x=='Suppressed' else 'Adult 18+')))

# %%
data['deaths_no_suppressed'] = data['Deaths'].apply(lambda x: 0 if x=='Suppressed' else int(x))
data_to_plot = data.groupby(['State', 'Year', 'age_group']).sum()['deaths_no_suppressed'].unstack()

for state in data['State'].unique():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
    data_to_plot.loc[state, 'Adult 18+'].plot(kind='bar', title=state, ax=axes[0], ylabel='Deaths Adult 18+')
    data_to_plot.loc[state, 'Child 0-11'].plot(kind='bar', title=state, ax=axes[1], ylabel='Deaths Child 0-11')
    data_to_plot.loc[state, 'Teen 12-17'].plot(kind='bar', title=state, ax=axes[2], ylabel='Deaths Teen 12-17')
    break

# %%
data.drop(columns=['Notes', 'State Code', 'Single-Year Ages Code', 'Gender Code', 'Year Code', 'Population', 'Crude Rate'], inplace=True)
data.rename(columns={'State': 'state', 'Gender': 'gender', 'Deaths': 'deaths'}, inplace=True)

# su internet ci sono paper che propongono metodi proprio per questo dataset


