# %%
import pandas as pd
import numpy as np

# %%
incidents_df = pd.read_csv('./data/final.csv', index_col=0)
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')
incidents_df['semester'] = (incidents_df['date'].dt.month // 7) + 1

# %%
incidents_df['Year'] = incidents_df['date'].dt.strftime('%Y')

# %%
incidents_df['Year']

# %%
# group by year, semester, state, congressional district

killed_grouped_df = incidents_df.groupby(['state', 'congressional_district', 'year', 'semester'])["n_killed"].agg("sum")
injured_grouped_df = incidents_df.groupby(['state', 'congressional_district', 'year', 'semester'])["n_injured"].agg("sum")

killed_grouped_df

# %%
injured_grouped_df

# %%
# total killed/injoured in a congressional district

killed_grouped_total_df = incidents_df.groupby(['state', 'congressional_district'])["n_killed"].agg("sum")
injured_grouped_total_df = incidents_df.groupby(['state', 'congressional_district'])["n_injured"].agg("sum")

killed_grouped_total_df

# %%
injured_grouped_total_df

# %%
killed_index = killed_grouped_df.__deepcopy__()
injured_index = injured_grouped_df.__deepcopy__()

# %%
for state in incidents_df["state"].unique():
    for cd in incidents_df.loc[incidents_df["state"] == state]["congressional_district"].unique():
        if not pd.isna(cd):
            total_killed = killed_grouped_total_df[(state, cd)]
            total_injured = injured_grouped_total_df[(state, cd)]
            killed_index[(state, cd)] = killed_index[(state, cd)].apply(lambda x:x/total_killed)
            injured_index[(state, cd)] = injured_index[(state, cd)].apply(lambda x:x/total_injured)

killed_index

# %%
injured_index


