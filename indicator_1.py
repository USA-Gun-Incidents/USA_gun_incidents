# %%
import pandas as pd
import numpy as np
from plot_utils import hist_box_plot
import matplotlib.pyplot as plt
%matplotlib inline

# %%
incidents_df = pd.read_csv('./data/final.csv', index_col=0)
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')

# %%
incidents_df['semester'] = (incidents_df['date'].dt.month // 7) + 1

# %%
incidents_df['city'] = incidents_df['city'].fillna('nan') # to treat all points without city as belonging to a fake city

# %%
def compute_ratio_indicator(df, gby, num, den, suffix, agg_fun):
    grouped_df = df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    df[num+'_'+den+suffix+'_ratio'] = df[num] / df[den+suffix]
    df.drop(columns=[den+suffix], inplace=True)
    return df

incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state'], 'n_males', 'n_males', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state'], 'n_females', 'n_females', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state'], 'n_males', 'n_participants', '_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_males', '_tot_year_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'congressional_district'], 'n_females', 'n_females', '_tot_year_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_participants', '_year_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'county'], 'n_males', 'n_males', '_tot_year_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'county'], 'n_females', 'n_females', '_tot_year_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'county'], 'n_males', 'n_participants', '_year_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_tot_year_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_tot_year_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'city'], 'n_males', 'n_participants', '_year_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_mean_year_city', 'mean')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_median_year_city', 'median')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_mean_year_city', 'mean')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_median_year_city', 'median')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state'], 'n_males', 'n_males', '_tot_semester_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state'], 'n_females', 'n_females', '_tot_semester_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state'], 'n_males', 'n_participants', '_semester_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_males', 'n_males', '_tot_semester_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_females', 'n_females', '_tot_semester_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'congressional_district'], 'n_males', 'n_participants', '_semester_congdist', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'county'], 'n_males', 'n_males', '_tot_semester_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'county'], 'n_females', 'n_females', '_tot_semester_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'county'], 'n_males', 'n_participants', '_semester_county', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'city'], 'n_males', 'n_males', '_tot_semester_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'city'], 'n_females', 'n_females', '_tot_semester_city', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'semester', 'state', 'city'], 'n_males', 'n_participants', '_semester_city', 'sum')

# %%
incidents_df.columns

# %%
ratios_wrt_tot = []
ratios_wrt_mean = []
ratios_wrt_median = []
for att in incidents_df.columns:
    if 'ratio' in att:
        if 'mean' in att:
            ratios_wrt_mean.append(att)
        elif 'median' in att:
            ratios_wrt_median.append(att)
        else:
            ratios_wrt_tot.append(att)

# %%
incidents_df[ratios_wrt_tot].describe() # females quantiles are 0 (that's why they suggested to do it for males only)

# %%
incidents_df[ratios_wrt_mean].describe()

# %%
incidents_df.boxplot(
    column=ratios_wrt_tot,
    rot=90,
    figsize=(20, 10)
) # city has the higher box (that's why they suggested to do it for cities only)

# %%
log_ratio_wrt_tot = ['log_'+col for col in ratios_wrt_tot]
incidents_df[log_ratio_wrt_tot] = np.log(incidents_df[ratios_wrt_tot]+0.000001)
incidents_df.boxplot(
    column=log_ratio_wrt_tot,
    rot=90,
    figsize=(20, 10)
)

# %%
sqrt_ratio_wrt_tot = ['sqrt_'+col for col in ratios_wrt_tot]
incidents_df[sqrt_ratio_wrt_tot] = np.sqrt(incidents_df[ratios_wrt_tot])
incidents_df.boxplot(
    column=sqrt_ratio_wrt_tot,
    rot=90,
    figsize=(20, 10)
)

# %%
incidents_df.boxplot(
    column=ratios_wrt_mean,
    rot=90
)

# %%
incidents_df.boxplot(
    column=ratios_wrt_median,
    rot=90
)

# %%
incidents_df[ratios_wrt_mean].describe()

# %%
hist_box_plot(
    incidents_df,
    'n_males_n_males_tot_year_city_ratio',
    title='n_males_n_males_tot_year_city_ratio',
    bins=50,
    figsize=(10, 5)
)

# %%
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state'], 'n_participants_child', 'n_participants_child', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state'], 'n_participants_teen', 'n_participants_teen', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants_adult', '_tot_year_state', 'sum')
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state'], 'n_participants_adult', 'n_participants', '_tot_year_state', 'sum')

# %%
incidents_df.boxplot(
    column=[
        'n_participants_child_n_participants_child_tot_year_state_ratio',
        'n_participants_teen_n_participants_teen_tot_year_state_ratio',
        'n_participants_adult_n_participants_adult_tot_year_state_ratio',
        'n_participants_adult_n_participants_tot_year_state_ratio'
        ],
    rot=90
)

# %%
population_df = pd.read_csv('./data/census_bureau.csv')
population_df = population_df[population_df['congressional_district'] != 'ZZ'] # delete rows with obsolete congressional district
population_df['congressional_district'] = population_df['congressional_district'].astype(int)
population_df.loc[population_df['state'] == 'District of Columbia', 'congressional_district'] = 0 # to use the same notion of other dataframes
population_df['state'] = population_df['state'].str.upper()
population_df['n_males'] = population_df['male_child'] + population_df['male_teen'] + population_df['male_adult']
population_df['n_females'] = population_df['female_child'] + population_df['female_teen'] + population_df['female_adult']
population_df


# %%
def compute_ratio_external_indicator(df, ext_df, gby, num, den, suffix, agg_fun):
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    df[num+'_'+den+suffix+'_ratio'] = df[num] / df[den+suffix]
    df.drop(columns=[den+suffix], inplace=True)
    return df

# %%
incidents_df = compute_ratio_external_indicator(incidents_df, population_df, ['year', 'state'], 'n_males', 'n_males', '_pop_year_state', 'sum')
incidents_df = compute_ratio_external_indicator(incidents_df, population_df, ['year', 'state', 'congressional_district'], 'n_males', 'n_males', '_pop_year_cong', 'sum')

# %%
pop_ratios = []
for att in incidents_df.columns:
    if 'pop' in att and 'ratio' in att:
        pop_ratios.append(att)

incidents_df.boxplot(
    column=pop_ratios,
    rot=90
)

# %%
# TODO:
# considerare i tag, età, povertà
# entropia?


