# %%
import pandas as pd
import numpy as np
from plot_utils import hist_box_plot

# %%
incidents_df = pd.read_csv('./data/final.csv', index_col=0)
incidents_df['date'] = pd.to_datetime(incidents_df['date'], format='%Y-%m-%d')

# %%
incidents_df['semester'] = (incidents_df['date'].dt.month // 7) + 1

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
incidents_df = compute_ratio_indicator(incidents_df, ['year', 'state', 'city'], 'n_females', 'n_females', '_mean_year_city', 'mean')
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
ratio_wrt_tot = []
ratio_wrt_mean = []
for att in incidents_df.columns:
    if 'ratio' in att:
        if 'mean' not in att:
            ratio_wrt_tot.append(att)
        else:
            ratio_wrt_mean.append(att)

# %%
incidents_df[ratio_wrt_tot].describe() # females quantiles are 0 (that's why they suggested to do it for males only)

# %%
incidents_df[ratio_wrt_mean].describe()

# %%
incidents_df.boxplot(
    column=ratio_wrt_tot,
    rot=90,
    figsize=(20, 10)
) # city has the higher box (that's why they suggested to do it for cities only)

# %%
log_ratio_wrt_tot = ['log_'+col for col in ratio_wrt_tot]
incidents_df[log_ratio_wrt_tot] = np.log(incidents_df[ratio_wrt_tot]) # to avoid log(0)? sostituire infinito con? sommare eps prima di applicare?
incidents_df.boxplot(
    column=log_ratio_wrt_tot,
    rot=90,
    figsize=(20, 10)
)

# %%
sqrt_ratio_wrt_tot = ['sqrt_'+col for col in ratio_wrt_tot]
incidents_df[sqrt_ratio_wrt_tot] = np.sqrt(incidents_df[ratio_wrt_tot])
incidents_df.boxplot(
    column=sqrt_ratio_wrt_tot,
    rot=90,
    figsize=(20, 10)
)

# %%
incidents_df.boxplot(
    column=ratio_wrt_mean,
    rot=90
)

# %%
hist_box_plot(
    incidents_df,
    'n_males_n_males_tot_year_city_ratio',
    title='n_males_n_males_tot_year_city_ratio',
    bins=50,
    figsize=(10, 5)
)

# %%
population_df = pd.read_csv('./data/census_bureau.csv')
population_df = population_df[population_df['congressional_district'] != 'ZZ'] # delete rows with obsolete congressional district
population_df['congressional_district'] = population_df['congressional_district'].astype(int)
population_df.loc[population_df['state'] == 'District of Columbia', 'congressional_district'] = 0 # to use the same notion of other dataframes
population_df['state'] = population_df['state'].str.upper()
population_df

# %%
population_per_year_state_by_gender = population_df.groupby(['year', 'state'])[['male_child', 'male_teen', 'male_adult', 'female_child', 'female_teen', 'female_adult']].sum()
population_per_year_state_by_gender['n_males'] = population_per_year_state_by_gender['male_child'] + population_per_year_state_by_gender['male_teen'] + population_per_year_state_by_gender['male_adult']
population_per_year_state_by_gender['n_females'] = population_per_year_state_by_gender['female_child'] + population_per_year_state_by_gender['female_teen'] + population_per_year_state_by_gender['female_adult']
population_per_year_state_by_gender = population_per_year_state_by_gender[['n_males', 'n_females']]
population_per_year_state_by_gender

# %%
incidents_df = incidents_df.merge(population_per_year_state_by_gender, on=['year', 'state'], how='left', suffixes=[None, '_pop_year_state'])
incidents_df['n_males_pop_year_state_ratio'] = incidents_df['n_males'] / incidents_df['n_males_pop_year_state']
incidents_df['n_females_pop_year_state_ratio'] = incidents_df['n_females'] / incidents_df['n_females_pop_year_state']

# %%
population_per_year_congdist_by_gender = population_df.groupby(['year', 'state', 'congressional_district'])[['male_child', 'male_teen', 'male_adult', 'female_child', 'female_teen', 'female_adult']].sum()
population_per_year_congdist_by_gender['n_males'] = population_per_year_congdist_by_gender['male_child'] + population_per_year_congdist_by_gender['male_teen'] + population_per_year_congdist_by_gender['male_adult']
population_per_year_congdist_by_gender['n_females'] = population_per_year_congdist_by_gender['female_child'] + population_per_year_congdist_by_gender['female_teen'] + population_per_year_congdist_by_gender['female_adult']
population_per_year_congdist_by_gender = population_per_year_congdist_by_gender[['n_males', 'n_females']]
population_per_year_congdist_by_gender

# %%
incidents_df = incidents_df.merge(population_per_year_congdist_by_gender, on=['year', 'state'], how='left', suffixes=[None, '_pop_year_congdist'])
incidents_df['n_males_pop_year_congdist_ratio'] = incidents_df['n_males'] / incidents_df['n_males_pop_year_congdist']
incidents_df['n_females_pop_year_congdist_ratio'] = incidents_df['n_females'] / incidents_df['n_females_pop_year_congdist']

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
# fare funzione anche per popolazione esterna e studiare distribuzioni
# fare gli stessi indici con child, teen, adult
# considerare i tag
# usare popolazione esterna per fare statistiche aggregate (indici a livello di stato, distretto congressuale - anno - semestre, non per record)
# aggregazione più fine-grossa sul tempo (non credo sia necessario)
# comninare genere e fasce di età (?)
# usare dati di città e contee da census?


