# -*- coding: utf-8 -*-
# %% [markdown]
# # Definition and study of the features to use for the classification task

# %% [markdown]
# We import the libraries:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
import json
sys.path.append(os.path.abspath('..'))
from plot_utils import *
# %matplotlib inline
from classification_utils import *
from enum import Enum
import pyproj
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# %% [markdown]
# We load the dataset, reaname some columns and drop incidents tags previously computed:

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=0,
    parse_dates=['date', 'date_original'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)
incidents_df.drop_duplicates(inplace=True)
incidents_df.rename(
    columns={
        'min_age_participants': 'min_age',
        'avg_age_participants': 'avg_age',
        'max_age_participants': 'max_age',
        'n_participants_child': 'n_child',
        'n_participants_teen': 'n_teen',
        'n_participants_adult': 'n_adult',
        'location_importance': 'location_imp',
        'party': 'democrat'
    },
    inplace=True
)
old_tags = [
    'firearm', 'air_gun', 'shots', 'aggression', 'suicide', 'injuries',
    'death', 'road', 'illegal_holding', 'house', 'school', 'children',
    'drugs', 'officers', 'organized', 'social_reasons', 'defensive',
    'workplace', 'abduction', 'unintentional'
]
incidents_df.drop(columns=old_tags, inplace=True)
dataset_original_columns = incidents_df.columns

# %% [markdown]
# We project latitude and longitude on the cartesian plane:

# %%
projector = pyproj.Proj(proj='utm', zone=14, ellps='WGS84', preserve_units=True) # UTM zone 14, US centered
incidents_df['x'], incidents_df['y'] = projector(incidents_df['longitude'], incidents_df['latitude'])

# %% [markdown]
# We plot the projection to check if it is correct:

# %%
plt.plot(incidents_df['x'], incidents_df['y'], 'o', markersize=1)
plt.axis('equal')

# %% [markdown]
# We define a function to compute ratio indicators:

# %%
def compute_record_level_ratio_indicator(df, num, den):
    df[num+'_'+den+'_ratio'] = df[num] / df[den]
    return df

# %% [markdown]
# We use the function defined above to compute the ratio between the cardinality of a subset of participants and the total number of participants involved in the incident and we visualize the distributions of the computed indicators:

# %%
incident_ratio_num_columns = ['n_males', 'n_females', 'n_adult', 'n_teen', 'n_child']
for feature in incident_ratio_num_columns:
    incidents_df = compute_record_level_ratio_indicator(df=incidents_df, num=feature, den='n_participants')
# store the names of the new features
record_level_ratios = []
for feature in incidents_df.columns:
    if 'ratio' in feature:
        record_level_ratios.append(feature)
# visualize the distributions of the features
fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data=incidents_df[record_level_ratios], ax=ax)
plt.xticks(rotation=90, ha='right');

# %% [markdown]
# We compute the age range of the participants involved in the incident and we visualize its distribution:

# %%
incidents_df['age_range'] = incidents_df['max_age'] - incidents_df['min_age']
sns.violinplot(data=incidents_df[['age_range']])

# %% [markdown]
# We compute binary tags describing the incident characteristics to use as indicators for the classification task:

# %%
class TagForClassification(Enum):
    aggression = 1
    accidental = 2
    defensive = 3
    suicide = 4

    road = 5
    house = 6
    school = 7
    business = 8

    illegal_holding = 9
    drug_alcohol = 10
    officers = 11

    organized = 12
    social_reasons = 13 
    abduction = 14

tags_map = {
    'ATF/LE Confiscation/Raid/Arrest': [TagForClassification.illegal_holding.name, TagForClassification.officers.name],
    'Accidental Shooting': [TagForClassification.accidental.name],
    'Accidental Shooting - Death': [TagForClassification.accidental.name],
    'Accidental Shooting - Injury': [TagForClassification.accidental.name],
    'Accidental Shooting at a Business': [TagForClassification.business.name, TagForClassification.accidental.name],
    'Accidental/Negligent Discharge': [TagForClassification.accidental.name],
    'Animal shot/killed': [],
    'Armed robbery with injury/death and/or evidence of DGU found': [TagForClassification.aggression.name, TagForClassification.business.name],
    'Assault weapon (AR-15, AK-47, and ALL variants defined by law enforcement)': [],
    'Attempted Murder/Suicide (one variable unsuccessful)': [TagForClassification.aggression.name, TagForClassification.suicide.name],
    'BB/Pellet/Replica gun': [],
    'Bar/club incident - in or around establishment': [TagForClassification.drug_alcohol.name, TagForClassification.business.name],
    'Brandishing/flourishing/open carry/lost/found': [TagForClassification.illegal_holding.name],
    'Car-jacking': [TagForClassification.aggression.name, TagForClassification.road.name],
    'Child Involved Incident': [],
    'Child picked up & fired gun': [],
    'Child with gun - no shots fired': [],
    'Cleaning gun': [TagForClassification.accidental.name],
    'Concealed Carry License - Perpetrator': [],
    'Concealed Carry License - Victim': [],
    'Criminal act with stolen gun': [TagForClassification.illegal_holding.name, TagForClassification.aggression.name],
    'Defensive Use': [TagForClassification.defensive.name],
    'Defensive Use - Crime occurs, victim shoots subject/suspect/perpetrator': [TagForClassification.defensive.name],
    'Defensive Use - Shots fired, no injury/death': [TagForClassification.defensive.name],
    'Defensive Use - Victim stops crime': [TagForClassification.defensive.name],
    'Defensive Use - WITHOUT a gun': [TagForClassification.defensive.name],
    'Domestic Violence': [TagForClassification.house.name, TagForClassification.aggression.name],
    'Drive-by (car to street, car to car)': [TagForClassification.aggression.name, TagForClassification.road.name],
    'Drug involvement': [TagForClassification.drug_alcohol.name],
    'Gang involvement': [TagForClassification.organized.name],
    'Ghost gun': [],
    'Gun at school, no death/injury - elementary/secondary school': [TagForClassification.school.name, TagForClassification.aggression.name],
    'Gun at school, no death/injury - university/college': [TagForClassification.school.name, TagForClassification.aggression.name],
    'Gun buy back action': [],
    'Gun range/gun shop/gun show shooting': [],
    'Gun shop robbery or burglary': [TagForClassification.illegal_holding.name, TagForClassification.business.name],
    'Gun(s) stolen from owner': [TagForClassification.illegal_holding.name],
    'Guns stolen from law enforcement': [TagForClassification.illegal_holding.name, TagForClassification.officers.name],
    'Hate crime': [TagForClassification.social_reasons.name, TagForClassification.aggression.name],
    'Home Invasion': [TagForClassification.house.name, TagForClassification.aggression.name],
    'Home Invasion - No death or injury': [TagForClassification.house.name, TagForClassification.aggression.name],
    'Home Invasion - Resident injured': [TagForClassification.aggression.name, TagForClassification.house.name],
    'Home Invasion - Resident killed': [TagForClassification.aggression.name, TagForClassification.house.name],
    'Home Invasion - subject/suspect/perpetrator injured': [TagForClassification.house.name, TagForClassification.defensive.name],
    'Home Invasion - subject/suspect/perpetrator killed': [TagForClassification.house.name, TagForClassification.defensive.name],
    'House party': [TagForClassification.house.name, TagForClassification.drug_alcohol.name],
    'Hunting accident': [TagForClassification.accidental.name],
    'Implied Weapon': [],
    'Institution/Group/Business': [TagForClassification.business.name],
    'Kidnapping/abductions/hostage': [TagForClassification.aggression.name, TagForClassification.abduction.name],
    'LOCKDOWN/ALERT ONLY: No GV Incident Occurred Onsite': [],
    'Mass Murder (4+ deceased victims excluding the subject/suspect/perpetrator , one location)': [TagForClassification.aggression.name],
    'Mass Shooting (4+ victims injured or killed excluding the subject/suspect/perpetrator, one location)': [TagForClassification.aggression.name],
    'Murder/Suicide': [TagForClassification.aggression.name, TagForClassification.suicide.name],
    'Non-Aggression Incident': [],
    'Non-Shooting Incident': [],
    'Officer Involved Incident': [TagForClassification.officers.name],
    'Officer Involved Incident - Weapon involved but no shots fired': [TagForClassification.officers.name],
    'Officer Involved Shooting - Accidental discharge - no injury required': [TagForClassification.officers.name, TagForClassification.accidental.name],
    'Officer Involved Shooting - Officer killed': [TagForClassification.aggression.name, TagForClassification.officers.name],
    'Officer Involved Shooting - Officer shot': [TagForClassification.officers.name, TagForClassification.aggression.name],
    'Officer Involved Shooting - Shots fired, no injury': [TagForClassification.officers.name],
    'Officer Involved Shooting - subject/suspect/perpetrator killed': [TagForClassification.officers.name, TagForClassification.defensive.name],
    'Officer Involved Shooting - subject/suspect/perpetrator shot': [TagForClassification.officers.name, TagForClassification.defensive.name],
    'Officer Involved Shooting - subject/suspect/perpetrator suicide at standoff': [TagForClassification.officers.name, TagForClassification.suicide.name],
    'Officer Involved Shooting - subject/suspect/perpetrator surrender at standoff': [TagForClassification.officers.name],
    'Officer Involved Shooting - subject/suspect/perpetrator unarmed': [TagForClassification.officers.name],
    'Pistol-whipping': [TagForClassification.aggression.name],
    'Police Targeted': [TagForClassification.officers.name, TagForClassification.aggression.name],
    'Political Violence': [TagForClassification.aggression.name, TagForClassification.social_reasons.name],
    'Possession (gun(s) found during commission of other crimes)': [],
    'Possession of gun by felon or prohibited person': [TagForClassification.illegal_holding.name],
    'Road rage': [TagForClassification.road.name, TagForClassification.aggression.name],
    'School Incident': [TagForClassification.school.name, TagForClassification.aggression.name],
    'School Shooting - elementary/secondary school': [TagForClassification.aggression.name, TagForClassification.school.name],
    'Sex crime involving firearm': [TagForClassification.aggression.name],
    'Shootout (where VENN diagram of shooters and victims overlap)': [TagForClassification.aggression.name],
    'Shot - Dead (murder, accidental, suicide)': [],
    'Shot - Wounded/Injured': [],
    'ShotSpotter': [],
    'Shots Fired - No Injuries': [],
    'Shots fired, no action (reported, no evidence found)': [],
    'Spree Shooting (multiple victims, multiple locations)': [TagForClassification.aggression.name],
    'Stolen/Illegally owned gun{s} recovered during arrest/warrant': [TagForClassification.illegal_holding.name],
    'Suicide - Attempt': [TagForClassification.suicide.name],
    'Suicide^': [TagForClassification.suicide.name],
    'TSA Action': [TagForClassification.officers.name],
    'Terrorism Involvement': [TagForClassification.aggression.name, TagForClassification.organized.name],
    'Under the influence of alcohol or drugs (only applies to the subject/suspect/perpetrator )': [TagForClassification.drug_alcohol.name],
    'Unlawful purchase/sale': [TagForClassification.illegal_holding.name],
    'Workplace shooting (disgruntled employee)': [TagForClassification.aggression.name, TagForClassification.business.name]
}

def set_tags(row):
    if pd.notnull(row['incident_characteristics1']):
        for tag in tags_map[row['incident_characteristics1']]:
            row[tag] = 1
    if pd.notnull(row['incident_characteristics2']):
        for tag in tags_map[row['incident_characteristics2']]:
            row[tag] = 1
    return row

def add_tags(df):
    for tag in TagForClassification:
        df[tag.name] = 0
    df = df.apply(set_tags, axis=1)
    return df

incidents_df = add_tags(incidents_df)

# %% [markdown]
# We search for inconsistencies:

# %%
incidents_df[(incidents_df['aggression']==1) & (incidents_df['defensive']==1)] # such incidents involve a defensive act, in response to an aggression

# %%
incidents_df[(incidents_df['aggression']==1) & (incidents_df['accidental']==1)]

# %%
incidents_df[
    (incidents_df['aggression']==0) & 
    ((incidents_df['organized']==1))
]

# %%
incidents_df[
    (incidents_df['aggression']==0) & 
    ((incidents_df['social_reasons']==1))
]

# %%
incidents_df[
    (incidents_df['aggression']==0) & 
    ((incidents_df['abduction']==1))
]

# %%
incidents_df[
    (incidents_df['accidental']==1) & 
    (
        (incidents_df['aggression']==1) |
        (incidents_df['defensive']==1) |
        (incidents_df['suicide']==1) |
        (incidents_df['organized']==1) |
        (incidents_df['social_reasons']==1) |
        (incidents_df['abduction']==1)
    )
]

# %%
incidents_df[
    (incidents_df['defensive']==1) & 
    (incidents_df['accidental']==1)
]

# %%
incidents_df[
    (incidents_df['defensive']==1) & 
    (incidents_df['suicide']==1)
]

# %%
incidents_df[
    (incidents_df['suicide']==1) & 
    (incidents_df['accidental']==1)
]

# %%
incidents_df[
    (incidents_df['organized']==1) & 
    (incidents_df['accidental']==1)
]

# %%
incidents_df[
    (incidents_df['social_reasons']==1) & 
    (incidents_df['accidental']==1)
]

# %%
incidents_df[
    (incidents_df['abduction']==1) & 
    (incidents_df['accidental']==1)
]

# %%
incidents_df[(incidents_df['aggression']==0) & (incidents_df['accidental']==0) & (incidents_df['defensive']==0) & (incidents_df['suicide']==0)]

# %% [markdown]
# We rename some indicators:

# %%
indicators_abbr = {
    'n_child_n_participants_ratio': 'n_child_prop',
    'n_teen_n_participants_ratio': 'n_teen_prop',
    'n_males_n_participants_ratio': 'n_males_prop',
}
incidents_df.rename(columns=indicators_abbr, inplace=True)

# %% [markdown]
# We convert to 0 or 1 the attribute indicating if the incident happened in a state with a democratic or republican governor:

# %%
incidents_df['democrat'].replace(['REPUBLICAN', 'DEMOCRAT'], [0, 1], inplace=True)

# %% [markdown]
# We convert to categorical codes the state attributes:

# %%
incidents_df['state_code'] = incidents_df['state'].astype('category').cat.codes

# %% [markdown]
# We convert months and days of week to their names:

# %%
incidents_df['month_name'] = incidents_df['month'].apply(lambda x: pd.to_datetime(x, format='%m').month_name())
incidents_df['day_of_week_name'] = incidents_df['date_original'].dt.day_name()

# %% [markdown]
# We compute the number of dayes from the first incident:

# %%
incidents_df['days_from_first_incident'] = (incidents_df['date_original'] - incidents_df['date_original'].min()).dt.days

# %% [markdown]
# We perform a cyclic encoding of the month, the day and the day of the week:

# %%
incidents_df['month_x'] = np.sin(2 * np.pi * incidents_df['month'] / 12.0)
incidents_df['month_y'] = np.cos(2 * np.pi * incidents_df['month'] / 12.0)

incidents_df['day_x'] = np.sin(2 * np.pi * incidents_df['day'] / 31.0)
incidents_df['day_y'] = np.cos(2 * np.pi * incidents_df['day'] / 31.0)

incidents_df['day_of_week_x'] = np.sin(2 * np.pi * incidents_df['day_of_week'] / 7.0)
incidents_df['day_of_week_y'] = np.cos(2 * np.pi * incidents_df['day_of_week'] / 7.0)

# %% [markdown]
# We visualize the distributions of the encoded attributes:

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.set_palette('tab20')
months = np.arange(1, 13)
months_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_xs = np.sin(2 * np.pi * np.array(months) / 12.0)
month_ys = np.cos(2 * np.pi * np.array(months) / 12.0)
for month_name, month_x, month_y in zip(months_names, month_xs, month_ys):
    axs[0].scatter(
        x=month_x,
        y=month_y,
        label=month_name
    )
axs[0].legend()
axs[0].set_title('Cyclic encoding of months')

month_days = np.arange(1, 32)
month_days_xs = np.sin(2 * np.pi * np.array(month_days) / 31.0)
month_days_ys = np.cos(2 * np.pi * np.array(month_days) / 31.0)
for month_day, month_day_x, month_day_y in zip(month_days, month_days_xs, month_days_ys):
    axs[1].scatter(
        x=month_day_x,
        y=month_day_y,
        label=month_day
    )
axs[1].set_title('Cyclic encoding of days of the month')
axs[1].legend(loc='lower center', ncols=31, bbox_to_anchor=(0.5, -0.2))

sns.set_palette('tab10')
days = np.arange(0, 7)
days_names = ['Mon', 'Tue', 'Thu', 'Wed', 'Fri', 'Sat', 'Sun']
days_xs = np.sin(2 * np.pi * np.array(days) / 7.0)
days_ys = np.cos(2 * np.pi * np.array(days) / 7.0)
for day, day_x, day_y in zip(days_names, days_xs, days_ys):
    axs[2].scatter(
        x=day_x,
        y=day_y,
        label=day
    )
axs[2].set_title('Cyclic encoding of days of the week')
axs[2].legend()

# %% [markdown]
# We load the data about the gun laws strictness:

# %%
laws_df = pd.read_csv('../data/external_data/gun_law_rank.csv')
laws_df

# %% [markdown]
# We merge the datasets:

# %%
incidents_df = incidents_df.merge(laws_df, how='left', on=['state', 'year'])

# %% [markdown]
# We define the list of indicators to use for the classification task:

# %%
indicators_names = [
    # spatial data
    'location_imp',
    'latitude',
    'longitude',
    'x',
    'y',
    'state_code',
    'congressional_district',
    # age data
    'age_range',
    'avg_age',
    'n_child_prop',
    'n_teen_prop',
    # gender data
    'n_males_prop',
    # characteristics data
    'n_participants',
    # temporal data
    'day',
    'day_x',
    'day_y',
    'day_of_week',
    'day_of_week_x',
    'day_of_week_y',
    'month',
    'month_x',
    'month_y',
    'year',
    'days_from_first_incident',
    # socio-economic data
    'poverty_perc',
    'democrat',
    'gun_law_rank'
]

# add the tags
for tag in TagForClassification._member_names_:
    indicators_names.append(tag)

# %% [markdown]
# We define the binary label to predict:

# %%
incidents_df['death'] = (incidents_df['n_killed']>0).astype(int)

# %% [markdown]
# We compute the correlation between the indicators (and the label to predict):

# %%
# compute the pearson's correlation coefficient
fig, ax = plt.subplots(figsize=(40, 15))
pearson_corr_matrix = incidents_df[indicators_names + ['death']].corr('pearson')
sns.heatmap(pearson_corr_matrix, annot=True, ax=ax, mask=np.triu(pearson_corr_matrix), cmap='coolwarm')

# %%
# compute the spearman's correlation coefficient
fig, ax = plt.subplots(figsize=(40, 15))
spearman_corr_matrix = incidents_df[indicators_names + ['death']].corr('spearman')
sns.heatmap(spearman_corr_matrix, annot=True, ax=ax, mask=np.triu(spearman_corr_matrix), cmap='coolwarm')

# %% [markdown]
# We observe a correlation between:
# - poverty percentage, gun_law_rank and latitude (southern states are more poor and have less strict gun laws)
# - road, house and aggression (when the tag aggression could be inferred also the place where the incident happened was known)

# %% [markdown]
# We check for duplicated rows:

# %%
n_duplicates = incidents_df[indicators_names].duplicated().sum()
print(f"Number of duplicated rows: {n_duplicates}")
print(f"Percentage of duplicated rows: {(n_duplicates/incidents_df[indicators_names].shape[0])*100:.2f}%")

# %% [markdown]
# We visualize the number of nan values for each indicator:

# %%
incidents_df[indicators_names].info()

# %%
print(f'The dataset has {incidents_df.shape[0]} rows')
print(f'Dropping rows with nan values in the indicators columns, {incidents_df[indicators_names].dropna().shape[0]} rows remain')

# %% [markdown]
# We display a summary of the descriptive statistics of the indicators:

# %%
incidents_df[indicators_names].describe()

# %% [markdown]
# We drop incidents having at least a nan indicator:

# %%
incidents_clf = incidents_df.dropna(subset=indicators_names).copy()
incidents_clf.drop_duplicates(subset=indicators_names, inplace=True) # same x, y, data, characteristics, etc. cannot be different incidents
incidents_nan = incidents_df[incidents_df[indicators_names].isna().any(axis=1)]

# %% [markdown]
# We save all the names of the indicators in a json file:

# %%
with open('../data/clf_indicators_names.json', 'w') as f:
    json.dump(indicators_names, f)

# %% [markdown]
# We visualize the distribution of mortal incidents:

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
incidents_clf['death'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], title='Death distribution (indicidents without nan indicators)')
incidents_df['death'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1], title='Death distribution (all incidents)')

# %% [markdown]
# We visualize the distribution of the incidents tag in the whole dataset and in the subset of mortal incidents:

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

plot1 = (incidents_clf[TagForClassification._member_names_].apply(lambda col: col.value_counts()).T.sort_values(by=1)/incidents_clf.shape[0]*100).plot(kind='barh', stacked=True, alpha=0.8, edgecolor='black', ax=axs[0])
for container in plot1.containers:
    plot1.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=8)
axs[0].set_title("Incidents characteristic distribution (both mortal and not)")

plot2 = (incidents_clf[incidents_clf['death']==1][TagForClassification._member_names_].apply(lambda col: col.value_counts()).T.sort_values(by=1)/incidents_clf[incidents_clf['death']==1].shape[0]*100).plot(kind='barh', stacked=True, alpha=0.8, edgecolor='black', ax=axs[1])
for container in plot2.containers:
    plot2.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=8)
axs[1].set_title("Mortal Incidents characteristic distribution")

# %% [markdown]
# We check how many mortal incidents happened in a school:

# %%
incidents_clf[(incidents_clf['death']==1) & (incidents_clf['school']==1)].shape[0]

# %% [markdown]
# We check the distribution of mortal incidents among the incident with unknown type:

# %%
incidents_clf[
    (incidents_clf['aggression']==0) &
    (incidents_clf['accidental']==0) &
    (incidents_clf['defensive']==0) &
    (incidents_clf['suicide']==0)
]['death'].value_counts().plot.pie(autopct='%1.1f%%', title='Death distribution (indicidents with unknown type)')

# %%
(   incidents_clf[(incidents_clf['suicide']==1) & (incidents_clf['death']==0)].shape[0] / \
    incidents_clf[(incidents_clf['suicide']==1) & (incidents_clf['death']==1)].shape[0]
)*100

# %% [markdown]
# Visualize incidents in PCA space:

# %%
pca = PCA()
std_scaler = MinMaxScaler()
numeric_indicators = ['location_imp', 'x', 'y', 'age_range', 'avg_age', 'n_child_prop', 'n_teen_prop', 'n_males_prop', 'n_participants', 'poverty_perc']
X_minmax = std_scaler.fit_transform(incidents_clf[numeric_indicators].values)
X_pca = pca.fit_transform(X_minmax)
scatter_pca_features_by_label(
    X_pca,
    n_components=4,
    labels=incidents_clf['death'].values,
    palette=sns.color_palette(n_colors=2)
)

# %% [markdown]
# Visualize distribution of the indicators in the first and second principal components space:

# %%
nplots = len(numeric_indicators)
ncols = 4
nrows = int(nplots/ncols)
if nplots % ncols != 0:
    nrows += 1
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 25), sharex=True, sharey=True)
for i, col in enumerate(numeric_indicators):
    axs[int(i/ncols)][i%ncols].scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40, c=incidents_clf[col], cmap='viridis')
    axs[int(i/ncols)][i%ncols].set_title(col)
    axs[int(i/ncols)][i%ncols].set_xlabel("1st eigenvector")
    axs[int(i/ncols)][i%ncols].set_ylabel("2nd eigenvector")
if nrows > 1:
    for ax in axs[nrows-1, i%ncols:]:
        ax.remove()

# %% [markdown]
# The first two principal components are correlated with the age and gender attributes.

# %% [markdown]
# Visualize distribution of the indicators in the first and third principal components space:

# %%
nplots = len(numeric_indicators)
ncols = 4
nrows = int(nplots/ncols)
if nplots % ncols != 0:
    nrows += 1
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 25), sharex=True, sharey=True)
for i, col in enumerate(numeric_indicators):
    axs[int(i/ncols)][i%ncols].scatter(X_pca[:, 0], X_pca[:, 2], edgecolor='k', s=40, c=incidents_clf[col], cmap='viridis')
    axs[int(i/ncols)][i%ncols].set_title(col)
    axs[int(i/ncols)][i%ncols].set_xlabel("1st eigenvector")
    axs[int(i/ncols)][i%ncols].set_ylabel("3rd eigenvector")
if nrows > 1:
    for ax in axs[nrows-1, i%ncols:]:
        ax.remove()

# %% [markdown]
# Visualize distribution of the indicators in the second and third principal components space:

# %%
nplots = len(numeric_indicators)
ncols = 4
nrows = int(nplots/ncols)
if nplots % ncols != 0:
    nrows += 1
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 25), sharex=True, sharey=True)
for i, col in enumerate(numeric_indicators):
    axs[int(i/ncols)][i%ncols].scatter(X_pca[:, 1], X_pca[:, 2], edgecolor='k', s=40, c=incidents_clf[col], cmap='viridis')
    axs[int(i/ncols)][i%ncols].set_title(col)
    axs[int(i/ncols)][i%ncols].set_xlabel("2nd eigenvector")
    axs[int(i/ncols)][i%ncols].set_ylabel("3rd eigenvector")
if nrows > 1:
    for ax in axs[nrows-1, i%ncols:]:
        ax.remove()

# %% [markdown]
# The third principal components does not seem to be correlated with any indicator.

# %% [markdown]
# We split the dataset in a train set (including 0.75% of the records) and a test test (including the remainig 0.25% of the records), stratifiying the split according to the label to predict, and we save the sets in csv files:

# %%
original_features_minus_indicators = [feature for feature in dataset_original_columns if feature not in indicators_names]
incidents_clf = incidents_clf[original_features_minus_indicators + indicators_names + ['death']]
incidents_nan = incidents_nan[original_features_minus_indicators + indicators_names + ['death']]
X_train_df, X_test_df, y_train, y_test = train_test_split(
    incidents_clf.drop(columns='death'),
    incidents_clf['death'],
    test_size=1/3,
    random_state=42,
    shuffle=True,
    stratify=incidents_clf['death'].values
)
pd.concat([X_train_df, incidents_nan.drop(columns='death')]).to_csv('../data/clf_indicators_train_nan.csv')
pd.concat([y_train, incidents_nan['death']]).to_csv('../data/clf_y_train_nan.csv')
X_train_df.to_csv('../data/clf_indicators_train.csv')
X_test_df.to_csv('../data/clf_indicators_test.csv')
y_train.to_csv('../data/clf_y_train.csv')
y_test.to_csv('../data/clf_y_test.csv')

# %% [markdown]
# We display the distributions of the indicators:

# %%
# apply MinMaxScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(incidents_clf[indicators_names])
# apply StandardScaler
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(incidents_clf[indicators_names])
# apply RobustScaler
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(incidents_clf[indicators_names])
# plot the distributions of the indicators after the transformations
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(40, 6))
axs[0].boxplot(incidents_clf[indicators_names])
axs[0].set_xticklabels(indicators_names, rotation=90);
axs[0].set_title('Original data');
axs[1].boxplot(X_minmax)
axs[1].set_xticklabels(indicators_names, rotation=90);
axs[1].set_title('Min-Max scaling');
axs[2].boxplot(X_std)
axs[2].set_xticklabels(indicators_names, rotation=90);
axs[2].set_title('Standard scaling');
axs[3].boxplot(X_robust)
axs[3].set_xticklabels(indicators_names, rotation=90);
axs[3].set_title('Robust scaling');
fig.suptitle('Distributions of the indicators', fontweight='bold');

# %% [markdown]
# We apply Min-Max scaling:

# %%
minmax_scaler.fit(X_train_df[indicators_names])
X_test_transf = minmax_scaler.transform(X_test_df[indicators_names])
X_test_df[indicators_names] = X_test_transf
X_test_df.to_csv('../data/clf_scaled_indicators_test.csv')

# %%
fig, axs = plt.subplots(figsize=(10, 6))
X_test_df[indicators_names].boxplot(ax=axs, rot=90)

# %% [markdown]
# We display train and test sizes:

# %%
train_test_infos = {}
train_test_infos['Fatal'] = [y_train.sum(), y_test.sum()]
train_test_infos['Fatal (%)'] = [y_train.sum()/y_train.shape[0]*100, y_test.sum()/y_test.shape[0]*100]
train_test_infos['Non-Fatal'] = [(y_train == False).sum(), (y_test == False).sum()]
train_test_infos['Non-Fatal (%)'] = [(y_train == False).sum()/y_train.shape[0]*100, (y_test == False).sum()/y_test.shape[0]*100]
train_test_infos['total'] = [X_train_df.shape[0], X_test_df.shape[0]]
pd.DataFrame(train_test_infos, index=['train', 'test'])

# %% [markdown]
# We write to json files the names of the features to use with distance or rule based classifiers:

# %%
# distance based features
features_db = [
    'location_imp', 'x', 'y',
    'age_range', 'avg_age', 'n_child_prop', 'n_teen_prop', 
    'n_males_prop',
    'n_participants',
    'day_x', 'day_y', 'day_of_week_x', 'day_of_week_y', 'month_x', 'month_y', 'year', 'days_from_first_incident',
    'poverty_perc', 'democrat', 'gun_law_rank', 
    'aggression', 'accidental', 'defensive', 'suicide', 'road', 'house', 'school', 'business', 'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]

with open('../data/clf_indicators_names_distance_based.json', 'w') as f:
    json.dump(features_db, f)

# rule based features
features_rb = [
    'location_imp', 'x', 'y',
    'age_range', 'avg_age', 'n_child_prop', 'n_teen_prop',
    'n_males_prop',
    'n_participants',
    'day', 'day_of_week', 'month', 'year', 'days_from_first_incident',
    'poverty_perc', 'democrat', 'gun_law_rank',
    'aggression', 'accidental', 'defensive', 'suicide', 'road', 'house', 'school', 'business', 'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]

categorical_features_rb = [
    'day', 'day_of_week', 'month', 'year',
    'democrat', 'gun_law_rank',
    'aggression', 'accidental', 'defensive', 'suicide',
    'road', 'house', 'school', 'business',
    'illegal_holding', 'drug_alcohol', 'officers', 'organized', 'social_reasons', 'abduction'
]

with open('../data/clf_indicators_names_rule_based.json', 'w') as f:
    json.dump(features_rb, f)

# %% [markdown]
# We perform Random over-sampling:

# %%
oversample = RandomOverSampler(sampling_strategy=0.4/0.6)
X_train_over_df, y_train_over = oversample.fit_resample(X_train_df, y_train)

X_train_over_df.to_csv('../data/clf_indicators_train_over.csv')
y_train_over.to_csv('../data/clf_y_train_over.csv')

train_over_infos = {}
train_over_infos['Fatal'] = [y_train_over.sum()]
train_over_infos['Fatal (%)'] = [(y_train_over.sum()/y_train_over.shape[0])*100]
train_over_infos['Non_Fatal'] = [(y_train_over == 0).sum()]
train_over_infos['Non_Fatal (%)'] = [((y_train_over == 0).sum()/y_train_over.shape[0])*100]
train_over_infos['total'] = [y_train_over.shape[0]]
pd.DataFrame(train_over_infos, index=['train'])

# %% [markdown]
# We perform oversampling using SMOTE, taking into account categorical attributes:

# %%
smote_oversample = SMOTENC(categorical_features=categorical_features_rb, sampling_strategy=0.4/0.6)
X_train_smote_df, y_train_smote = smote_oversample.fit_resample(X_train_df[features_rb], y_train)
# recompute splitted indicators
X_train_smote_df['month_x'] = np.sin(2 * np.pi * X_train_smote_df['month'] / 12.0)
X_train_smote_df['month_y'] = np.cos(2 * np.pi * X_train_smote_df['month'] / 12.0)
X_train_smote_df['day_x'] = np.sin(2 * np.pi * X_train_smote_df['day'] / 31.0)
X_train_smote_df['day_y'] = np.cos(2 * np.pi * X_train_smote_df['day'] / 31.0)
X_train_smote_df['day_of_week_x'] = np.sin(2 * np.pi * X_train_smote_df['day_of_week'] / 7.0)
X_train_smote_df['day_of_week_y'] = np.cos(2 * np.pi * X_train_smote_df['day_of_week'] / 7.0)

X_train_smote_df.to_csv('../data/clf_indicators_train_smote.csv')
y_train_smote.to_csv('../data/clf_y_train_smote.csv')

train_smote_infos = {}
train_smote_infos['Fatal'] = [y_train_smote.sum()]
train_smote_infos['Fatal (%)'] = [(y_train_smote.sum()/y_train_smote.shape[0])*100]
train_smote_infos['Non_Fatal'] = [(y_train_smote == 0).sum()]
train_smote_infos['Non_Fatal (%)'] = [((y_train_smote == 0).sum()/y_train_smote.shape[0])*100]
train_smote_infos['total'] = [y_train_smote.shape[0]]
pd.DataFrame(train_smote_infos, index=['train'])

# %% [markdown]
# We define a function to compute and save the most frequent values for each feature:

# %%
def save_default_feature_values(df, path):
    # compute the most frequent value for each feature
    default = df.agg(lambda x: x.value_counts().index[0])
    if 'day_of_week_x' in df.columns: # distance based features
        default_day_of_week_x, default_day_of_week_y = df.groupby(["day_of_week_x", "day_of_week_y"]).size().sort_values(ascending=False).index[0]
        default['day_of_week_x'] = default_day_of_week_x
        default['day_of_week_y'] = default_day_of_week_y
        default_day_x, default_day_y = df.groupby(["day_x", "day_y"]).size().sort_values(ascending=False).index[0]
        default['day_x'] = default_day_x
        default['day_y'] = default_day_y
        default_month_x, default_month_y = df.groupby(["month_x", "month_y"]).size().sort_values(ascending=False).index[0]
        default['month_x'] = default_month_x
        default['month_y'] = default_month_y
    default.to_frame().T.to_csv(path, index=False)

# %% [markdown]
# We apply that function to compute the most frequent values for each feature in fatal and non-fatal incidents:

# %%
save_default_feature_values(df=X_train_df[y_train==1][features_rb], path='../data/classification_results/fatal_rb_default_features.csv')
save_default_feature_values(df=X_train_df[y_train==0][features_rb], path='../data/classification_results/non_fatal_rb_default_features.csv')
save_default_feature_values(df=X_train_df[y_train==1][features_db], path='../data/classification_results/fatal_db_default_features.csv')
save_default_feature_values(df=X_train_df[y_train==0][features_db], path='../data/classification_results/non_fatal_db_default_features.csv')

# %% [markdown]
# TODO: compilare una volta definiti
#
# # Final Indicators semantics
# | Name | Description | Present in the original dataset |
# | :--: | :---------: | :-----------------------------: |
# | location_imp | Location importance according to Geopy | No |
# | latitude | Latitude of the incident | Yes |
# | longitude | Longitude of the incident | Yes |
# | x | Projection of the longitude of the incident | No |
# | y | Projection of the latitude of the incident | No |
# | age_range | Difference between the maximum and the minimum age of the participants involved in the incident | No |
# | avg_age | Average age of the participants involved in the incident | Yes |
# | n_child_prop | Ratio between the number of child involved in the incident and number of people involved in the incident | No |
# | n_teen_prop | Ratio between the number of teen involved in the incident and number of people involved in the incident | No |
# | n_males_prop | Ratio between the number of males and the number of people involed in the incident | No |
# | n_participants | Number of participants involved in the incident | Yes |
# | month | Month in which the incident happened | Yes (in date) |
# | day_of_week | Day of the week in which the incident happened | | No (computed from date) |
# | poverty_perc | Poverty percentage in the state and year of the incident | Yes |
# | democrat | Winning party in the congressional_district and year of the incident | Yes |
# | aggression | Whether the incident involved an aggression (both with a gun or not) | No (extracted from the incident characteristics) |
# | road | Whether the incident happened in a road | No (extracted from the incident characteristics) |
# | illegal_holding | Whether the incident involved a stealing act or a gun was illegally possessed | No (extracted from the incident characteristics) |
# | house | Whether the incident happened in a house | No (extracted from the incident characteristics) |
# | school | Whether the incident happened in a school | No (extracted from the incident characteristics) |
# | drugs | Whether the incident involved drugs | No (extracted from the incident characteristics) |
# | officers | Whether one or more officiers were involved in the incident | No (extracted from the incident characteristics) |
# | organized | Whether the action was planned by an organization or a group | No (extracted from the incident characteristics) |
# | social_reasons | Whether the incident involved social discriminations or terrorism | No (extracted from the incident characteristics) |
# | defensive | Whether the incident involved the use of a gun for defensive purposes | No (extracted from the incident characteristics) |
# | workplace | Whether the incident happened in a workplace | No (extracted from the incident characteristics) |
# | abduction | Whether the incident involved any form of abduction | No (extracted from the incident characteristics) |
# | unintentional | Whether the incident was unintentional | No (extracted from the incident characteristics) |


