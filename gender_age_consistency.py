# %% [markdown]
# ### Age

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'
poverty_path = FOLDER + 'poverty_by_state_year.csv'
congressional_winner_path = FOLDER + 'year_state_district_house.csv'

# %%
# Load data from csv files
incidents_data = pd.read_csv(incidents_path)
poverty_data = pd.read_csv(poverty_path)
congressional_winner_data = pd.read_csv(congressional_winner_path)

# %%
# plot age
incidents_data[['participant_age1', 'participant_age_group1', 'participant_gender1', 'min_age_participants', 'avg_age_participants', 'max_age_participants', 'n_participants_child', 'n_participants_teen', 'n_participants_adult', 'n_males', 'n_females', 'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 'n_participants']].head(10)

# %%
# check age and age group

wrong = 0
for index, row in incidents_data.iterrows():
    if not np.isnan(row['participant_age1']):
        
        # check for unconsistent age group value and correct it
        if (row['participant_age1'] >= 18.0 and row['participant_age_group1'] != 'Adult 18+'):
            wrong += 1
            incidents_data.loc[index, 'participant_age_group1'] = 'Adult 18+'
            print(str(row['date']) + "\t" + str(row['participant_age1']) + "\t" + str(row['participant_age_group1']))
        elif (row['participant_age1'] >= 12.0 and row['participant_age1'] <= 17.0 and row['participant_age_group1'] != 'Teen 12-17'):
            wrong += 1
            incidents_data.loc[index, 'participant_age_group1'] = 'Teen 12-17'
            print(str(row['date']) + "\t" + str(row['participant_age1']) + "\t" + str(row['participant_age_group1']))
        elif (row['participant_age1'] <= 11.0 and row['participant_age_group1'] != 'Child 0-11') :
            wrong += 1
            incidents_data.loc[index, 'participant_age_group1'] = 'Child 0-11'
            print(str(row['date']) + "\t" + str(row['participant_age1']) + "\t" + str(row['participant_age_group1']))

print(wrong) #1295 initially
incidents_data['participant_age_group1'].unique()

# %%
# just to show the result :)

wrong = 0
for index, row in incidents_data.iterrows():
    if not np.isnan(row['participant_age1']):
        
        if (row['participant_age1'] >= 18.0 and row['participant_age_group1'] != 'Adult 18+'):
            wrong += 1
            incidents_data.loc[index, 'participant_age_group1'] = 'Adult 18+'
            print(str(row['date']) + "\t" + str(row['participant_age1']) + "\t" + str(row['participant_age_group1']))
        elif (row['participant_age1'] >= 12.0 and row['participant_age1'] <= 17.0 and row['participant_age_group1'] != 'Teen 12-17'):
            wrong += 1
            incidents_data.loc[index, 'participant_age_group1'] = 'Teen 12-17'
            print(str(row['date']) + "\t" + str(row['participant_age1']) + "\t" + str(row['participant_age_group1']))
        elif (row['participant_age1'] <= 11.0 and row['participant_age_group1'] != 'Child 0-11') :
            wrong += 1
            incidents_data.loc[index, 'participant_age_group1'] = 'Child 0-11'
            print(str(row['date']) + "\t" + str(row['participant_age1']) + "\t" + str(row['participant_age_group1']))

print(wrong)


