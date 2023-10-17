# %%
import pandas as pd
import numpy as np

# %%
FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'

incidents_data = pd.read_csv(incidents_path)
incidents_data.drop_duplicates(inplace=True)

# %%
incidents_data.head(10)

# %% [markdown]
# Data:
# - participant_age1
# - participant_age_group1
# - participant_gender1
# - min_age_participants
# - avg_age_participants
# - max_age_participants
# - n_participants_child
# - n_participants_teen
# - n_participants_adult
# - n_males
# - n_females
# - n_killed
# - n_injured
# - n_arrested
# - n_unharmed
# - n_participants

# %%
# participant_age1,participant_age_group1,participant_gender1,min_age_participants,avg_age_participants,max_age_participants,n_participants_child,n_participants_teen,n_participants_adult,n_males,n_females,n_killed,n_injured,n_arrested,n_unharmed,n_participants
data = incidents_data[['participant_age1', 'participant_age_group1', 'participant_gender1', 
    'min_age_participants', 'avg_age_participants', 'max_age_participants',
    'n_participants_child', 'n_participants_teen', 'n_participants_adult', 
    'n_males', 'n_females',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants']]

# %%
data.head(10)

# %%
data.info()

# %%
data['participant_age_group1'].unique()

# %%
from utils import check_age_gender_data_consistency
def compute_clean_data(data):
    clean_data = data.apply(lambda row: check_age_gender_data_consistency(row), axis=1)
    # save data
    clean_data.to_csv(FOLDER + 'post_proc/temporary_columns_age_gender.csv', index=False)

def load_clean_data():
    clean_data = pd.read_csv(FOLDER + 'post_proc/temporary_columns_age_gender.csv')
    return clean_data

# %%
# checkpoint
#compute_clean_data(data)
clean_data = load_clean_data()

# %% [markdown]
# ### Some plots

# %%
clean_data.head(10)

# %%
clean_data.info()

# %%
clean_data[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']]

# %%
clean_data[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].describe()

# %%
clean_data[clean_data['consistency_participant1']==False].count().sum()

# %%
data[data['participant_age1'] == 18]['participant_age_group1'].value_counts()

# %%
print('Number of rows with null values: ', clean_data[clean_data['nan_values'] == True].shape[0])
print('Number of rows with inconsistent values in age data: ', clean_data[clean_data['consistency_age'] == False].shape[0])
print('Number of rows with inconsistent values in number of participants data: ', clean_data[clean_data[
    'consistency_n_participant'] == False].shape[0])
print('Number of rows with inconsistent values in gender data: ', clean_data[clean_data['consistency_gender'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 data: ', clean_data[clean_data[
    'consistency_participant1'] == False].shape[0])

# %%
print('Number of rows with inconsistent values for participants1: ', clean_data[clean_data[
    'consistency_participant1'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt all other data: ', clean_data[clean_data[
    'consistency_participants1_wrt_n_participants'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age data: ', clean_data[clean_data[
    'participant1_age_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt age range data: ', clean_data[clean_data[
    'participant1_age_range_consistency_wrt_all_data'] == False].shape[0])
print('Number of rows with inconsistent values in participants1 wrt gender data: ', clean_data[clean_data[
    'participant1_gender_consistency_wrt_all_data'] == False].shape[0])

# %%
print('Number of rows with null values in age data: ', clean_data[clean_data['consistency_age'].isna()].shape[0])
print('Number of rows with null values in number of participants data: ', clean_data[clean_data[
    'consistency_n_participant'].isna()].shape[0])
print('Number of rows with null values in gender data: ', clean_data[clean_data['consistency_gender'].isna()].shape[0])
print('Number of rows with null values in participants1 data: ', clean_data[clean_data[
    'consistency_participant1'].isna()].shape[0])

# %%
print('Number of rows with all null data: ', clean_data.isnull().all(axis=1).sum())

# %%
print('Range age: ', clean_data['min_age_participants'].min(), '-', clean_data['max_age_participants'].max())
print('Average number of participants: ', clean_data['n_participants'].mean())

# %%
import matplotlib.pyplot as plt

# %%
# distribuition of age participant1
plt.hist(clean_data['participant_age1'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of age participant1')

# %%
# ddistribuition min age
plt.hist(clean_data['min_age_participants'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of min age')

# %%
# distribuition max age
plt.hist(clean_data['max_age_participants'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of max age')

# %%
# distribuition number of participants
plt.hist(clean_data['n_participants'], bins=100)
plt.xlim(0, 15)
plt.xlabel('Number of participants')
plt.ylabel('Frequency')
plt.title('Distribution of number of participants')

# %%
plt.hist(clean_data[clean_data['consistency_gender'] == True]['n_males'], bins=15, density=True, histtype='bar',
    color='blue', label='Male participants')
plt.hist(clean_data[clean_data['consistency_gender'] == True]['n_females'], bins=15, density=True, histtype='bar',
    color='red', label='Female participants')
plt.xlim(0, 15)
plt.legend()
plt.xlabel('Number of participants')
plt.ylabel('Frequency')
plt.title('Distribution of number of participants')


# %% [markdown]
# ### Fix incosistente data

# %%
new_data = pd.DataFrame(columns=[
    'min_age_participants', 'avg_age_participants', 'max_age_participants', 
    'n_participants_child', 'n_participants_teen', 'n_participants_adult', 
    'n_males', 'n_females',
    'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
    'n_participants'])

# %%
from utils import  set_gender_age_consistent_data

def compute_new_data(clean_data):
    new_data = clean_data.apply(lambda row:  set_gender_age_consistent_data(row), axis=1)
    # save data
    new_data.to_csv(FOLDER + 'post_proc/new_columns_age_gender.csv', index=False)

def load_new_data():
    new_data = pd.read_csv(FOLDER + 'post_proc/new_columns_age_gender.csv')
    return new_data

# %%
# checkpoint
compute_new_data(clean_data)
new_data = load_new_data()

# %%
new_data.head(10)

# %%
new_data.info()

# %%
print('Number of rows in which all data are null: ', new_data.isnull().all(axis=1).sum())
print('Number of rows with some null data: ', new_data.isnull().any(axis=1).sum())
print('Number of rows in which number of participants is null: ', new_data[new_data['n_participants'].isnull()].shape[0])
print('Number of rows in which number of participants is 0: ', new_data[new_data['n_participants'] == 0].shape[0])
print('Number of rows in which number of participants is null and n_killed is not null: ', new_data[
    new_data['n_participants'].isnull() & new_data['n_killed'].notnull()].shape[0])


# %% [markdown]
# ### Some plots

# %%
plt.hist(new_data['n_participants'], bins=100)
plt.xlim(0, 15)
plt.xlabel('Number of participants')
plt.ylabel('Frequency')
plt.title('Distribution of number of participants')
plt.show()

# %%
import seaborn as sns
sns.heatmap(new_data.isnull(), cbar=False)

# %%
print('Average number of participants for incident: ', new_data['n_participants'].mean())
print('Average number of participants for incident with 1 killed: ', new_data[new_data['n_killed'] == 1]['n_participants'].mean())

# %%
plt.plot(new_data['n_participants'], new_data['n_participants_child'], 'o', color='blue', label='Children')
plt.plot(new_data['n_participants'], new_data['n_participants_teen'], 'o', color='red', label='Teens')
plt.plot(new_data['n_participants'], new_data['n_participants_adult'], 'o', color='green', label='Adults')
plt.xlabel('Number of participants')
plt.ylabel('Number of participants')
plt.title('Number of children, teens and adults with respect to number of participants')
plt.legend()
plt.show()

# %%
plt.hist(new_data['avg_age_participants'], bins=100)
plt.xlim(0, 100)
plt.xlabel('Participants average age')
plt.ylabel('Frequency')
plt.title('Distribution of participants average age')
plt.show()


