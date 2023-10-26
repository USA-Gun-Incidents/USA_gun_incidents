# %%
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# We read the data and drop the duplicates rows

# %%
#TODO: quando mettiamo insieme i data set ricordiamoci di non togliere gli indici + la pulizia di questi dati 
# dovrebbe essere fatta partendo da i dati geografici puliti

# %%
FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'

incidents_data = pd.read_csv(incidents_path)
incidents_data.drop_duplicates(inplace=True)

# %%
incidents_data.head(10)

# %% [markdown]
# ## Date

# %% [markdown]
# ## Geographic data

# %% [markdown]
# ## Age, gender and number of participants data

# %% [markdown]
# Columns of the dataset are considered in order to verify the correctness and consistency of data related to age, gender, and the number of participants for each incident:
# - *participant_age1*+
# - *participant_age_group1*
# - *participant_gender1*
# - *min_age_participants*
# - *avg_age_participants*
# - *max_age_participants*
# - *n_participants_child*
# - *n_participants_teen*
# - *n_participants_adult*
# - *n_males*
# - *n_females*
# - *n_killed*
# - *n_injured*
# - *n_arrested*
# - *n_unharmed*
# - *n_participants*

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

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
data.info()

# %%
data['participant_age_group1'].unique()

# %% [markdown]
# ### Data consistency

# %% [markdown]
# We create some functions to identify and, if possible, correct missing and inconsistent data.
# Below, we provide a brief summary of all the functions used to check data consistency:

# %% [markdown]
# First of all, we convert all the values to type int if the values were consistent (i.e., values related to age and the number of participants for a particular category must be a positive number), all the values that are out of range or contain alphanumeric strings were set to *NaN*.

# %% [markdown]
# Checks done to evaluate the consistency of data related to the minimum, maximum, and average ages of participants, as well as the composition of the age groups:
# 
# - min_age_participants $<$ avg_age_participants $<$ max_age_participants
# - n_participants_child $+$ n_participants_teen $+$ n_participants_adult $>$ 0
# 
# - $if$ min_age_participants $<$ 12 $then$ n_participants_child $>$ 0
# - $if$ 12 $\leq$ min_age_participants $<$ 18 $then$ n_participants_teen $>$ 0
# - $if$ min_age_participants $\geq$ 18 $then$ n_participants_adult $>$ 0
# 
# - $if$ max_age_participants $<$ 12 $then$ n_participants_child $>$ 0 and n_participants_teen $=$ 0 and n_participants_adult $=$ 0
# - $if$ max_age_participants $<$ 18 $then$ n_participants_teen $>$ 0 or n_participants_child $>$ 0 and n_participants_adult $=$ 0
# - $if$ max_age_participants $\geq$ 18 $then$ n_participants_adult $>$ 0
# 
# Note that: child = 0-11, teen = 12-17, adult = 18+

# %% [markdown]
# Checks done to evaluate the consistency of data related to number of participants divided by gender and other participants class:
# 
# - n_participants $\geq$ 0
# - n_participants $==$ n_males $+$ n_females
# - n_killed $+$ n_injured $\leq$ n_participants
# - n_arrested $\leq$ n_participants
# - n_unharmed $\leq$ n_participants

# %% [markdown]
# We also considered data of participants1, a randomly chosen participant whose data related to gender and age are reported in the dataset. For participants, we have the following features: *participant_age1*, *participant_age_group1*, *participant_gender1*.
# 
# Values related to participant_age_group1 and participant_gender1 have been binarized using one-hot encoding, thus creating the boolean features *participant1_child*, *participant1_teen*, *participant1_adult*, *participant1_male*, *participant1_female*.
# 
# The following checks are done in order to verify the consistency of the data among them and with respect to the other features of the incident:
# 
# - $if$ participant_age1 $<$ 12 $then$ participant_age_group1 $=$ *Child*
# - $if$ 12 $\leq$ participant_age1 $<$ 18 $then$ participant_age_group1 $=$ *Teen*
# - $if$ participant_age1 $\geq$ 18 $then$ participant_age_group1 $==$ *Adult*
# 
# - $if$ participant_age_group1 $==$ *Child* $then$ n_participants_child $>$ 0
# - $if$ participant_age_group1 $==$ *Teen* $then$ n_participants_teen $>$ 0
# - $if$ participant_age_group1 $==$ *Adult* $then$ n_participants_adult $>$ 0
# 
# - $if$ participant_gender1 $==$ *Male* $then$ n_males $>$ 0
# - $if$ participant_gender1 $==$ *Female* $then$ n_females $>$ 0

# %% [markdown]
# In the initial phase, only the values that were not permissible were set to *NaN*. 
# 
# We kept track of the consistency of admissible values by using variables (which could take on the boolean value *True* if they were consistent, *False* if they were not, or *NaN* in cases where data was not present). 
# 
# These variables were temporarily included in the dataframe so that we could later replace them with consistent values, if possible, or remove them if they were outside the acceptable range.
# 
# Variables:
# - *consistency_age*: Values related to the minimum, maximum, and average ages consistent with the number of participants by age groups.
# - *consistency_n_participant*: The number of participants for different categories consistent with each other.
# - *consistency_gender*: The number of participants by gender consistent with the total number of participants.
# - *consistency_participant1*: Values of features related to participant1 consistent with each other.
# 
# - *consistency_participants1_wrt_n_participants*: If *consistency_participants1_wrt_n_participants*, *participant1_age_range_consistency_wrt_all_data*, and *participant1_gender_consistency_wrt_all_data* are all *True*.
# 
# - *participant1_age_consistency_wrt_all_data*: Age of participant1 consistent with the minimum and maximum age values of the participants.
# - *participant1_age_range_consistency_wrt_all_data*: Value of the age range (*Child*, *Teen*, or *Adult*) consistent with the age groups of the participants.
# - *participant1_gender_consistency_wrt_all_data*: Gender value of participant1 consistent with the gender breakdown values of the group.
# 
# - *nan_values*: Presence of "NaN" values in the row.

# %%
from utils import check_age_gender_data_consistency
def compute_clean_data(data):
    clean_data = data.apply(lambda row: check_age_gender_data_consistency(row), axis=1)
    # save data
    clean_data.to_csv(FOLDER + 'post_proc/temporary_columns_age_gender.csv')

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

# %% [markdown]
# We display a concise summary of the DataFrame:

# %%
clean_data.info()

# %% [markdown]
# We assess the correctness of the checks performed by printing the consistency variable for the first 5 rows and providing a concise summary of their most frequent values.

# %%
clean_data[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].head(5)

# %%
clean_data[['consistency_age', 'consistency_n_participant', 'consistency_gender', 
    'consistency_participant1', 'consistency_participants1_wrt_n_participants',
    'participant1_age_consistency_wrt_all_data', 'participant1_age_range_consistency_wrt_all_data',
    'participant1_gender_consistency_wrt_all_data', 'nan_values']].describe()

# %% [markdown]
# Below, we print the number of rows with 'NaN' or inconsistent data.

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

# %% [markdown]
# We can notice that:
# - The data in our dataset related to participant1, excluding the 1295 cases where age and age group data were inconsistent with each other, always appear to be consistent with the data in the rest of the dataset and can thus be used to fill in missing or incorrect data.
# - In the data related to age and gender, some inconsistencies are present, but they account for only 2.09% and 6.75% of the total dataset rows, respectively.
# - In 93806 rows, at least one field had a *NaN* value.

# %%
data[data['participant_age1'] == 18]['participant_age_group1'].value_counts()

# %% [markdown]
# Some preliminary plots to understand how data is distributed. \
# Note that inconsistent data are still present on this date.

# %%
# distribuition of age participant1
plt.hist(clean_data['participant_age1'], bins=100)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of age participant1')
plt.show()

# %%
# distribuition min age
plt.hist(clean_data['min_age_participants'], bins=100)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of min age')
plt.show()

# %%
# distribuition max age
plt.hist(clean_data['max_age_participants'], bins=100)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of max age')
plt.show()

# %%
# distribuition number of participants
plt.hist(clean_data['n_participants'], bins=100)
plt.xlim(0, 15)
plt.xlabel('Number of participants')
plt.ylabel('Frequency')
plt.title('Distribution of number of participants (0-15 participants)')
plt.show()

# %%
plt.hist(clean_data[clean_data['consistency_gender'] == True]['n_males'], bins=60, density=False, histtype='step',
    color='blue', label='Male participants')
plt.hist(clean_data[clean_data['consistency_gender'] == True]['n_females'], bins=15, density=False, histtype='step',
    color='red', label='Female participants')
plt.xlim(0, 15)
plt.legend()
plt.xlabel('Number of participants')
plt.ylabel('Frequency')
plt.title('Distribution of number of participants (groups btw 0 and 15)')
plt.show()


# %% [markdown]
# In the table below, we can see how many data related to the 'number of participants' are clearly out of range, divided by age groups.

# %%
clean_data[clean_data['n_participants_adult'] > 63][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %% [markdown]
# Based on the tables above, we have evidence to set the maximum number of participants to 103.

# %%
clean_data[(clean_data['n_participants_child']+clean_data['n_participants_teen']+clean_data['n_participants_adult']
    )!=clean_data['n_participants']]

# %%
clean_data.loc[175445]

# %%
clean_data.iloc[236017]

# %% [markdown]
# This data visualization has been helpful in understanding the exceptions in the dataset and correcting them when possible, using other data from the same entry.
# 
# In cases where we were unable to obtain consistent data for a certain value, we have set the corresponding field to NaN.

# %% [markdown]
# ### Fix incosistente data

# %% [markdown]
# We have created a new DataFrame where we have recorded the corrected and consistent data. The functions and checks necessary for this process are documented in the 'utils.py' file. \
# Below, we have provided a brief documentation of the methods used.

# %%
# TODO: documentazione da utils

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
    new_data.to_csv(FOLDER + 'post_proc/new_columns_age_gender.csv')

def load_new_data():
    new_data = pd.read_csv(FOLDER + 'post_proc/new_columns_age_gender.csv')
    return new_data

# %%
clean_data[['n_males', 'n_females']].sum(axis=1)

# %%
# checkpoint
#compute_new_data(clean_data)
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
plt.hist(new_data['n_participants'], bins=90)
plt.xlim(0, 15)
plt.xlabel('Number of participants')
plt.ylabel('Frequency')
plt.title('Distribution of number of participants (0-15 participants)')
plt.show()

# %%
plt.hist(new_data['n_participants'], bins=90)
plt.xlabel('Number of participants')
plt.ylabel('Frequency')
plt.title('Distribution of number of participants')
plt.show()

# %%
import seaborn as sns
sns.heatmap(new_data.isnull(), cbar=False)

# %%
print('Max number of participants: ', new_data['n_participants'].max())
print('Max number of children: ', new_data['n_participants_child'].max())
print('Max number of teens: ', new_data['n_participants_teen'].max())
print('Max number of adults: ', new_data['n_participants_adult'].max())

# %%
plt.hist(new_data[new_data['n_participants_child']>0]['n_participants_child'], bins=100, density=False, histtype='step',
    color='blue', label='Children')
plt.hist(new_data[new_data['n_participants_teen']>0]['n_participants_teen'], bins=100, density=False, histtype='step',
    color='magenta', label='Teens')
plt.hist(new_data[new_data['n_participants_adult']>0]['n_participants_adult'], bins=100, density=False, histtype='step',
   color='green', label='Adults')
plt.xlabel('Number of participants')
plt.ylabel('Number of participants')
plt.title('Number of children, teens and adults for incident')
plt.legend()
plt.show()

# %%
plt.hist(new_data['avg_age_participants'], bins=100)
plt.xlim(0, 100)
plt.xlabel('Participants average age')
plt.ylabel('Frequency')
plt.title('Distribution of participants average age')
plt.show()

# %%
print('Average number of participants for incident: ', new_data['n_participants'].mean())
print('Average number of participants for incident with 1 killed: ', new_data[new_data['n_killed'] == 1]['n_participants'].mean())

# %%
print('Max number of participants for incident: ', new_data['n_participants'].max())
print('Max number of adult participants for incident: ', new_data['n_participants_adult'].max())
print('Max number of child participants for incident: ', new_data['n_participants_child'].max())
print('Max number of teen participants for incident: ', new_data['n_participants_teen'].max())

# %%
new_data[new_data['n_participants_adult'] > 63][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
# show rows with n_participants > 15
new_data[new_data['n_participants'] > 15][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %% [markdown]
# I expected to have no row for all the following query

# %%
# check if there is incidents with more than 103 participants
new_data[new_data['n_participants'] > 103][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %%
# check if there is incosistency in number of participants
new_data[new_data['n_participants'] != (new_data['n_participants_child'] + 
    new_data['n_participants_teen'] + new_data['n_participants_adult'])][['n_participants', 'n_participants_adult', 
    'n_participants_child', 'n_participants_teen']]

# %% [markdown]
# If sum is 0, all values are NaN 

# %%
new_data[new_data['n_participants'] != (new_data['n_participants_child'] + 
    new_data['n_participants_teen'] + new_data['n_participants_adult'])][['n_participants_adult', 
    'n_participants_child', 'n_participants_teen']].sum()

# %%
# check if there is incosistency in number of participants
new_data[((new_data['n_participants'] != (new_data['n_males'] + new_data['n_females'])) &
    new_data['n_males'].notna() & new_data['n_females'].notna())]['n_participants'].sum()

# %%
new_data[new_data['n_participants'] == 0]

# %%
# check if there is missing data that I can fill
new_data[(new_data['n_participants'].isna() + new_data['n_participants_child'].notna() +
    new_data['n_participants_teen'].notna() + new_data['n_participants_adult'].notna()) == 3]

# %%
# check if there is missing data that I can fill
new_data[(new_data['n_participants'].isna() + new_data['n_males'].notna() + new_data['n_females'].notna()) == 2]

# %% [markdown]
# ### Final check

# %%
print('Total rows with all values null: ', new_data.isnull().all(axis=1).sum())
print('Total rows with null value for n_participants: ', new_data['n_participants'].isnull().sum())
print('Total rows with null value for n_participants_child: ', new_data['n_participants_child'].isnull().sum())
print('Total rows with null value for n_participants_teen: ', new_data['n_participants_teen'].isnull().sum())
print('Total rows with null value for n_participants_adult: ', new_data['n_participants_adult'].isnull().sum())
print('Total rows with null value for n_males: ', new_data['n_males'].isnull().sum())
print('Total rows with null value for n_females: ', new_data['n_females'].isnull().sum())

# %% [markdown]
# ### Check with tag

# %%
FOLDER = './data/'
tag_path = FOLDER + 'post_proc/incidents_with_tags.csv'

# Load data from csv files
tag_data = pd.read_csv(tag_path)

# drop duplicates
tag_data.drop_duplicates(inplace=True)

# %% [markdown]
# TAG: Firearm, Shots, Aggression, Suicide, Injuries, Death, Road, Illegal holding, House, 
# School, Children, Drugs, Officers, Organized, Social reasons, Defensive, Workplace, Tag Consistency

# %%
# concatenate tag data to new_data
df = pd.concat([new_data, tag_data[['Firearm', 'Shots', 'Aggression', 'Suicide', 'Injuries', 'Death', 'Road', 
    'Illegal holding', 'House', 'School', 'Children', 'Drugs', 'Officers', 'Organized', 'Social reasons', 
    'Defensive', 'Workplace', 'Tag Consistency']]], axis=1)

# %%
df.head(10)

# %%
print('Number of rows in which tag are incosistent: ', df[df['Tag Consistency']==False].count())

# %%
from utils import check_consistency_tag

count = 0
for index, row in df.iterrows():
    if not check_consistency_tag(row):
        print(index)
        count += 1

print('Number of rows with incosistency btw tags and other attributes: ', count)


