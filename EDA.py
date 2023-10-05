#!/usr/bin/env python
# coding: utf-8

# # Gun Incidents in the USA
# Dataset descriptions and explorative data analysis

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

FOLDER = './gun-data/data/'
incidents_path = FOLDER + 'incidents.csv'
poverty_path = FOLDER + 'povertyByStateYear.csv'
congressional_winner_path = FOLDER + 'year_state_district_house.csv'


# In[3]:


# Load data from csv files
incidents_data = pd.read_csv(incidents_path)
poverty_data = pd.read_csv(poverty_path)
congressional_winner_data = pd.read_csv(congressional_winner_path)


# ## Incidents data
# Contains information about gun incidents in the USA.

# In[4]:


incidents_data.head()


# In[5]:


incidents_data.columns


# In[6]:


print('Number of entries: ', incidents_data.shape[0])
print('Numebr of columns: ', incidents_data.shape[1])


# In[7]:


# Check for missing values
incidents_data.isnull().sum()


# In[8]:


# Check for duplicate entries
incidents_data.duplicated().sum()


# In[9]:


# Visualize how many incidents happened in each state
incidents_data['state'].value_counts().plot(kind='bar', figsize=(15, 5))
plt.title('Number of incidents in each state')
plt.xlabel('State')
plt.ylabel('Number of incidents')


# ## Poverty data
# Contains information about the poverty percentage for each USA state and year.

# In[11]:


poverty_data.head()


# In[12]:


print('Number of entries: ', poverty_data.shape[0])
print('Numebr of columns: ', poverty_data.shape[1])


# In[13]:


# Check for missing values
poverty_data.isnull().sum()


# In[14]:


# Check for duplicate entries
poverty_data.duplicated().sum()


# In[25]:


print('Range of years: ', poverty_data['year'].min(), poverty_data['year'].max())
print('Number of states: ', poverty_data['state'].nunique())


# In[34]:


# Visualize how poverty percentage changed over the years
poverty_data.groupby('year')['povertyPercentage'].mean().plot(kind='line', figsize=(15, 5))
plt.title('Poverty percentage over the years')
plt.xlabel('Year')
plt.ylabel('Poverty percentage (%)')


# In[30]:


# Visualize how poverty percentage changed in each state
poverty_data.groupby('state')['povertyPercentage'].mean().plot(kind='bar', figsize=(15, 5))
plt.title('Poverty rate in each state')
plt.xlabel('State')
plt.ylabel('Poverty percentage (%)')


# ## Congressional winner data
# Contains information about the winner of the congressional elections in the USA, for each year, state and congressional district.

# In[15]:


congressional_winner_data.head()


# In[16]:


print('Number of entries: ', congressional_winner_data.shape[0])
print('Numebr of columns: ', congressional_winner_data.shape[1])


# In[17]:


# Check for missing values
congressional_winner_data.isnull().sum()


# In[18]:


# Check for duplicate entries
congressional_winner_data.duplicated().sum()


# In[36]:


print('Range of years: ', congressional_winner_data['year'].min(), congressional_winner_data['year'].max())
print('Number of states: ', congressional_winner_data['state'].nunique())
print('Number of parties: ', congressional_winner_data['party'].nunique())
print('Parties: ', congressional_winner_data['party'].unique())


# In[43]:


print('Total number of total votes won by each party')
congressional_winner_data.groupby('party')['totalvotes'].sum()


# In[46]:


# Visualize how many votes each party won in 2022
congressional_winner_data[congressional_winner_data['year'] == 2022].groupby('party')['totalvotes'].sum().plot(kind='bar', figsize=(15, 5))
plt.title('Total number of votes won by each party in 2022')
plt.xlabel('Party')
plt.ylabel('Total votes')


# In[48]:


# Visualize how many votes democrats won in 2022 in each state
congressional_winner_data[congressional_winner_data['year'] == 2022][congressional_winner_data['party'] == 'DEMOCRAT'].groupby('state')['totalvotes'].sum().plot(kind='bar', figsize=(15, 5))
plt.title('Total number of votes won by democrats in 2022 in each state')
plt.xlabel('State')
plt.ylabel('Total votes')


# In[49]:


# Visualize how many votes republicans won in 2022 in each state
congressional_winner_data[congressional_winner_data['year'] == 2022][congressional_winner_data['party'] == 'REPUBLICAN'].groupby('state')['totalvotes'].sum().plot(kind='bar', figsize=(15, 5))
plt.title('Total number of votes won by republicans in 2022 in each state')
plt.xlabel('State')
plt.ylabel('Total votes')

