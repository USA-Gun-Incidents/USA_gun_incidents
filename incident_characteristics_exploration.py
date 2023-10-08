# %% [markdown]
# # Incident characteristics

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# read data
FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'

incidents_data = pd.read_csv(incidents_path)

# %%
incidents_data[['incident_characteristics1', 'incident_characteristics2']]

# %% [markdown]
# ### Incident characteristics 1

# %%
# incident_characteristics1
print('number of classes for incident_characteristics1: ', len(incidents_data['incident_characteristics1'].unique()))
display(incidents_data['incident_characteristics1'].unique())

# %%
incidents_data['incident_characteristics1'].value_counts()

# %%
# number of classes for incident_characteristics1 with more that 5 occurrences
print('Number of classes for incident_characteristics1 with more that 5 occurrences: ', 
len(incidents_data['incident_characteristics1'].value_counts()[incidents_data['incident_characteristics1'].value_counts() > 5]))

print('Number of classes for incident_characteristics1 with more that 10 occurrences: ', 
len(incidents_data['incident_characteristics1'].value_counts()[incidents_data['incident_characteristics1'].value_counts() > 10]))

print('Number of classes for incident_characteristics1 with more that 50 occurrences: ', 
len(incidents_data['incident_characteristics1'].value_counts()[incidents_data['incident_characteristics1'].value_counts() > 50]))

print('Number of classes for incident_characteristics1 with more that 100 occurrences: ', 
len(incidents_data['incident_characteristics1'].value_counts()[incidents_data['incident_characteristics1'].value_counts() > 100]))

print('Number of classes for incident_characteristics1 with more that 500 occurrences: ', 
len(incidents_data['incident_characteristics1'].value_counts()[incidents_data['incident_characteristics1'].value_counts() > 500]))

# %%
# plot the distribution of incident_characteristics1 for class with more than 100 samples
incidents_data['incident_characteristics1'].value_counts()[incidents_data['incident_characteristics1'].value_counts() 
    > 100].plot(kind='bar', figsize=(20, 6))
plt.grid()
plt.xlabel('incident characteristics')
plt.ylabel('number of samples')
plt.show()

# %%
# plot the distribution of incident_characteristics1 for class with more than 500 samples
incidents_data['incident_characteristics1'].value_counts()[incidents_data['incident_characteristics1'].value_counts() 
    > 500].plot(kind='bar', figsize=(20, 6))
plt.grid()
plt.xlabel('incident characteristics')
plt.ylabel('number of samples')
plt.show()

# %% [markdown]
# ### Incident characteristics 2

# %%
# incident_characteristics2
print('number of classes for incident_characteristics2: ', len(incidents_data['incident_characteristics2'].unique()))
display(incidents_data['incident_characteristics2'].unique())

# %%
incidents_data['incident_characteristics2'].value_counts()

# %%
print('Number of classes for incident_characteristics2 with more that 5 occurrences: ', 
len(incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() > 5]))

print('Number of classes for incident_characteristics2 with more that 10 occurrences: ', 
len(incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() > 10]))

print('Number of classes for incident_characteristics2 with more that 50 occurrences: ', 
len(incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() > 50]))

print('Number of classes for incident_characteristics2 with more that 100 occurrences: ', 
len(incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() > 100]))

print('Number of classes for incident_characteristics2 with more that 500 occurrences: ', 
len(incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() > 500]))

print('Number of classes for incident_characteristics2 with more that 1000 occurrences: ', 
len(incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() > 1000]))

print('Number of classes for incident_characteristics2 with more that 5000 occurrences: ', 
len(incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() > 5000]))

print('Number of classes for incident_characteristics2 with more that 10000 occurrences: ', 
len(incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() > 10000]))

# %%
# plot the distribution of incident_characteristics2 for class with more than 500 samples
incidents_data['incident_characteristics2'].value_counts()[incidents_data['incident_characteristics2'].value_counts() 
    > 500].plot(kind='bar', figsize=(20, 6))
plt.grid()
plt.xlabel('incident characteristics')
plt.ylabel('number of samples')
plt.show()

# %% [markdown]
# ### Correlation between incidente characteristics

# %%
type(incidents_data['incident_characteristics1'][2])

# %%
type(incidents_data['incident_characteristics2'][0])

# %%
# convert incident_characteristics1 and incident_characteristics2 to string
incidents_data['incident_characteristics1'] = incidents_data['incident_characteristics1'].astype(str)
incidents_data['incident_characteristics2'] = incidents_data['incident_characteristics2'].astype(str)

# %%
# scatter plot of incident_characteristics1 and incident_characteristics2
plt.figure(figsize=(20, 9))
plt.scatter(incidents_data['incident_characteristics2'], incidents_data['incident_characteristics1'])
plt.xlabel('incident_characteristics2')
plt.ylabel('incident_characteristics1')
plt.xticks([])
plt.yticks([])
plt.show()

# %%
# incident_characteristics1 class with more than 500 samples
list_characteristics1 = incidents_data['incident_characteristics1'].value_counts()[
    incidents_data['incident_characteristics1'].value_counts() > 500]
list_characteristics1 = list_characteristics1.index.tolist()
list_characteristics1

# %%
# characteristcs1 class distribution
plt.figure(figsize=(20, 6))
plt.bar(list_characteristics1, incidents_data['incident_characteristics1'].value_counts()[list_characteristics1])
plt.xticks(rotation=90)
plt.grid()
plt.xlabel('incident characteristics')
plt.ylabel('number of samples')
plt.show()

# %%
# incident_characteristics2 class with more than 1000 samples
list_characteristics2 = incidents_data['incident_characteristics2'].value_counts()[
    incidents_data['incident_characteristics2'].value_counts() > 1000]
del list_characteristics2['nan'] # delete nan class from list
list_characteristics2 = list_characteristics2.index.tolist()
list_characteristics2

# %%
# characteristcs2 class distribution
plt.figure(figsize=(20, 6))
plt.bar(list_characteristics2, incidents_data['incident_characteristics2'].value_counts()[list_characteristics2])
plt.xticks(rotation=90)
plt.grid()
plt.xlabel('incident characteristics')
plt.ylabel('number of samples')
plt.show()

# %%
# scatterplot for incidents with incident_characteristics1 in list_characteristics1
plt.figure(figsize=(20, 6))
plt.scatter(incidents_data['incident_characteristics2'][incidents_data['incident_characteristics1'].isin(list_characteristics1)], 
            incidents_data['incident_characteristics1'][incidents_data['incident_characteristics1'].isin(list_characteristics1)])
plt.xlabel('incident_characteristics2')
plt.ylabel('incident_characteristics1')
plt.xticks(rotation=90)
plt.show()


# %%
# scatterplot for incidents with incident_characteristics2 in list_characteristics2
plt.figure(figsize=(20, 12))
plt.scatter(incidents_data['incident_characteristics2'][incidents_data['incident_characteristics2'].isin(list_characteristics2)], 
            incidents_data['incident_characteristics1'][incidents_data['incident_characteristics2'].isin(list_characteristics2)])
plt.xlabel('incident_characteristics2')
plt.ylabel('incident_characteristics1')
plt.xticks(rotation=90)
plt.show()


