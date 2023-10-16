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
# drop duplicates rows
incidents_data.drop_duplicates(inplace=True)

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

# %% [markdown]
# ### Use incident_characteristics1 and incident_characteristics2 as the same feature

# %%
# check if some class of incident_characteristics1 are in incident_characteristics2 and vice versa
common_class = incidents_data['incident_characteristics1'][incidents_data['incident_characteristics1'
    ].isin(list_characteristics2)].unique()
common_class


# %%
len(common_class)

# %%
# check if same sample have the same class for incident_characteristics1 and incident_characteristics2
incidents_data['incident_characteristics1'][incidents_data['incident_characteristics1'
    ].isin(list_characteristics2)].equals(incidents_data['incident_characteristics2'][incidents_data['incident_characteristics2'
    ].isin(list_characteristics1)])

# %%
# crreate a list with all the classes of incident_characteristics1 and incident_characteristics2 with out repetition
list_characteristics = incidents_data['incident_characteristics1'].unique().tolist() + incidents_data[
    'incident_characteristics2'].unique().tolist()

# remove repetition
for c in common_class:
    del list_characteristics[list_characteristics.index(c)]


# %%
len(list_characteristics) # 53 + 91 - 26 = 118

# %%
# order list_characteristics alphabetically
list_characteristics.sort()

# delete nan class from list_characteristics
del list_characteristics[list_characteristics.index('nan')]
del list_characteristics[list_characteristics.index('nan')]

list_characteristics

# %%
# function to map the classes of incident_characteristics1 and incident_characteristics2 to a number in range 0-117, alphabetically
def map_characteristics(characteristics):
    if characteristics in list_characteristics:
        return list_characteristics.index(characteristics)
    else:
        return -1 # 'nan'

# %%
# map incident_characteristics1 and incident_characteristics2 to a number in range 0-117, alphabetically
incidents_data['incident_char1'] = incidents_data['incident_characteristics1'].map(map_characteristics)
incidents_data['incident_char2'] = incidents_data['incident_characteristics2'].map(map_characteristics)

# %%
# create one only feature for incident_characteristics
for index, row in incidents_data.iterrows():
    first_char = min(row['incident_char1'], row['incident_char2'])
    second_char = max(row['incident_char1'], row['incident_char2'])
    incidents_data.loc[index, 'numeric_characteristics1'] = first_char
    incidents_data.loc[index, 'numeric_characteristics2'] = second_char
    incidents_data.loc[index, 'numeric_characteristics'] = str(first_char) + '-' + str(second_char)

# %% [markdown]
# ### Characteristics as a unique feature 

# %%
# number of samples for each class of numeric_characteristics
incidents_data['numeric_characteristics'].value_counts()

# %% [markdown]
# -1 is 'nan'

# %%
# number of classes for numeric_characteristics
len(incidents_data['numeric_characteristics'].unique())

# %%
# plot the distribution of numeric_characteristics for class with more than 1000 samples
incidents_data['numeric_characteristics'].value_counts()[incidents_data['numeric_characteristics'].value_counts() 
    > 1000].plot(kind='bar', figsize=(20, 6))
plt.grid()
plt.xlabel('incident characteristics')
plt.ylabel('number of samples')
plt.xticks(rotation=90)
plt.show()

# %%
# save in a .txt file the list of classes of numeric_characteristics and the associate number
# add also the nan class with -1 number
with open(FOLDER + 'post_proc/numeric_characteristics.txt', 'w') as f:
    for i, c in enumerate(list_characteristics):
        f.write(str(i) + ' ' + c + '\n')
    f.write('-1 nan')

# %%
# save in a .csv file the numeric_characteristics
incidents_data[['numeric_characteristics']].to_csv(
    FOLDER + 'post_proc/numeric_characteristics_colums.csv', index=False)


