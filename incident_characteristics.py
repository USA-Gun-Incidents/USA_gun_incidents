# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'

# Load data from csv files
incidents_data = pd.read_csv(incidents_path)

# %%
# plot characteristics
incidents_data_characteristics = incidents_data[['incident_characteristics1', 'incident_characteristics2']]
incidents_data_characteristics.head(10)

# %%
# main characteristics
print(incidents_data.pivot_table(columns=['incident_characteristics1'], aggfunc='size').sort_values(ascending=False))

# %%
#detailed characteristics
print(incidents_data.pivot_table(columns=['incident_characteristics2'], aggfunc='size').sort_values(ascending=False))

# %%
print(incidents_data['incident_characteristics1'].nunique())
print(incidents_data['incident_characteristics2'].nunique())

# %%
#create dataframe of characteristics corrispondence
dictionary_descriptions = {}
for characteristic1 in incidents_data['incident_characteristics1'].unique():
    util = {}
    for characteristic2 in incidents_data['incident_characteristics2'].unique():
        util[characteristic2] = 0
    dictionary_descriptions[characteristic1] = util

df_descriptions = pd.DataFrame(dictionary_descriptions)
df_descriptions

# %%
def characteristic2_to_int(characteristic):
    try:
        return df_descriptions.axes[0].get_loc(characteristic)
    except ValueError as ve:
        return -1
    
def characteristic1_to_int(characteristic):
    try:
        return df_descriptions.axes[1].get_loc(characteristic)
    except ValueError as ve:
        return -1

# %%
#set corrispondence values in the dataframe
for index, record in incidents_data_characteristics.iterrows():
    df_descriptions.iloc[characteristic2_to_int(record["incident_characteristics2"]), [characteristic1_to_int(record["incident_characteristics1"])]] += 1

df_descriptions

# %%
sns.heatmap(df_descriptions[["Shot - Wounded/Injured"]].sort_values(by="Shot - Wounded/Injured", inplace=False, ascending=False).tail(-1))

# %%
all_characteristics = list(incidents_data['incident_characteristics1'].unique()) + list(incidents_data['incident_characteristics2'].unique())
all_characteristics = np.array(all_characteristics)
all_characteristics = np.unique(all_characteristics)

#all_characteristics

print(incidents_data.pivot_table(columns=['incident_characteristics2'], aggfunc='size').sort_values(ascending=False)[39:])

# %%
#add tags to dataframe

tags = ["Firearm", "Shots", "Aggression", "Suicide", "Injuries", "Death", "Road", "Illegal holding", "House", "School", "Children", "Drugs", "Officers", "Organized", "Social reasons", "Defensive", "Workplace"]
zeros = [False] * incidents_data_characteristics.shape[0]

for tag in tags:
    incidents_data_characteristics.insert(incidents_data_characteristics.shape[1], tag, zeros)
  
incidents_data_characteristics

# %%
import csv

#read csv to populate tags

characteristics_folder_name = FOLDER + 'characteristics_tags_mapping/'
filename = characteristics_folder_name + 'characteristics_tags_mapping.csv'
characteristics_tags_list = []
with open(filename, mode='r') as file: #create list with categories and tags
    for line in file:
        line_read = line.rstrip()
        characteristics_tags_list.append(line_read.split(';'))

indicization_list = []
for lst in characteristics_tags_list:
    indicization_list.append(lst[0])

# %%
#given characteristic
#return all the tags 
def get_tags(characteristic):
    if not isinstance(characteristic, str): #if is nan
        return []
    index = indicization_list.index(characteristic)
    tags = characteristics_tags_list[index]
    if len(tags) == 1:
        return []
    return tags[1:]

# %%
for index, record in incidents_data_characteristics.iterrows():
    tags = set(get_tags(record['incident_characteristics1']) + get_tags(record['incident_characteristics2']))
    for tag in tags: #set values to tags binary mask
        incidents_data_characteristics.at[index, tag] = True


incidents_data_characteristics

# %%
tag_consistency_attr_name = "Tag Consistency"
col = [True] * incidents_data_characteristics.shape[0] #tag consistency assumed true
incidents_data_characteristics.insert(incidents_data_characteristics.shape[1], tag_consistency_attr_name, col)

incidents_data_characteristics

# %%
#consistency check
shooting_inconsistencies = 0
aggression_inconsistencies = 0
for index, record in incidents_data_characteristics.iterrows():
    if((record["incident_characteristics1"] == "Non-Shooting Incident" or record["incident_characteristics2"] == "Non-Shooting Incident") and
       record["Shots"] == True): #consistency for non-shooting incidents
        incidents_data_characteristics.at[index, tag_consistency_attr_name] = False
        shooting_inconsistencies += 1
    elif((record["incident_characteristics1"] == "Non-Aggression Incident" or record["incident_characteristics2"] == "Non-Aggression Incident") and
        record["Aggression"] == True): #consistency for non-shooting incidents
        incidents_data_characteristics.at[index, tag_consistency_attr_name] = False
        aggression_inconsistencies += 1

print("Non-Shooting Incident inconcistencies: " + str(shooting_inconsistencies))
print("Non-Aggression Incident inconsistencies: " + str(aggression_inconsistencies))

# %%
incidents_data_characteristics = incidents_data_characteristics.drop(["incident_characteristics1", "incident_characteristics2"], axis=1)

incidents_data_characteristics

# %%
#concatenate tags on original dataset
incidents_data = pd.concat([incidents_data, incidents_data_characteristics], axis=1)

incidents_data

# %%
from pathlib import Path

filename = FOLDER + 'post_proc/incidents_with_tags.csv'
filepath = Path(filename)
incidents_data.to_csv(filepath)


