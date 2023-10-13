# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

tags = ["Firearm", "Shots", "Suicide", "Injuries", "Death", "Road", "Illegal holding", "House", "School", "Children", "Drugs", "Officiers", "Organized", "Social reasons", "Defensive", "Workplace"]
zeros = [0] * incidents_data_characteristics.shape[0]

for tag in tags:
    incidents_data_characteristics.insert(incidents_data_characteristics.shape[1], tag, zeros)
  
incidents_data_characteristics


