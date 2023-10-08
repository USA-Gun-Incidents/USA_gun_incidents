# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# %%
FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'

# %%
# Load data from csv files
incidents_data = pd.read_csv(incidents_path)

# %%
incidents_data.info()

# %%
incidents_data[incidents_data.duplicated()] 

# %%
incidents_data.drop_duplicates(inplace=True)
incidents_data.info()

# %%
def max_min(df, split_num):
    max = - np.inf
    min = + np.inf
    for _, row in df.iterrows():
        value = int(row['date'].split('-')[split_num])
        if value < min:
            min = value
        if value > max:
            max = value
    print(f"Max: {max}")
    print(f"Min: {min}")

print("Year")
max_min(incidents_data, 0)
print("Month")
max_min(incidents_data, 1)
print("Day")
max_min(incidents_data, 2)

# %%
# convert incidents_data['date] to datetime
incidents_data['date'] = pd.to_datetime(incidents_data['date'], format="%Y/%m/%d") # dovrebbe crashare con e.g. 30 febbraio
incidents_data['date'].describe()

# %%
n_months = ((incidents_data['date'].max() - incidents_data['date'].min()).days)//30

# %%
plt.figure(figsize=(20, 10))
plt.hist(incidents_data['date'], bins=n_months)
plt.title('Number of incidents by date')
plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.show()

# %%
incidents_data['date'].dt.year.value_counts().sort_index()


# %%
# get the number of incidents with same date, latitude and longitude
groups = incidents_data.groupby(['date', 'latitude', 'longitude']).size().sort_values(ascending=False)
# get groups with more than one incident
groups = groups[groups > 1]
groups

# %%
# get the rows with date 2018-01-16, latitude 36.7387 and longitude -119.7900
incidents_data[(incidents_data['date'] == '2018-01-16') & (incidents_data['latitude'] == 36.7387) & (incidents_data['longitude'] == -119.7900)]

# %% [markdown]
# California, Fresno 1225 M St è la sede dello sceriffo
# https://www.fresnosheriff.org/jail.html

# %%
incidents_data[(incidents_data['date'] == '2018-03-17') & (incidents_data['latitude'] == 36.7387) & (incidents_data['longitude'] == -119.7900)]

# %%
groups = incidents_data.groupby(['latitude', 'longitude']).size().sort_values(ascending=False)
groups = groups[groups > 1]
groups

# %%
# drop di un attributo per volta
for col in incidents_data.columns:
    incidents_data_dropped = incidents_data.drop(columns=[col])
    if incidents_data_dropped.duplicated().any() == True:
        print(col)

# %%
# TASTIERINO
# 7 8 9
# 4 5 6
# 1 2 3
# 0

# %% [markdown]
# # Record dal futuro
# 
# 2028-12-28,Wisconsin,Milwaukee,Locust and Humboldt Blvd,43.0711,-87.8977,4.0,19.0,7.0,,Adult 18+,Male,,,,0.0,0.0,1.0,1.0,0.0,0,0,0.0,1.0,1.0,"Perp, drinking for 17 hours, waved and pointed gun at officer, who disarmed him. Blew a .22 No shots fired or injuries.",,
# 
# https://www.fox6now.com/news/officer-arrests-intoxicated-suspect-for-pointing-aiming-gun
# 
# 1 gennaio 2014...

# %%
# get the unique date values in the incidents_data happened in 2028
incidents_data_2028 = incidents_data[incidents_data['date'].dt.year == 2028]
incidents_data_2028['date'].unique()


# %%
incidents_data_2029 = incidents_data[incidents_data['date'].dt.year == 2029]
incidents_data_2029['date'].unique()

# %%
incidents_data_2030 = incidents_data[incidents_data['date'].dt.year == 2030]
incidents_data_2030['date'].unique()

# %%
incidents_data_past = incidents_data[incidents_data['date'].dt.year < 2028].copy()
incidents_data_past

# %%
incidents_data_future = incidents_data[incidents_data['date'].dt.year >= 2028].copy()
incidents_data_future

# %%
incidents_data_future = incidents_data_future.drop(columns=['date'])
incidents_data_past = incidents_data_past.drop(columns=['date'])

# %%
# search for incidents_data_future in incidents_data_past without considering the date
incidents_data_past[incidents_data_past.isin(incidents_data_future).all(axis=1)]

# %%
# search for incidents_data_past in incidents_data_future without considering the date
incidents_data_future[incidents_data_future.isin(incidents_data_past).all(axis=1)]

# %%
incidents_data_future[incidents_data_future.duplicated()]

# %%
ch1 = incidents_data['incident_characteristics1'].unique()
ch1

# %%
ch2 = incidents_data['incident_characteristics2'].unique()
ch2

# %%
print("----CH1 not in CH2----")
for ch in ch1:
    if ch not in ch2:
        print(ch)

print("----CH2 NOT IN CH1----")
for ch in ch2:
    if ch not in ch1:
        print(ch)


