# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as plt
import geopy.distance
import plot_utils

# %%
dirname = os.path.dirname(' ')
data = pd.read_csv(os.path.join(dirname, 'data/post_proc/final_incidents_2.csv'), index_col=0, dtype={'latitude':float, 'logitude':float}, low_memory=False)
data.groupby(['state', 'county', 'city']).count()

# %%
data['latitude'].isna().sum()

# %%
for f in data['latitude']:
    a = []
    a.append(np.isnan(f))
sum(a)

# %%
for i in data.loc[data['city'].isna() & np.isnan(data['latitude'])].index:
    print(data.loc[i]['latitude'])

# %%
a = len(data.loc[(data['latitude'].notna()) & (data['county'].notna()) & (data['city'].notna())])
b = len(data.loc[(data['latitude'].notna()) & (data['county'].notna()) & (data['city'].isna())])
c = len(data.loc[(data['latitude'].notna()) & (data['county'].isna()) & (data['city'].notna())])
d = len(data.loc[(data['latitude'].notna()) & (data['county'].isna()) & (data['city'].isna())])
e = len(data.loc[(data['latitude'].isna()) & (data['county'].notna()) & (data['city'].notna())])
f = len(data.loc[(data['latitude'].isna()) & (data['county'].notna()) & (data['city'].isna())])
g = len(data.loc[(data['latitude'].isna()) & (data['county'].isna()) & (data['city'].notna())])
h = len(data.loc[(data['latitude'].isna()) & (data['county'].isna()) & (data['city'].isna())])

print('LAT/LONG --- COUNTY --- CITY')
print( ' 0 --- 0 --- 0\t', a)
print( ' 0 --- 0 --- 1\t', b)
print( ' 0 --- 1 --- 0\t', c)
print( ' 0 --- 1 --- 1\t', d)
print( ' 1 --- 0 --- 0\t', e)
print( ' 1 --- 0 --- 1\t', f)
print( ' 1 --- 1 --- 0\t', g)
print( ' 1 --- 1 --- 1\t', h)
print( ' ---- TOT ----\t', a+b+c+d+e+f+g+h)
print( ' ---- GOOD ---\t', a+b+c+d)
print( ' ---- BAD ----\t', e+f+g+h)

# %%
centroids = data.loc[data['latitude'].notna() & data['city'].notna()][['latitude', 'longitude', 'city', 'state', 'county']].groupby(['state', 'county', 'city']).mean()
centroids.head(10)

# %%
print(centroids.index.to_list())

# %%
centroids.sample()

# %%
info_city = pd.DataFrame(columns=['5', '15', '25', '35', '45', '55', '65', '75', '85', '95', 'tot_points', 'min', 'max', 'avg', 'centroid_lat', 'centroid_lon'], index=centroids.index)
info_city.info()

# %%
for state, county, city in centroids.index:
    dummy = []
    for lat, long in zip(data.loc[(data['city'] == city) & (data['state'] == state) & (data['county'] == county) & data['latitude'].notna()]['latitude'], 
                         data.loc[(data['city'] == city) & (data['state'] == state) & (data['county'] == county) & data['longitude'].notna()]['longitude']):
        dummy.append(geopy.distance.geodesic([lat, long], centroids.loc[state, county, city]).km)
    dummy = sorted(dummy)
    pc = np.quantile(dummy, np.arange(0,1, 0.05))
    for i in range(len(info_city.columns) - 6):
        info_city.loc[state, county, city][i] = pc[i*2 + 1]
    info_city.loc[state, county, city][len(info_city.columns) - 6] = len(dummy)
    info_city.loc[state, county, city][len(info_city.columns) - 5] = min(dummy)
    info_city.loc[state, county, city][len(info_city.columns) - 4] = max(dummy)
    info_city.loc[state, county, city][len(info_city.columns) - 3] = sum(dummy)/len(dummy)
    info_city.loc[state, county, city][len(info_city.columns) - 2] = centroids.loc[state, county, city]['latitude']
    info_city.loc[state, county, city][len(info_city.columns) - 1] = centroids.loc[state, county, city]['longitude']




# %%
info_city

# %%
info_city.loc[info_city['tot_points'] > 1].info()

# %%
plot_utils.plot_scattermap_plotly(info_city, 'tot_points', x_column='centroid_lat', y_column='centroid_lon', hover_name=False)

# %%
for i in [  5955,  19567,  22995,  23433,  35631,  39938,  45163,  55557,  55868,
        60596,  65016,  69992,  70730,  73290,  73949,  78689, 104390, 116673,
       133043, 150273, 153933, 160492, 162559, 178887, 192938, 196820, 206125,
       225494, 227231, 227287, 230283]:
       print(data.iloc[i][['latitude', 'longitude']])
print(data.iloc[i])

# %%
data.sample()

# %%
def substitute_city(row, info_city):
    if pd.isna(row['city']) and not np.isnan(row['latitude']):
        for state, county, city in info_city.index:
            if row['state'] == state and row['county'] == county:
                if info_city.loc[state, county, city]['tot_points'] > 1:
                    max_radius = info_city.loc[state, county, city]['75'] #0.75 esimo quantile
                    centroid_coord = [info_city.loc[state, county, city]['centroid_lat'], info_city.loc[state, county, city]['centroid_lon']]
                    if geopy.distance.geodesic([row['latitude'], row['longitude']], centroid_coord).km <= max_radius:
                            row['city'] = city
                            break
                    
    return row


final_data = data.apply(lambda row: substitute_city(row, info_city), axis=1)

# %%
a = len(final_data.loc[(final_data['latitude'].notna()) & (final_data['county'].notna()) & (final_data['city'].notna())])
b = len(final_data.loc[(final_data['latitude'].notna()) & (final_data['county'].notna()) & (final_data['city'].isna())])
c = len(final_data.loc[(final_data['latitude'].notna()) & (final_data['county'].isna()) & (final_data['city'].notna())])
d = len(final_data.loc[(final_data['latitude'].notna()) & (final_data['county'].isna()) & (final_data['city'].isna())])
e = len(final_data.loc[(final_data['latitude'].isna()) & (final_data['county'].notna()) & (final_data['city'].notna())])
f = len(final_data.loc[(final_data['latitude'].isna()) & (final_data['county'].notna()) & (final_data['city'].isna())])
g = len(final_data.loc[(final_data['latitude'].isna()) & (final_data['county'].isna()) & (final_data['city'].notna())])
h = len(final_data.loc[(final_data['latitude'].isna()) & (final_data['county'].isna()) & (final_data['city'].isna())])

print('LAT/LONG --- COUNTY --- CITY')
print( ' 0 --- 0 --- 0\t', a)
print( ' 0 --- 0 --- 1\t', b)
print( ' 0 --- 1 --- 0\t', c)
print( ' 0 --- 1 --- 1\t', d)
print( ' 1 --- 0 --- 0\t', e)
print( ' 1 --- 0 --- 1\t', f)
print( ' 1 --- 1 --- 0\t', g)
print( ' 1 --- 1 --- 1\t', h)
print( ' ---- TOT ----\t', a+b+c+d+e+f+g+h)
print( ' ---- GOOD ---\t', a+b+c+d)
print( ' ---- BAD ----\t', e+f+g+h)

# %%
final_data.to_csv(os.path.join(dirname, 'data/post_proc/final_incidents_city_inf.csv'))
info_city.to_csv(os.path.join(dirname, 'data/post_proc/info_city.csv'))


# %%
a = len(data.loc[(data['latitude'].notna()) & (data['county'].notna()) & (data['city'].notna())])
b = len(data.loc[(data['latitude'].notna()) & (data['county'].notna()) & (data['city'].isna())])
c = len(data.loc[(data['latitude'].notna()) & (data['county'].isna()) & (data['city'].notna())])
d = len(data.loc[(data['latitude'].notna()) & (data['county'].isna()) & (data['city'].isna())])
e = len(data.loc[(data['latitude'].isna()) & (data['county'].notna()) & (data['city'].notna())])
f = len(data.loc[(data['latitude'].isna()) & (data['county'].notna()) & (data['city'].isna())])
g = len(data.loc[(data['latitude'].isna()) & (data['county'].isna()) & (data['city'].notna())])
h = len(data.loc[(data['latitude'].isna()) & (data['county'].isna()) & (data['city'].isna())])

print('LAT/LONG --- COUNTY --- CITY')
print( ' 0 --- 0 --- 0\t', a)
print( ' 0 --- 0 --- 1\t', b)
print( ' 0 --- 1 --- 0\t', c)
print( ' 0 --- 1 --- 1\t', d)
print( ' 1 --- 0 --- 0\t', e)
print( ' 1 --- 0 --- 1\t', f)
print( ' 1 --- 1 --- 0\t', g)
print( ' 1 --- 1 --- 1\t', h)
print( ' ---- TOT ----\t', a+b+c+d+e+f+g+h)
print( ' ---- GOOD ---\t', a+b+c+d)
print( ' ---- BAD ----\t', e+f+g+h)

# %%
plot_utils.plot_scattermap_plotly(data.loc[(data['latitude'].notna()) & (data['county'].notna()) & (data['city'].isna())], 'state')

# %%
plot_utils.plot_scattermap_plotly(data.loc[(data['latitude'].notna()) & (data['state'] == 'Missouri') & (data['county'] == 'Platte County') & (data['city'] == 'Kansas City')], 'latitude')
len(data.loc[(data['latitude'].notna()) & (data['state'] == 'Missouri') & (data['county'] == 'Platte County') & (data['city'] == 'Kansas City')])



