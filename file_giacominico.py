# %% [markdown]
# Data Understanding con i Bro

# %%
%matplotlib inline

#lib
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


from collections import defaultdict

# %%
#the dataset is probably read right
inc = pd.read_csv('data/incidents.csv', sep=',') 
inc.info()

# %%
inc.nunique()

# %% [markdown]
# Firts we check the data consistecy, about the states, city and lat-long. 
# Then we can use only lat and long to refer to the position of the incident

# %%
for col in inc:
    dummy = inc[col].unique()
    print( [ col, dummy, len(dummy)] )

# %%
#eliminiamo duplicati
inc_no_dup = inc.drop_duplicates(subset=['date', 'latitude', 'longitude'])
#inc_dup = pd.DataFrame(columns=inc.columns)
#inc_dup.info()

#creo una lista delle righe che voglio usare per creare un nuovo dataset
inc_dup = []
#print(inc.columns)
#print(inc.index)

for if_dup, x in zip(inc.duplicated(), inc.to_records()): #serie di booleani che indicano se un elemento è o meno duplicato
    #se è duplicata la inserisco
    if if_dup:
        #il primo elemeto della tupla è l'indice da rimuovere 
        inc_dup.append(list(x)[1:])

#creo il dataframe, dovrebbe essere la stratgia meno costosa
inc_dup = pd.DataFrame(data=inc_dup, columns=inc.columns)
#stampa info
inc_dup.info()

plt.scatter(inc_no_dup['longitude'], inc_no_dup['latitude'], color='g', label='NOT duplicated')
plt.scatter(inc_dup['longitude'], inc_dup['latitude'], color='r',label='duplicated')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('position pattern')
plt.legend()
plt.show()

# %%
inc_no_dup.groupby(['congressional_district', 'state_house_district']).size().head(40) #fyuavbei9diuhgybhihgyv


