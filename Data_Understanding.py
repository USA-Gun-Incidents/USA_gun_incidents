# %% [markdown]
# 

# %%
#librerie utili ed eventuali
#%matplotlib inline
import math
import numpy as np
import pandas as pd
#import scipy.stats as stats
import matplotlib.pyplot as plt

from collections import defaultdict

# %%
#carico il dataset, forse male? forse bene? boh!
inc = pd.read_csv('gun-data\data\incidents.csv', sep=',') 
#tipi feaetures dei pattern
inc.info()

# %%
inc.head()

# %%
#in realtà sembra caricato bene
#godo

#eliminiamo duplicati
inc_no_dup = inc.drop_duplicates()
#inc_dup = pd.DataFrame(columns=inc.columns)
#inc_dup.info()

#creo una lista delle righe che voglio usare per creare un nuovo dataset
inc_dup = []
#print(inc.columns)
#print(inc.index)

for if_dup, x in zip(inc.duplicated(keep=False), inc.to_records()): #serie di booleani che indicano se un elemento è o meno duplicato

    #se è duplicata la inserisco
    if if_dup:
        #il primo elemeto della tupla è l'indice da rimuovere 
        inc_dup.append(list(x)[1:])

#creo il dataframe, dovrebbe essere la stratgia meno costosa
inc_dup = pd.DataFrame(data=inc_dup, columns=inc.columns)
#stampa info
inc_dup.info()
    

# %%
inc_dup.drop()


