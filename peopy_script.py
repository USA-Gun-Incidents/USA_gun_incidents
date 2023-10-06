#lib
import numpy as np
import pandas as pd
import scipy as stats
import matplotlib.pyplot as plt
from datetime import datetime
import os
from geopy.geocoders import Nominatim
import json
import time
import random

DIR = cur_path = os.path.dirname(__file__) + '\\data\\geopy\\'
OUT_FILE = open(DIR + 'geopy.txt', 'a+')
ERR_FILE = open(DIR + 'geopy_error.txt', 'a+')
STATS = open(DIR + 'geopy_stats.txt', 'a+')
CONFIG = open(DIR + 'geopy_config.txt', 'r')

start = int(CONFIG.readline())
CONFIG.close()

PAUSES = [0.1, 1, 4]
if type(start) is not int:
    print('ERRORE CONFIG')
    exit()
else:
    print('started from: ' + str(start))

OUT_FILE.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------- \tSTART\t ----------\n')
STATS.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------- \tSTART\t ----------\n')
ERR_FILE.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------- \tSTART\t ----------\n')

#the dataset is probably read rightcon
inc = pd.read_csv(os.path.dirname(__file__) + '\\data\\incidents.csv', sep=',') 
#inc.drop_duplicates(subset=['latitude', 'longitude'], inplace=True)


geolocator = Nominatim(user_agent="DM_project")

stats_d = {'total_r':0, 'partial_r':0, 'total_em':0, 'partial_em':0, 'total_ef':0, 'partial_ef':0, 'rate':-1}
stats_t = time.time()
stats_ind = 0
stats_rate_alpha = 0.2
for index, row in inc.iloc[start:].iterrows():

    er = 0
    while er < 3:
        try:
            location = geolocator.reverse(str(row['latitude']) + ' ' + str(row['longitude'])).raw
            json.dump([index, location], OUT_FILE)
            OUT_FILE.write('\n')

            stats_d['total_r'] += 1
            stats_d['partial_r'] += 1

            dummy = random.randrange(0, 100)
            if dummy < 2:
                print('SUPER PING PONG')
            elif dummy < 51 :
                print('ping')
            else:
                print('pong')
            
            er = 10

        except Exception as e:
            print('ERR: ' + str(er))
            time.sleep(PAUSES[er])
            er += 1
            if er >= 3:
                stats_d['total_ef'] += 1
                stats_d['partial_ef'] += 1
                ERR_FILE.write('date: ' + str(datetime.now()) + ' index: ' + str(index) + ' coord: ' + str(row['latitude']) + ' ' + str(row['longitude']) + ' err: ' + str(e) + '\n')
                ERR_FILE.flush()
            else:
                stats_d['total_em'] += 1
                stats_d['partial_em'] += 1
    
    if stats_ind > 25:

        stats_t = time.time() - stats_t
        if stats_d['rate'] == -1:
            stats_d['rate'] = stats_d['partial_r']/stats_t
        else:
            stats_d['rate'] = (1 - stats_rate_alpha)*stats_d['rate'] + stats_rate_alpha*(stats_d['partial_r']/stats_t)


        stats_ind = 0
        json.dump(stats_d, STATS)
        STATS.write('\n')
        STATS.flush()
        stats_t = time.time()

        stats_d['partial_ef'] = 0
        stats_d['partial_em'] = 0
        stats_d['partial_r'] = 0
        
        break
    else:
        stats_ind += 1




OUT_FILE.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------------------- END ---------------------\n')
STATS.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------------------- END ---------------------\n')
ERR_FILE.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------------------- END ---------------------\n')

OUT_FILE.close()
ERR_FILE.close()
STATS.close()