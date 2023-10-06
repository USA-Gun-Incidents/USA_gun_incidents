
'''For those who want to contribute to data collection, 
choose a correct index, put it as an integer in the first line of the config file,
 and put your name in the variable NAME.

the program should run smoothly without errors, it should save everything it imports, 
and to stop it just do ctrl + c and save the state (it doesn't update the config though)

it is important to create the config file before running it,
look at mine for syntax and folders location'''


import pandas as pd
from datetime import datetime
import os
from geopy.geocoders import Nominatim
import json
import time
import random


#PUT NAME HERE
#(STRING)
NAME = 'giacomo'

DIR = cur_path = os.path.dirname(__file__) + '\\data\\geopy\\' + NAME + '_'
OUT_FILE = open(DIR + 'geopy.txt', 'a+')
ERR_FILE = open(DIR + 'geopy_error.txt', 'a+')
STATS_FILE = open(DIR + 'geopy_stats.txt', 'a+')
CONFIG = open(DIR + 'geopy_config.txt', 'r+')

start = int(CONFIG.readline())
CONFIG.close()

#seconds before retrying the request
PAUSES = [0.1, 1, 4]
if type(start) is not int:
    print('ERRORE CONFIG')
    exit()
else:
    print('started from: ' + str(start))

#inizializing files
OUT_FILE.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------------------- \tSTART\t ----------------------\n')
STATS_FILE.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------------------- \tSTART\t ----------------------\n')
ERR_FILE.write('#DATE: ' + str(datetime.now()) +'\t' + str(start) +  '\t ---------------------- \tSTART\t ----------------------\n')


#i chose to not drop any row because they are approximately 250 <<<<<< total row
inc = pd.read_csv(os.path.dirname(__file__) + '\\data\\incidents.csv', sep=',') 
#inc.drop_duplicates(subset=['latitude', 'longitude'], inplace=True)

#i don't know if is needed to change the name????
geolocator = Nominatim(user_agent="DM_project_" + NAME)

stats_d = {'total_r':0, 'partial_r':0, 'total_em':0, 'partial_em':0, 'total_ef':0, 'partial_ef':0, 'rate':-1}
stats_t = time.time()
stats_ind = 0
stats_rate_alpha = 0.2

try:
    for index, row in inc.iloc[start:].iterrows():

        er = 0
        while er < 3:
            try:
                #request
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

            #let's try again up to 3 times
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
        
        #just some stats
        if stats_ind > 300:

            stats_t = time.time() - stats_t
            if stats_d['rate'] == -1:
                stats_d['rate'] = stats_d['partial_r']/stats_t
            else:
                stats_d['rate'] = (1 - stats_rate_alpha)*stats_d['rate'] + stats_rate_alpha*(stats_d['partial_r']/stats_t)


            stats_ind = 0
            json.dump(stats_d, STATS_FILE)
            STATS_FILE.write('\n')
            STATS_FILE.flush()
            stats_t = time.time()

            stats_d['partial_ef'] = 0
            stats_d['partial_em'] = 0
            stats_d['partial_r'] = 0
            
        else:
            stats_ind += 1

except KeyboardInterrupt:
    print('INTERRUPTION...')


#termination
OUT_FILE.write('#DATE: ' + str(datetime.now()) + '\t ---------------------- END ---------------------\n')
json.dump(stats_d, STATS_FILE)
STATS_FILE.write('\n')
STATS_FILE.write('#DATE: ' + str(datetime.now()) + '\t ---------------------- END ---------------------\n')
ERR_FILE.write('#DATE: ' + str(datetime.now()) + '\t ---------------------- END ---------------------\n')

#closing all open files
OUT_FILE.close()
ERR_FILE.close()
STATS_FILE.close()


#bye bye
#ciao ciao