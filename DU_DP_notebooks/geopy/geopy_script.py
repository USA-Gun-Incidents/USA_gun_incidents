
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
import pandas
#import clean_data_utils as cdu


def start_requests():
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
    PAUSES = [0.5, 1, 3]
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
                    lat = row['latitude']
                    long = row['longitude']
                    if not pd.isna(lat) and not pd.isna(long):
                        location = geolocator.reverse(str(row['latitude']) + ' ' + str(row['longitude'])).raw
                    else:
                        er = 10
                        raise Exception('ultra lazy solution')
                    
                    json.dump([index, location], OUT_FILE)
                    OUT_FILE.write('\n')

                    stats_d['total_r'] += 1
                    stats_d['partial_r'] += 1

                    #stamp ping pong
                    dummy = random.randrange(0, 100)
                    if dummy < 2:
                        print('SUPER PING PONG')
                    elif dummy < 51 :
                        print('ping')
                    else:
                        print('pong')
                    
                    er = 100

                #let's try again up to 3 times
                except Exception as e:
                    print('ERR: ' + str(er))

                    if er >= 3:

                        stats_d['total_ef'] += 1
                        stats_d['partial_ef'] += 1
                        ERR_FILE.write('date: ' + str(datetime.now()) + ' index: ' + str(index) + ' coord: ' + str(row['latitude']) + ' ' + str(row['longitude']) + ' err: ' + str(e) + '\n')
                        ERR_FILE.flush()
                    else:
                        time.sleep(PAUSES[er])
                        er += 1
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

def unify_files(names):   
    DIR = cur_path = os.path.dirname(__file__) + '\\data\\geopy\\'

    OUT_FILE = open(DIR + 'geopy.txt', 'w+')
    ERR_FILE = open(DIR + 'geopy_error.txt', 'w+')
    STATS_FILE = open(DIR + 'geopy_stats.txt', 'w+')

    for name in names:
        DIR_N = cur_path = os.path.dirname(__file__) + '\\data\\geopy\\' + name + '_'

        OUT_FILE_N = open(DIR_N + 'geopy.txt', 'r+')
        lines = OUT_FILE_N.readlines()
        OUT_FILE_N.close()
        for l in lines:
            if not l.startswith('#'):
                OUT_FILE.write(l)

        ERR_FILE_N = open(DIR_N + 'geopy_error.txt', 'r+')
        lines = ERR_FILE_N.readlines()
        ERR_FILE_N.close()
        for l in lines:
            if not l.startswith('#'):
                ERR_FILE.write(l)

        STATS_FILE_N = open(DIR_N + 'geopy_stats.txt', 'r+')
        lines = STATS_FILE_N.readlines()
        STATS_FILE_N.close()
        for l in lines:
            if not l.startswith('#'):
                STATS_FILE.write(l)

    OUT_FILE.close()
    ERR_FILE.close()
    STATS_FILE.close()       

def check_error(file):

    error_types = {}
    
    F = open(file, 'r+')
    lines = F.readlines()
    F.close()

    for line in lines:
        line = cdu.delete_space(line)

        data = line.partition('data:')[2]
        index = int(line.partition('index:')[2].partition('coord:')[0])
        err = line.partition('err:')[2]

        if err in error_types:
            error_types[err].append(index)
        else:
            error_types[err] = [index]

    
    for k in error_types:
        print(len(error_types[k]))
        error_types[k] = list(set(error_types[k]))
        error_types[k].sort()
        print(len(error_types[k]))

    return error_types

def process_datas(IN, OUT):
    DIR = os.path.dirname(__file__) + '\\data\\geopy\\'
    STATS_FILE = DIR + 'geopy_stats.txt'

    F = open(IN, 'r+')
    lines = F.readlines()
    F.close()

    #getting all the data extracted from geopy
    datas = []
    for line in lines:
        datas.append(json.loads(line))

    #adding a tag
    for el in datas:
        el[1]['coord_presence'] = True

    #adding errors
    errors = json.load(open(STATS_FILE, 'r'))
    for key in errors:
        for el in errors[key]:
            datas.append([el, {'coord_presence':False}])
    
    datas.sort(key=lambda x: x[0])


    def extract(x):
        y = x[1]
        y['index'] = x[0]
        return x[1]
    new_datas = [extract(item) for item in datas]

    

    F = open(OUT, 'w+')
    json.dump(new_datas, F, indent=2)
    F.close()

def remove_dup(IN, OUT):
    data = json.load(open(IN, 'r'))

    result = list(
        {
            dictionary['index']: dictionary
            for dictionary in data
        }.values()
    )
    F = open(OUT, 'w+')
    json.dump(result, F, indent=2)
    F.close()
    
def to_csv(IN, OUT):
    data = json.load(open(IN, 'r'))

    for el in data:
        if el['coord_presence']:
            for key in el['address']:
                el[key] = el['address'][key]

            del el['address']


    df = pandas.DataFrame.from_dict(data)
    df.set_index('index', inplace=True)
    df.drop(['licence', 'osm_type', 'osm_id'],axis=1, inplace=True)
    df.to_csv(OUT)

def get_coord():
    from geopy import Nominatim
    import time

    geolocator = Nominatim(user_agent="DM_project_giacomo")

    addr = ['Oklahoma, Tulsa']
    for line in addr:
        location = geolocator.geocode(line)
        print (location.latitude, location.longitude)
        time.sleep(1)



DIR = cur_path = os.path.dirname(__file__) + '\\data\\geopy\\'
ERR_DIR = DIR + 'geopy_error.txt'
STATS_FILE = DIR + 'geopy_stats.txt'
FINAL_DATA = DIR + 'geopy_merged.json'
OUT_FILE = DIR + 'geopy.csv'

#unify_files(['giacomo', 'irene', 'luca', 'giulia'])
#json.dump(check_error(ERR_DIR), STATS_FILE, indent=2)

#to_csv(FINAL_DATA, OUT_FILE)

#remove_dup(FINAL_DATA, FINAL_DATA_2)

get_coord()
#geolocator = Nominatim(user_agent="DM_project")
#print('39.7462, -105.058', geolocator.reverse('39.7462, -105.058').raw)