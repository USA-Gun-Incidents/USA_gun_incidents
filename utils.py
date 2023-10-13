####################### Geographical data cleaning #######################
import pandas as pd
import re
import jellyfish
import nltk
from stop_words import get_stop_words
from nltk.corpus import stopwords
import numpy as np

# epics variables
DAMERAU_LEVENSHTEIN_DISTANCE_TRESHOLD = 2
ADDRESS_SIMILARITY_TRESHOLD = 2
MOST_USED_WORDS = ['of', 'block', 'Street', 'and', 'Avenue', 'St', 'Ave', 'Road', 'Drive', 'Rd', 'South', 'West', 'N', 'North', 'Dr', 'S', 'W', 'E', 'East', 'Blvd', 'Boulevard']

# basic cleaning functions
def lower_case(data):
    return data.lower()

def delete_space(data):
    return data.replace(" ", "")

def split_where_parenthesis(data):
    # split data where parenthesis
    data1, data2 = data.split("(")
    # delete close parenthesis and return two strings
    return data1, data2.replace(")", "")

def check_parenthesis(data):
    if "(" in data:
        return True
    else:
        return False

def delete_punctuation(data):
    # delete all puntuction but parenthesis
    return re.sub(r'[^\w\s\(\)]', '', data)

def delete_numbers(data):
    return re.sub(r'[0-9]', '', data)

# clean data functions
def clean_data_incidents(data):
    data = delete_space(data)
    data = lower_case(data)
    data = delete_numbers(data)

    data = data.replace('county', '')

    if check_parenthesis(data):
        data1, data2 = split_where_parenthesis(data)
        data1 = delete_punctuation(data1)
        data2 = delete_punctuation(data2)

        return [data1, data2]
    else:
        data = delete_punctuation(data)
        return [data]
    
def clean_data_geopy(data):
    data = lower_case(data)
    data = delete_numbers(data)
    data = delete_punctuation(data)
    data = delete_space(data)
    
    return data.replace('county', '')

def split_by_stop_word(data):
    output = data.replace('and', '||').replace('of', '||').replace(',', '||').replace('or', '||').split('||')
    return output

# check if two strings are the same
'''def check_typo(data1, data2):

    # check if two string are different for at least two letters
    if len(data1) == len(data2):
        count = 0
        for i in range(len(data1)):
            if data1[i] != data2[i]:
                count += 1
        if count <= 2:
            return True
        
    # check if one string is a substring
    if data1 in data2 or data2 in data1:
            return True
    
    # check if one string is a substring of the other with a typo
    j = 0
    for i in range(len(data1)):
        if j < len(data2):
            if data1[i] == data2[j]:
                j += 1
            elif (j+1) < len(data2) and data1[i] == data2[j+1]:
                j += 2
            elif (j+2) < len(data2) and (data1[i] == data2[j+2]):
                j += 3
        else: break
    if j in range(len(data2)-2, len(data2)+3):
        return True
    
    # no matching
    return False
'''

def check_typo_jellyfish(s1, s2):
    if np.isnan(s1): return -1
    if np.isnan(s2): return -1

    dis = jellyfish.damerau_levenshtein_distance(s1, s2)
    return int(dis <= DAMERAU_LEVENSHTEIN_DISTANCE_TRESHOLD)

#A2 MUST BE THE GEOPY DATA
def check_address(a1, a2):
    if np.isnan(a1): return -1
    if np.isnan(a2): return -1

    for sep in MOST_USED_WORDS:
        a1 = a1.replace(sep, '|+|')
    a1_splitted = a1.split('|+|')

    cardinality_a1_in_a2 = 0
    for el_of_a1 in a1_splitted:
        if el_of_a1 in a2:
            cardinality_a1_in_a2 += 1

    return int(cardinality_a1_in_a2 <= ADDRESS_SIMILARITY_TRESHOLD)

# check consistency between two addresses
def check_consistency_geopy(row):

    # 0 -> false
    # 1 -> true
    # -1 -> null
    
    state_consistency = 0
    county_city_consistency = 0
    address_consistency = 0
    
    # STATE
    state = clean_data_incidents(row['state']) # our data
    state_geopy = clean_data_geopy(row['state_geopy']) # geopy data

    for s in state:
        dummy = check_typo_jellyfish(s, state_geopy)
        if state_consistency == 0:
            state_consistency = dummy
        if dummy == 1:
            state_consistency = dummy
        #state_consistency = state_consistency or check_typo_jellyfish(s, state_geopy)

   # CITY OR COUNTY
    incidents_couty_city = clean_data_incidents(row['city_or_county']) #our data

    geopy_couty_city_town_village = []
    geopy_col = ['county_geopy', 'city_geopy', 'town_geopy', 'village_geopy']
    for col in geopy_col:
            geopy_couty_city_town_village.append(clean_data_geopy(row[col]))

    for cc in incidents_couty_city:
        for i, val in enumerate(geopy_couty_city_town_village):
            dummy = check_typo_jellyfish(cc, val)

            if county_city_consistency == 0:
                county_city_consistency = dummy
            if dummy == 1:
                county_city_match = geopy_col[i]
                county_city_consistency = dummy


    # ADDRESS
    address_consistency = check_address(row['address'], row['display_name'])

    return state_consistency, county_city_consistency, county_city_match, address_consistency #, address_consistency, similar_words

def check_consistency_additional_data(state, county, additional_data):
    state_consistency = False
    state_current = np.nan

    if state in additional_data['State or equivalent'].unique():
        state_consistency = True
        state_current = state
    else:
        # check typo
        clean_data_geopy(state)
        for s in additional_data['State or equivalent'].unique():
            state = clean_data_geopy(s)
            if check_typo_jellyfish(state, s) == 1:
                state_consistency = True
                state_current = s
                break


    if state_consistency:
        if county in additional_data[additional_data['State or equivalent'] == state_current]['County or equivalent'].unique():
            return state_current, county
        else:
            # check typo
            county_list = clean_data_incidents(county)
            for c in additional_data[additional_data['State or equivalent'] == state_current]['County or equivalent'].unique():

                c = clean_data_geopy(c)
                for county_incidents in county_list:     
                    if check_typo_jellyfish(county_incidents, c) == 1:
                        return state_current, c

    '''
    else:
        county1, county2 = clean_data_city_county(county)
        for c in additional_data['County or equivalent']:
            if c is not None and type(c) == str:
                clean_data_state(c)
                if check_typo(county1, c):
                    county_current = c
                    state_current = additional_data[additional_data['County or equivalent'] == c]['State or equivalent'].values[0]
                    return state_current, county_current
                elif check_typo(county2, c):
                    county_current = c
                    state_current = additional_data[additional_data['County or equivalent'] == c]['State or equivalent'].values[0]
                    return state_current, county_current
    
    if state_consistency:
        return state_current, None
    else:
        return None, None'''
    
    return state_current, np.nan
    

# main function
def check_geographical_data_consistency(row, additional_data):
    # initialize clean_geo_data
    clean_geo_data_row = pd.Series(index=['state', 'county', 'city', 'display_name', 'latitude', 'longitude'], dtype=str)
    
    #columns=['state', 'city_or_county', 'address', 'latitude', 'longitude', 
    #'display_name', 'village_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy']

    # initialize consistency variables
    state_consistency = -1
    county_consistency = -1
    county_city_match = []
    address_consistency = -1


    
    # check consistency with geopy data
    if row['coord_presence']:
        state_consistency, county_consistency, county_city_match, address_consistency = check_consistency_geopy(row)

    if ((state_consistency==1 and (county_consistency==1 or (county_consistency==-1 and address_consistency!=0))) 
        or
        (county_consistency==1 and address_consistency==1)):
        # set geopy data
        clean_geo_data_row.loc[['state']] = row['state_geopy']
        clean_geo_data_row.loc[['county']] = row['county_geopy']

        if county_city_match == 'county_geopy':
            if row['city_geopy'] is not None:
                clean_geo_data_row.loc[['city']] = row['city_geopy']
            elif row['town_geopy'] is not None:
                clean_geo_data_row.loc[['city']] = row['town_geopy']
            else:
                clean_geo_data_row.loc[['city']] = row['village_geopy']
        else:
            clean_geo_data_row.loc[['city']] = row[county_city_match]

        clean_geo_data_row.loc[['latitude']] = row['latitude']
        clean_geo_data_row.loc[['longitude']] = row['longitude'] 

        #TODO: ADD IMPORTACNE AND BLA BLA
    elif (state_consistency==1 and county_consistency==-1 and address_consistency==0 and np.isnan(row['city_or_county'])):
        clean_geo_data_row.loc[['state']] = row['state_geopy']

    else:
        state, county = check_consistency_additional_data(row['state'], row['city_or_county'], additional_data)

        clean_geo_data_row.loc[['state']] = state
        clean_geo_data_row.loc[['county']] = county

    return clean_geo_data_row

'''
    elif (state_consistency + county_consistency + town_consistency) >= 2:
        # set geopy data
        clean_geo_data_row.loc[['state']] = row['state_geopy']
        clean_geo_data_row.loc[['county']] = row['county_geopy']
        clean_geo_data_row.loc[['city']] = row['town_geopy']
        clean_geo_data_row.loc[['road']] = row['road_geopy']
        clean_geo_data_row.loc[['latitude']] = row['latitude']
        clean_geo_data_row.loc[['longitude']] = row['longitude']
    else:
        # check consistency with data from additional_data
        state, county = check_consistency_additional_data(row['state'], row['city_or_county'], additional_data)
        if state_consistency:
            clean_geo_data_row.loc[['state']] = row['state_geopy'] # assign geopy data
        elif state is not None:
            clean_geo_data_row.loc[['state']] = state # assign additional data
        if county_consistency:
            clean_geo_data_row.loc[['county']] = row['county_geopy'] # assign geopy data
        elif county is not None:
            clean_geo_data_row.loc[['county']] = county # assign additional data
    
    '''

    