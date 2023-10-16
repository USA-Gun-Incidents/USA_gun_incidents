import pandas as pd
import re
import jellyfish
import numpy as np
import json
import os
import sys

# default variables
DAMERAU_LEVENSHTEIN_DISTANCE_THRESHOLD = 2
SAME_WORDS_ADDRESS_THRESHOLD = 2
FREQUENT_WORDS = ['of', 
    'block', 
    'Street', 
    'and', 
    'Avenue', 
    'St', 
    'Ave', 
    'Road', 
    'Drive', 
    'Rd', 
    'South', 
    'West', 
    'N', 
    'North', 
    'Dr', 
    'S', 
    'W', 
    'E', 
    'East', 
    'Blvd', 
    'Boulevard']

####################### Geographical data cleaning #######################
def lower_case(data):
    """put data in lower case"""
    return data.lower()

def delete_space(data):
    """delete all spaces in data"""
    return data.replace(" ", "")

def split_where_parenthesis(data):
    """return two strings where parenthesis are present in data"""    
    # split data where parenthesis
    data1, data2 = data.split("(")
    # delete close parenthesis and return two strings
    return data1, data2.replace(")", "")

def check_parenthesis(data):
    """check if parenthesis are present in data"""
    if "(" in data:
        return True
    else:
        return False

def delete_punctuation(data):
    """delete all punctuation but parenthesis in data"""
    return re.sub(r'[^\w\s\(\)]', '', data)

def delete_numbers(data):
    """delete all numbers in data"""
    return re.sub(r'[0-9]', '', data)


def clean_data_incidents(data):
    """clean data from incidents dataset"""
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
    """clean data from geopy dataset"""
    if pd.isnull(data): return data

    data = lower_case(data)
    data = delete_numbers(data)
    data = delete_punctuation(data)
    data = delete_space(data)
    
    return data.replace('county', '')

def check_string_typo(string1, string2, sensibility=DAMERAU_LEVENSHTEIN_DISTANCE_THRESHOLD):
    """check if two strings are the same with at most a typo
    according to the Damerau-Levenshtein distance"""
    if pd.isnull(string1): return -1
    if pd.isnull(string2): return -1

    string_distance = jellyfish.damerau_levenshtein_distance(string1, string2)
    return int(string_distance <= sensibility)

def check_address(address1, address2_geopy):
    """check if two addresses are the same with at most a typo"""
    if pd.isnull(address1): return -1
    if pd.isnull(address2_geopy): return -1

    for sep in FREQUENT_WORDS:
        # replace frequent words with a separator
        address1 = address1.replace(sep, '|+|')
    address1_splitted = address1.split('|+|')

    cardinality_address1_in_address2 = 0
    for word in address1_splitted:
        if word in address2_geopy:
            cardinality_address1_in_address2 += 1

    return int(cardinality_address1_in_address2 >= SAME_WORDS_ADDRESS_THRESHOLD)

def check_consistency_geopy(row):
    """check consistency between address in incidents dataset and geopy dataset
    return 0 if not consistent, 1 if consistent, -1 if null values in one of the two addresses"""
    state_consistency = 0
    county_city_consistency = 0
    county_city_match = '-1'
    address_consistency = 0
    
    # set state
    state = clean_data_incidents(row['state']) # our data
    state_geopy = clean_data_geopy(row['state_geopy']) # geopy data

    for s in state:
        dummy = check_string_typo(s, state_geopy)
        if state_consistency == 0:
            state_consistency = dummy
        if dummy == 1:
            state_consistency = dummy

   # set city or county
    incidents_couty_city = clean_data_incidents(row['city_or_county']) #our data

    geopy_couty_city_town_village = []
    geopy_col = ['county_geopy', 'city_geopy', 'town_geopy', 'village_geopy']
    for col in geopy_col:
            geopy_couty_city_town_village.append(clean_data_geopy(row[col]))

    for cc in incidents_couty_city:
        for i, val in enumerate(geopy_couty_city_town_village):
            dummy = check_string_typo(cc, val)

            if county_city_consistency == 0:
                county_city_consistency = dummy
            if dummy == 1:
                county_city_match = geopy_col[i]
                county_city_consistency = dummy

    # set address
    address_consistency = check_address(row['address'], row['address_geopy'])

    return state_consistency, county_city_consistency, county_city_match, address_consistency

def check_consistency_additional_data(state, county, additional_data):
    """check consistency between address in incidents dataset and additional data"""
    state_consistency = False
    state_current = np.nan

    if state in additional_data['State or equivalent'].unique():
        state_consistency = True
        state_current = state
    else: # check typo
        clean_data_geopy(state)
        for s in additional_data['State or equivalent'].unique():
            state = clean_data_geopy(s)
            if check_string_typo(state, s) == 1:
                state_consistency = True
                state_current = s
                break

    if state_consistency:
        if county in additional_data[additional_data['State or equivalent'] == state_current
                                     ]['County or equivalent'].unique():
            return state_current, county
        else: # check typo
            county_list = clean_data_incidents(county)
            for c in additional_data[additional_data['State or equivalent'] == state_current
                                     ]['County or equivalent'].unique():

                c = clean_data_geopy(c)
                for county_incidents in county_list:     
                    if check_string_typo(county_incidents, c) == 1:
                        return state_current, c
    
    return state_current, np.nan
    
def check_geographical_data_consistency(row, additional_data):
    """check consistency between our data, geopty data and additional data
    return consistent data if consistent, else return nan values"""
    # initialize clean_geo_data
    clean_geo_data_row = pd.Series(index=['state', 'county', 'city', 'latitude', 'longitude', 'state_consistency', 
                                'county_consistency', 'address_consistency', 'importance', 'address_type'], dtype=str)
    
    # initialize consistency variables
    state_consistency = -1
    county_consistency = -1
    county_city_match = []
    address_consistency = -1

    # check consistency with geopy data
    if row['coord_presence']: # if geopy data is present
        state_consistency, county_consistency, county_city_match, address_consistency = check_consistency_geopy(row)

    if ((state_consistency==1 and (county_consistency==1 or (county_consistency==-1 and address_consistency!=0))) 
        or (county_consistency==1 and address_consistency==1)):
        # set geopy data
        clean_geo_data_row.loc[['state']] = row['state_geopy']
        clean_geo_data_row.loc[['county']] = row['county_geopy']

        if county_city_match == 'county_geopy' or county_city_match == '-1':
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
        clean_geo_data_row.loc[['importance']] = row['importance_geopy']
        clean_geo_data_row.loc[['address_type']] = row['addresstype_geopy']

    elif (state_consistency==1 and county_consistency==-1 and address_consistency==0 and pd.isnull(row['city_or_county'])):
        # set not null geopy data
        clean_geo_data_row.loc[['state']] = row['state_geopy']
        clean_geo_data_row.loc[['importance']] = row['importance_geopy']
        clean_geo_data_row.loc[['address_type']] = row['addresstype_geopy']

    else: # check consistency with additional data
        state, county = check_consistency_additional_data(row['state'], row['city_or_county'], additional_data)
        clean_geo_data_row.loc[['state']] = state
        clean_geo_data_row.loc[['county']] = county

    clean_geo_data_row.loc[['state_consistency']] = state_consistency
    clean_geo_data_row.loc[['county_consistency']] = county_consistency
    clean_geo_data_row.loc[['address_consistency']] = address_consistency

    return clean_geo_data_row





def create_fips_dict():
    DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\geopy')
    IN = open(os.path.join(DIR, 'FIPS.txt'), 'r')
    
    states_ended = False
    out_dict = {}
    reverse_states = {}
    for line in IN.readlines():
        if line.startswith('#'):
                continue
        elif not states_ended:
            if line.startswith('&&&&&&&&&&'):
                states_ended = True
                continue
            else:
                splitted_line = line.strip().replace('        ', '|').split('|')
                out_dict[splitted_line[1]] = [splitted_line[0],{}]
                reverse_states[splitted_line[0]] = splitted_line[1]
        else:
            splitted_line = line.strip().replace('        ', '|').split('|')
            out_dict[reverse_states[splitted_line[0][:2]]][1][splitted_line[1]] = splitted_line[0]

    json.dump([out_dict, reverse_states], open(os.path.join(DIR, 'FIPS.json'), 'w+'), indent=2)

def get_fips_codes(state:str, county:str, fips_dict:dict, sensibility:int=1):
    
    # very very bad in terms of complexity but ok
    for key in fips_dict.keys():
        if check_string_typo(clean_data_geopy(key), clean_data_geopy(state), sensibility) == 1:
            for key_2 in fips_dict[key][1].keys():
                if check_string_typo(clean_data_geopy(key_2), clean_data_geopy(county), sensibility) == 1:
                    return fips_dict[key][0], fips_dict[key][1][key_2]
            
            return fips_dict[key][0], np.nan
    
    return np.nan, np.nan

def check_address(address1, address2_geopy):
    """check if two addresses are the same with at most a typo"""
    if pd.isnull(address1): return -1
    if pd.isnull(address2_geopy): return -1

    for sep in FREQUENT_WORDS:
        # replace frequent words with a separator
        address1 = address1.replace(sep, '|+|')
    address1_splitted = address1.split('|+|')

    cardinality_address1_in_address2 = 0
    for word in address1_splitted:
        if word in address2_geopy:
            cardinality_address1_in_address2 += 1

    return int(cardinality_address1_in_address2 >= SAME_WORDS_ADDRESS_THRESHOLD)

def check_consistency_additional_data_2(state, county, fips_dict):
    """check consistency between address in incidents dataset and additional data"""
    county_list = clean_data_incidents(county)

    state_current, county_current = get_fips_codes(state,county_list[0], fips_dict)

    if pd.isna(county_current) and len(county_list) > 1:
        a, b = get_fips_codes(state,county_list[1], fips_dict)
        if not pd.isna(a):
            state_current = a
            county_current = b

    return state_current, county_current
    
def check_geographical_data_consistency_2(row, fips_dict):
    """check consistency between our data, geopty data and additional data
    return consistent data if consistent, else return nan values"""
    # initialize clean_geo_data
    clean_geo_data_row = pd.Series(index=['state', 'county', 'city', 'latitude', 'longitude', 'state_consistency', 
                                'county_consistency', 'address_consistency', 'importance', 'address_type'], dtype=str)
    
    #is a list of 2 dict with the FIPS codes of the states and then the counties


    # initialize consistency variables
    state_consistency = -1
    county_consistency = -1
    county_city_match = []
    address_consistency = -1

    # check consistency with geopy data
    if row['coord_presence']: # if geopy data is present
        state_consistency, county_consistency, county_city_match, address_consistency = check_consistency_geopy(row)

    if ((state_consistency==1 and (county_consistency==1 or (county_consistency==-1 and address_consistency!=0))) 
        or (county_consistency==1 and address_consistency==1)):
        # set geopy data
        
        # using the FIPS codes
        a,b = get_fips_codes(row['state_geopy'], row['county_geopy'], fips_dict)
        if pd.isna(a) or (pd.isna(b) and not pd.isna(row['county_geopy'])):
            print('aiaiai','-', row['state_geopy'],'-', row['county_geopy'],'-', a,'-', b)

        clean_geo_data_row.loc[['state']] = a
        clean_geo_data_row.loc[['county']] = b

        if county_city_match == 'county_geopy' or county_city_match == '-1':
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
        clean_geo_data_row.loc[['importance']] = row['importance_geopy']
        clean_geo_data_row.loc[['address_type']] = row['addresstype_geopy']

    elif (state_consistency==1 and county_consistency==-1 and address_consistency==0 and pd.isnull(row['city_or_county'])):
        
        a,b = get_fips_codes(row['state_geopy'], row['county_geopy'], fips_dict)

        # set not null geopy data
        clean_geo_data_row.loc[['state']] = a
        clean_geo_data_row.loc[['importance']] = row['importance_geopy']
        clean_geo_data_row.loc[['address_type']] = row['addresstype_geopy']

    else: # check consistency with additional data
        state, county = check_consistency_additional_data_2(row['state'], row['city_or_county'], fips_dict)
        clean_geo_data_row.loc[['state']] = state
        clean_geo_data_row.loc[['county']] = county

    clean_geo_data_row.loc[['state_consistency']] = state_consistency
    clean_geo_data_row.loc[['county_consistency']] = county_consistency
    clean_geo_data_row.loc[['address_consistency']] = address_consistency

    return clean_geo_data_row

####################### Age-gender and categorical data cleaning #######################
