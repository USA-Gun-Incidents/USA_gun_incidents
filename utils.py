import pandas as pd
import re
import jellyfish
import numpy as np

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

def check_string_typo(string1, string2):
    """check if two strings are the same with at most a typo
    according to the Damerau-Levenshtein distance"""
    if pd.isnull(string1): return -1
    if pd.isnull(string2): return -1

    string_distance = jellyfish.damerau_levenshtein_distance(string1, string2)
    return int(string_distance <= DAMERAU_LEVENSHTEIN_DISTANCE_THRESHOLD)

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

####################### Age-gender and categorical data cleaning #######################
def convert_age_to_int(data):
    """return age as a int if it is numeric and between 0 and 100
    else return nan"""
    data = convert_data_to_int(data)
    if data not in [np.nan]:
        return exclude_inconsistent_age(data)
    else: return np.nan

def exclude_inconsistent_age(data):
    """return nan if age is negative or greater than 100"""
    if (data >= 0 and data <= 100):
        return data
    else: return np.nan

def convert_group_cardinality_to_int(data):
    """return group cardinality as a int if it is numeric and greater than 0"""
    data = convert_data_to_int(data)
    if data not in [np.nan]:
        return exclude_negative_value(data)
    else: return np.nan

def exclude_negative_value(data):
    """return nan if group cardinality is negative"""
    if data < 0:
        return np.nan
    else: return data

def convert_data_to_int(data):
    """convert data to int if it is numeric"""
    try:
        data = int(float(data))
        return data
    except:
        return np.nan

def age_groups_consistency(min_age, max_age, avg_age, n_child, n_teen, n_adult):
    """check consistency between age groups attributes"""
    if min_age not in [np.nan]:
        if min_age > max_age or min_age > avg_age:
            return False
        if min_age < 12:
            if n_child <= 0:
                return False
        elif 12 <= min_age < 18:
            if n_child > 0 and n_teen <= 0:
                return False
        else:
            if n_child > 0 or n_teen > 0 or n_adult <= 0:
                return False

    if max_age not in [np.nan]:
        if max_age < 12:
            if n_child <= 0 or n_teen > 0 or n_adult > 0:
                return False
        elif 12 <= max_age < 18:
            if n_teen <= 0 or n_adult > 0:
                return False
        else:
            if n_adult <= 0:
                return False

    if n_child not in [np.nan] and n_teen not in [np.nan] and n_adult not in [np.nan]:
        if n_child + n_teen + n_adult <= 0:
            return False

    return True

def gender_consistecy(n_males, n_females, n_participants):
    """check consistency between number of participants divided by gender"""
    if n_males not in [np.nan] and n_females not in [np.nan] and n_participants not in [np.nan]:
        return (n_males + n_females == n_participants)
    return np.nan

def category_consistency(n_killed, n_injured, n_arrested, n_unharmed, n_participants):
    """check consistency between number of participants and number of killed, injured, arrested and unharmed"""
    if (n_killed not in [np.nan] and n_injured not in [np.nan] and n_arrested not in [np.nan] and n_unharmed not in [np.nan] and 
        n_participants not in [np.nan]):
        return ((n_killed + n_injured <= n_participants) and
            (n_arrested <= n_participants) and
            (n_unharmed <= n_participants))
    return np.nan

def ages_groups_participant1(participant_gender1):
    """Binarize participant1 age groups attribute"""
    if participant_gender1 == 'Child 0-11':
        return [True, False, False] #'Child 0-11'
    elif participant_gender1 == 'Teen 12-17':
        return [False, True, False] #'Teen 12-17'
    elif participant_gender1 == 'All Ages':
        return [False, False, True] #'Adult 18+'
    else: 
        return [np.nan, np.nan, np.nan]
    
def gender_participant1(participant_gender1):
    """Binarize participant1 gender attribute"""
    if participant_gender1 == 'Male':
        return [True, False]
    elif participant_gender1 == 'Female':
        return [False, True]
    else:
        return [np.nan, np.nan]

def participant1_age_data_consistency(participant_age1, participant1_child, participant1_teen, participant1_adult):
    """check consistency between participant1 age groups attributes"""
    if participant_age1 not in [np.nan]:
        if participant_age1 < 12:
            if participant1_child is not True:
                return False
        elif 12 <= participant_age1 < 18:
            if participant1_teen is not True:
                return False
        else:
            if participant1_adult is not True:
                return False
    return True

def participant1_age_consistency_wrt_all_data(participant_age1, min_age, max_age):
    """check consistency between participant1 age and age groups attributes"""
    if participant_age1 not in [np.nan] and min_age not in [np.nan] and max_age not in [np.nan]:
        return (participant_age1 >= min_age and participant_age1 <= max_age)
    return np.nan

def participant1_age_range_consistency_wrt_all_data(participant1_child, participant1_teen, participant1_adult,
    n_participants_child, n_participants_teen, n_participants_adult):
    """check consistency between participant1 age groups and age groups attributes"""
    if participant1_child is True:
        return (n_participants_child > 0)
    elif participant1_teen is True:
        return (n_participants_teen > 0)
    elif participant1_adult is True:
        return (n_participants_adult > 0) 
    return np.nan

def participant1_gender_consistency_wrt_all_data(participant1_male, participant1_female, n_males, n_female):
    """check consistency between participant1 gender groups and gender groups attributes"""
    if participant1_male is True:
        return (n_males > 0)
    elif participant1_female is True:
        return (n_female > 0)
    return np.nan

def check_age_gender_data_consistency(row):
    """clean data and check consistency between age, gender and cardinality of groups attributes
    return clean as integer or nan and
    consistency as boolean (True if there is consistence, False else) or nan if there are not values to check"""
    # initialize clean_data_row
    clean_data_row = pd.Series(index=['participant_age1', 
        'participant1_child', 'participant1_teen', 'participant1_adult',
        'participant1_male', 'participant1_female',
        'min_age_participants', 'avg_age_participants', 'max_age_participants', 
        'n_participants_child', 'n_participants_teen', 'n_participants_adult', 
        'n_males', 'n_females',
        'n_killed', 'n_injured', 'n_arrested', 'n_unharmed', 
        'n_participants',
        'consistency_age', 'consistency_n_participant', 'consistency_gender', 
        'consistency_participant1', 'consistency_participants1_wrt_n_participants',
        'nan_values'], dtype=int)
    
    # convert ot integer participants age range attributes
    clean_data_row.loc[['min_age_participants']] = convert_age_to_int(row['min_age_participants'])
    clean_data_row.loc[['max_age_participants']] = convert_age_to_int(row['max_age_participants'])
    clean_data_row.loc[['avg_age_participants']] = convert_age_to_int(row['avg_age_participants'])
    clean_data_row.loc[['n_participants_child']] = convert_group_cardinality_to_int(row['n_participants_child'])
    clean_data_row.loc[['n_participants_teen']] = convert_group_cardinality_to_int(row['n_participants_teen'])
    clean_data_row.loc[['n_participants_adult']] = convert_group_cardinality_to_int(row['n_participants_adult'])

    # check consistency participants age range attributes
    clean_data_row.loc[['consistency_age']] = age_groups_consistency(
        clean_data_row['min_age_participants'], clean_data_row['max_age_participants'], 
        clean_data_row['avg_age_participants'], clean_data_row['n_participants_child'], 
        clean_data_row['n_participants_teen'], clean_data_row['n_participants_adult'])
    
    # convert to integer attributes releted to number of participants
    clean_data_row.loc[['n_males']] = convert_group_cardinality_to_int(row['n_males'])
    clean_data_row.loc[['n_females']] = convert_group_cardinality_to_int(row['n_females'])
    clean_data_row.loc[['n_killed']] = convert_group_cardinality_to_int(row['n_killed'])
    clean_data_row.loc[['n_injured']] = convert_group_cardinality_to_int(row['n_injured'])
    clean_data_row.loc[['n_arrested']] = convert_group_cardinality_to_int(row['n_arrested'])
    clean_data_row.loc[['n_unharmed']] = convert_group_cardinality_to_int(row['n_unharmed'])
    clean_data_row.loc[['n_participants']] = convert_group_cardinality_to_int(row['n_participants'])

    # check genderes consistency w.r.t. number of participants
    clean_data_row.loc[['consistency_gender']] = gender_consistecy(clean_data_row['n_males'],
        clean_data_row['n_females'], clean_data_row['n_participants'])
    
    # check consistency between number of participants and number of killed, injured, arrested and unharmed
    clean_data_row.loc[['consistency_n_participant']] = category_consistency(clean_data_row['n_killed'], 
        clean_data_row['n_injured'], clean_data_row['n_arrested'], clean_data_row['n_unharmed'], 
        clean_data_row['n_participants'])

    # converto to integer participants1 attributes
    clean_data_row.loc[['participant_age1']] = convert_age_to_int(row['participant_age1'])
    clean_data_row.loc[['participant1_child', 'participant1_teen', 'participant1_adult']] = ages_groups_participant1(
        clean_data_row['participant_age1'])
    clean_data_row.loc[['participant1_male', 'participant1_female']] = gender_participant1(row['participant_gender1'])

    # check consistency between participants1 attributes
    clean_data_row.loc[['consistency_participant1']] = participant1_age_data_consistency(clean_data_row['participant_age1'],
        clean_data_row['participant1_child'], clean_data_row['participant1_teen'], clean_data_row['participant1_adult'])
    
    # check consistency between participants1 attributes and number of participants
    clean_data_row.loc[['consistency_participants1_wrt_n_participants']] = (
        participant1_age_data_consistency(clean_data_row['participant_age1'], clean_data_row['participant1_child'],
            clean_data_row['participant1_teen'], clean_data_row['participant1_adult']) and
        participant1_age_consistency_wrt_all_data(clean_data_row['participant_age1'], 
            clean_data_row['min_age_participants'], clean_data_row['max_age_participants']) and
        participant1_age_range_consistency_wrt_all_data(clean_data_row['participant1_child'],
            clean_data_row['participant1_teen'], clean_data_row['participant1_adult'],
            clean_data_row['n_participants_child'], clean_data_row['n_participants_teen'], 
            clean_data_row['n_participants_adult']) and
        participant1_gender_consistency_wrt_all_data(clean_data_row['participant1_male'], 
            clean_data_row['participant1_female'], clean_data_row['n_males'], clean_data_row['n_females']))

    # check if nan values are present
    clean_data_row.loc[['nan_values']] = True if row.isnull().sum() > 0 else False

    return clean_data_row