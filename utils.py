####################### Geographical data cleaning #######################
import pandas as pd
import re
import jellyfish


# epics variables
DL_DISTANCE_TRESHOLD = 4

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
    return data


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
    dis = jellyfish.damerau_levenshtein_distance(s1, s2)
    return dis < DL_DISTANCE_TRESHOLD

# check consistency between two addresses
def check_consistency_geopy(row):

    state_consistency = False
    county_city_consistency = False
    address_consistency = False

    #TODO: null deve essere una cosa positiva nei confronti
    #TODO: 

    if (row['state'] is not None and type(row['state']) == str and type(row['state_geopy']) == str): 
        
        state = clean_data_incidents(row['state']) # our data
        state_geopy = clean_data_geopy(row['state_geopy']) # GeoPy data
  
        for s in state:
            state_consistency = state_consistency or check_typo_jellyfish(s, state_geopy)
    
    if row['city_or_county'] is not None and type(row['city_or_county']) == str:
        incidents_couty_city = clean_data_incidents(row['city_or_county']) #our data

        geopy_couty_city_town_village = []
        geopy_col = ['county_geopy', 'city_geopy', 'town_geopy', 'village_geopy']
        for col in geopy_col:
                geopy_couty_city_town_village.append(clean_data_geopy(row[col]))

        for cc in incidents_couty_city:
            for i, val in enumerate(geopy_couty_city_town_village):
                county_city_consistency = check_typo_jellyfish(cc, val)
                if county_city_consistency:
                    county_city_match = geopy_col[i]
                    if county_city_consistency: break

            if county_city_consistency: break


        '''
        if row['county_geopy'] is not None and type(row['county_geopy']) == str:
            county_geopy = clean_data_state(row['county_geopy']) # GeoPy data
            # check country
            if city1 == county_geopy or city2 == county_geopy:
                county_consistency = True
            else:
                county1_typo = False
                county2_typo = False
                if city1 is not None: county1_typo = check_typo(city1, county_geopy)
                if city2 is not None: county2_typo = check_typo(city2, county_geopy)
                county_consistency = county1_typo or county2_typo 

        if row['city_geopy'] is not None and type(row['city_geopy']) == str: 
            city_geopy = clean_data_state(row['city_geopy']) # GeoPy data
            # check city  
            if city1 == city_geopy or city2 == city_geopy:
                city_consistency = True
            else:
                city1_typo = False
                city2_typo = False
                if city1 is not None: city1_typo = check_typo(city1, city_geopy)
                if city2 is not None: city2_typo = check_typo(city2, city_geopy)
                city_consistency = city1_typo or city2_typo
        elif row['town_geopy'] is not None and type(row['city_geopy']) == str: 
            town_geopy = clean_data_state(row['town_geopy']) # GeoPy data
            # check city  
            if city1 == town_geopy or city2 == town_geopy:
                town_consistency = True
            else:
                city1_typo = False
                city2_typo = False
                if city1 is not None: city1_typo = check_typo(city1, town_geopy)
                if city2 is not None: city2_typo = check_typo(city2, town_geopy)
                town_consistency = city1_typo or city2_typo
        '''
    
    
    return state_consistency, county_city_consistency, county_city_match, address_consistency

def check_consistency_additional_data(state, county, additional_data):
    state_consistency = False
    state_current = ''
    county_current = ''

    if state is not None and type(state) == str:
        if state in additional_data['State or equivalent'].unique():
            state_consistency = True
            state_current = state
        else:
            # check typo
            clean_data_state(state)
            for s in additional_data['State or equivalent'].unique():
                if s is not None and type(s) == str:
                    state = clean_data_state(s)
                    if check_typo(state, s):
                        state_consistency = True
                        state_current = s
                        break

    if county is not None and type(county) == str:
        if state_consistency:
            if county in additional_data[additional_data['State or equivalent'] == state]['County or equivalent'].unique():
                county_consistency = True
            else:
                # check typo
                county1, county2 = clean_data_city_county(county)
                for c in additional_data[additional_data['State or equivalent'] == state_current]['County or equivalent'].unique():
                    if c is not None and type(c) == str:
                        clean_data_state(c)
                        if county1 is not None and type(county1) == str:
                            if check_typo(county1, c):
                                county_current = c
                                return state_current, county_current
                        elif county2 is not None and type(county2) == str:
                            if check_typo(county2, c):
                                    county_current = c
                                    return state_current, county_current

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
        return None, None
    

# main function
def check_geographical_data_consistency(row, additional_data):
    # initialize clean_geo_data
    clean_geo_data_row = pd.Series(index=['state', 'county', 'city', 'display_name', 'latitude', 'longitude'], dtype=str)
    
    #columns=['state', 'city_or_county', 'address', 'latitude', 'longitude', 
    #'display_name', 'village_geopy', 'town_geopy', 'city_geopy', 'county_geopy', 'state_geopy']

    # initialize consistency variables
    state_consistency = False
    county_consistency = False
    city_consistency = False
    town_consistency = False

    # check consistency with geopy data
    if row['coord_presence']:
        state_consistency, county_consistency, city_consistency, town_consistency = check_consistency_geopy(row)

    if (state_consistency + county_consistency + city_consistency) >= 2:
        # set geopy data
        clean_geo_data_row.loc[['state']] = row['state_geopy']
        clean_geo_data_row.loc[['county']] = row['county_geopy']
        clean_geo_data_row.loc[['city']] = row['city_geopy']
        clean_geo_data_row.loc[['road']] = row['road_geopy']
        clean_geo_data_row.loc[['latitude']] = row['latitude']
        clean_geo_data_row.loc[['longitude']] = row['longitude']
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
    
    return clean_geo_data_row