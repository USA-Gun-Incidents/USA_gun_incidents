import re

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
def clean_data_city_county(data):
    data = delete_space(data)
    data = lower_case(data)
    data = delete_numbers(data)
    if check_parenthesis(data):
        data1, data2 = split_where_parenthesis(data)
        data1 = delete_punctuation(data1)
        data2 = delete_punctuation(data2)
        return data1, data2
    else:
        data = delete_punctuation(data)
        return data, None
    
def clean_data_state(data):
    data = lower_case(data)
    data = delete_numbers(data)
    data = delete_punctuation(data)
    return data


# check if two strings are the same
def check_typo(data1, data2):
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


# check consistency between two addresses
def check_address_consistency(row):
    state_consistency = False
    county_consistency = False
    city_consistency = False

    # our data
    state_dataset = row['state']
    city_or_county_dataset = row['city_or_county']

    # GeoPy data
    state_geopy = clean_data_state(row['state_geopy'])
    county_geopy = clean_data_state(row['county_geopy'])
    city_geopy = clean_data_state(row['city_geopy'])

    # clean data
    state = clean_data_state(state_dataset)
    city1, city2 = clean_data_city_county(city_or_county_dataset)
    
    # check state
    if state == state_geopy:
        state_consistency = True
    else:
        # check typo
        state_consistency = check_typo(state, state_geopy)
    
    # check country
    if city1 == county_geopy or city2 == county_geopy:
        county_consistency = True
    else:
        county_consistency = (check_typo(city1, county_geopy) or check_typo(city2, county_geopy))

    # check city  
    if city1 == city_geopy or city2 == city_geopy:
        city_consistency = True
    else:
        city_consistency = (check_typo(city1, city_geopy) or check_typo(city2, city_geopy))
    
    return state_consistency, county_consistency, city_consistency

def state_county_consistency(state, county, counties_data):
    state_consistency = False
    state_current = ''
    county_current = ''

    if state in counties_data['State'].unique():
        state_consistency = True
        state_current = state
    else:
        # check typo
        clean_data_state(state)
        for s in counties_data['State'].unique():
            state = clean_data_state(s)
            if check_typo(state, s):
                state_consistency = True
                state_current = s
                break

    if state_consistency:
        if county in counties_data[counties_data['State'] == state]['County'].unique():
            county_consistency = True
        else:
            # check typo
            county1, county2 = clean_data_city_county(county)
            for c in counties_data[counties_data['State'] == state_current]['County'].unique():
                clean_data_state(c)
                if check_typo(county1, c):
                    county_current = c
                    return state_current, county_current
                elif check_typo(county2, c):
                    county_current = c
                    return state_current, county_current
    else:
        county1, county2 = clean_data_city_county(county)
        for c in counties_data['County']:
            clean_data_state(c)
            if check_typo(county1, c):
                county_current = c
                state_current = counties_data[counties_data['County'] == c]['State'].values[0]
                return state_current, county_current
            elif check_typo(county2, c):
                county_current = c
                state_current = counties_data[counties_data['County'] == c]['State'].values[0]
                return state_current, county_current
    
    if state_consistency:
        return state_current, None
    else:
        return None, None
