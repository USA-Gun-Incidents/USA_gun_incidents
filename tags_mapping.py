# TODO: aggiungere nel notebook Task_1.ipynb

from enum import Enum
import pandas as pd

class IncidentTag(Enum):
    firearm = 1
    shots = 2
    aggression = 3
    suicide = 4
    injuries = 5
    death = 6
    road = 7
    illegal_holding = 8
    house = 9
    school = 10
    children = 11
    drugs = 12
    officers = 13
    organized = 14
    social_reasons = 15
    defensive = 16
    workplace = 17

# TODO: riempire
tags_map = {
    'Accidental Shooting': [IncidentTag.firearm.name, IncidentTag.shots.name]
    # ...
}

def set_tags(row):
    for tag in tags_map[row['incident_characteristics1']]:
        row[tag] = True
    for tag in tags_map[row['incident_characteristics2']]:
        row[tag] = True
    return row

data = pd.read_csv('./data/incidents.csv')

for tag in IncidentTag:
    data[tag.name] = False

data = data.apply(set_tags, axis=1)

# TODO: controllo consistenza?