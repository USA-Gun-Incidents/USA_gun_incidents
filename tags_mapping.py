# TODO: aggiungere nel notebook Task_1.ipynb

from enum import Enum
import pandas as pd

class IncidentTag(Enum):
    firearm = 1
    air_gun = 2
    shots = 3
    aggression = 4
    suicide = 5
    injuries = 6
    death = 7
    road = 8
    illegal_holding = 9
    house = 10
    school = 11
    children = 12
    drugs = 13
    officers = 14
    organized = 15
    social_reasons = 16
    defensive = 17
    workplace = 18
    abduction = 19
    unintentional = 20

tags_map = {
    # when a gun was not used?
    'ATF/LE Confiscation/Raid/Arrest': [IncidentTag.illegal_holding.name, IncidentTag.officers.name],
    'Accidental Shooting': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.unintentional.name],
    'Accidental Shooting - Death': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.death.name, IncidentTag.unintentional.name],
    'Accidental Shooting - Injury': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.injuries.name, IncidentTag.unintentional.name],
    'Accidental Shooting at a Business': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.workplace.name, IncidentTag.unintentional.name],
    'Accidental/Negligent Discharge': [IncidentTag.firearm.name, IncidentTag.unintentional.name],
    'Animal shot/killed': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.death.name],
    'Armed robbery with injury/death and/or evidence of DGU found': [IncidentTag.firearm.name, IncidentTag.aggression.name, IncidentTag.injuries.name, IncidentTag.illegal_holding.name],
    'Assault weapon (AR-15, AK-47, and ALL variants defined by law enforcement)': [IncidentTag.firearm.name],
    'Attempted Murder/Suicide (one variable unsuccessful)': [],
    'BB/Pellet/Replica gun': [IncidentTag.air_gun.name],
    'Bar/club incident - in or around establishment': [],
    'Brandishing/flourishing/open carry/lost/found': [IncidentTag.firearm.name],
    'Car-jacking': [IncidentTag.aggression.name, IncidentTag.road.name, IncidentTag.illegal_holding.name],
    'Child Involved Incident': [IncidentTag.children.name],
    'Child picked up & fired gun': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.children.name],
    'Child with gun - no shots fired': [IncidentTag.firearm.name, IncidentTag.children.name],
    'Cleaning gun': [IncidentTag.firearm.name, IncidentTag.shots.name],
    'Concealed Carry License - Perpetrator': [IncidentTag.firearm.name],
    'Concealed Carry License - Victim': [IncidentTag.firearm.name],
    'Criminal act with stolen gun': [IncidentTag.firearm.name, IncidentTag.illegal_holding.name],
    'Defensive Use': [IncidentTag.firearm.name, IncidentTag.defensive.name],
    'Defensive Use - Crime occurs, victim shoots subject/suspect/perpetrator': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.defensive.name],
    'Defensive Use - Shots fired, no injury/death': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.defensive.name],
    'Defensive Use - Victim stops crime': [IncidentTag.firearm.name, IncidentTag.defensive.name],
    'Defensive Use - WITHOUT a gun': [IncidentTag.defensive.name],
    'Domestic Violence': [IncidentTag.house.name, IncidentTag.aggression.name],
    'Drive-by (car to street, car to car)': [IncidentTag.firearm.name, IncidentTag.aggression.name, IncidentTag.road.name],
    'Drug involvement': [IncidentTag.drugs.name],
    'Gang involvement': [IncidentTag.organized.name],
    'Ghost gun': [IncidentTag.firearm.name],
    'Gun at school, no death/injury - elementary/secondary school': [IncidentTag.firearm.name, IncidentTag.school.name, IncidentTag.children.name],
    'Gun at school, no death/injury - university/college': [IncidentTag.firearm.name, IncidentTag.school.name],
    'Gun buy back action': [IncidentTag.firearm.name, IncidentTag.illegal_holding.name],
    'Gun range/gun shop/gun show shooting': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name],
    'Gun shop robbery or burglary': [IncidentTag.firearm.name, IncidentTag.illegal_holding.name],
    'Gun(s) stolen from owner': [IncidentTag.firearm.name, IncidentTag.illegal_holding.name],
    'Guns stolen from law enforcement': [IncidentTag.firearm.name, IncidentTag.illegal_holding.name, IncidentTag.officers.name],
    'Hate crime': [IncidentTag.social_reasons.name],
    'Home Invasion': [IncidentTag.house.name],
    'Home Invasion - No death or injury': [IncidentTag.house.name],
    'Home Invasion - Resident injured': [IncidentTag.aggression.name, IncidentTag.injuries.name, IncidentTag.house.name],
    'Home Invasion - Resident killed': [IncidentTag.aggression.name, IncidentTag.death.name, IncidentTag.house.name],
    'Home Invasion - subject/suspect/perpetrator injured': [IncidentTag.injuries.name, IncidentTag.house.name],
    'Home Invasion - subject/suspect/perpetrator killed': [IncidentTag.death.name, IncidentTag.house.name],
    'House party': [IncidentTag.house.name],
    'Hunting accident': [IncidentTag.firearm.name, IncidentTag.unintentional.name],
    'Implied Weapon': [IncidentTag.firearm.name],
    'Institution/Group/Business': [IncidentTag.workplace.name],
    'Kidnapping/abductions/hostage': [IncidentTag.aggression.name, IncidentTag.abduction.name],
    'LOCKDOWN/ALERT ONLY: No GV Incident Occurred Onsite': [],
    'Mass Murder (4+ deceased victims excluding the subject/suspect/perpetrator , one location)': [IncidentTag.aggression.name, IncidentTag.death.name],
    'Mass Shooting (4+ victims injured or killed excluding the subject/suspect/perpetrator, one location)': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name, IncidentTag.injuries.name],
    'Murder/Suicide': [IncidentTag.death.name],
    'Non-Aggression Incident': [],
    'Non-Shooting Incident': [IncidentTag.firearm.name],
    'Officer Involved Incident': [IncidentTag.officers.name],
    'Officer Involved Incident - Weapon involved but no shots fired': [IncidentTag.firearm.name, IncidentTag.officers.name],
    'Officer Involved Shooting - Accidental discharge - no injury required': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.officers.name, IncidentTag.unintentional.name],
    'Officer Involved Shooting - Officer killed': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name, IncidentTag.death.name, IncidentTag.officers.name],
    'Officer Involved Shooting - Officer shot': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name, IncidentTag.officers.name],
    'Officer Involved Shooting - Shots fired, no injury': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name, IncidentTag.officers.name],
    'Officer Involved Shooting - subject/suspect/perpetrator killed': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.death.name, IncidentTag.officers.name],
    'Officer Involved Shooting - subject/suspect/perpetrator shot': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.officers.name],
    'Officer Involved Shooting - subject/suspect/perpetrator suicide at standoff': [IncidentTag.suicide.name, IncidentTag.officers.name],
    'Officer Involved Shooting - subject/suspect/perpetrator surrender at standoff': [IncidentTag.officers.name],
    'Officer Involved Shooting - subject/suspect/perpetrator unarmed': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.officers.name],
    'Pistol-whipping': [IncidentTag.firearm.name, IncidentTag.aggression.name],
    'Police Targeted': [IncidentTag.officers.name],
    'Political Violence': [IncidentTag.aggression.name, IncidentTag.social_reasons.name],
    'Possession (gun(s) found during commission of other crimes)': [IncidentTag.firearm.name],
    'Possession of gun by felon or prohibited person': [IncidentTag.firearm.name, IncidentTag.illegal_holding.name],
    'Road rage': [IncidentTag.road.name],
    'School Incident': [IncidentTag.school.name],
    'School Shooting - elementary/secondary school': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name, IncidentTag.school.name, IncidentTag.children.name],
    'Sex crime involving firearm': [IncidentTag.firearm.name, IncidentTag.aggression.name],
    'Shootout (where VENN diagram of shooters and victims overlap)': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name],
    'Shot - Dead (murder, accidental, suicide)': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.death.name],
    'Shot - Wounded/Injured': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name, IncidentTag.injuries.name],
    'ShotSpotter': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name],
    'Shots Fired - No Injuries': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name],
    'Shots fired, no action (reported, no evidence found)': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name],
    'Spree Shooting (multiple victims, multiple locations)': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name, IncidentTag.death.name],
    'Stolen/Illegally owned gun{s} recovered during arrest/warrant': [IncidentTag.firearm.name, IncidentTag.illegal_holding.name],
    'Suicide - Attempt': [IncidentTag.suicide.name],
    'Suicide^': [IncidentTag.suicide.name, IncidentTag.death.name],
    'TSA Action': [IncidentTag.officers.name],
    'Terrorism Involvement': [IncidentTag.aggression.name, IncidentTag.organized.name],
    'Under the influence of alcohol or drugs (only applies to the subject/suspect/perpetrator )': [IncidentTag.drugs.name],
    'Unlawful purchase/sale': [IncidentTag.illegal_holding.name],
    'Workplace shooting (disgruntled employee)': [IncidentTag.firearm.name, IncidentTag.shots.name, IncidentTag.aggression.name, IncidentTag.workplace.name]
}

def set_tags(row):
    if pd.notnull(row['incident_characteristics1']):
        for tag in tags_map[row['incident_characteristics1']]:
            row[tag] = True
    if pd.notnull(row['incident_characteristics2']):
        for tag in tags_map[row['incident_characteristics2']]:
            row[tag] = True
    return row


def build_tagged_dataframe(base_folder): #build the dataframe with tags
    data = pd.read_csv(base_folder + 'incidents.csv')

    for tag in IncidentTag:
        data[tag.name] = False

    data = data.apply(set_tags, axis=1)

    data.to_csv(base_folder + 'incidents_tagged.csv')

    return data