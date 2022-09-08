"""
Get ODiN data and transform it to survey_*.csv files for activitysim.

Created on Wed Sep 15 12:40:13 2021

@author: asharib

Modified on Tue Apr 19 17:22 2022

@editor: David Matheus
"""

import config as cfg
import functions as fn
import pandas as pd
import sql_queries as q
import time
from pathlib import Path
import random
import os
import sys
import numpy as np
from activitysim.core.util import reindex

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

sys.path.append(parent_directory)

from constants import EURO_2_USDOLLAR

DELFT_CASE = False
SAMPLE = True
delft_mrdh_zones = [i + 1 for i in range(239)]

case = 'base_case'

output_directory = Path(parent_directory, f'data_{case}/survey_data')
configs_dir = Path(parent_directory, 'configs')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if DELFT_CASE:
    case = case + '_delft'

# File extracted from CBS that maps income deciles to median income levels
income_dist_file = 'hhbesteedbaarinkomen_2019.csv'



# Start timer
tic = time.time()


# %% Read data
# Get wogem lookup table
# wogem_pc4 = fn.find_pc4_wogem()

# Get ODiN data
if cfg.debug:
    cfg.logger.warning("Debug mode")
    if cfg.odin_debug_file.exists():
        cfg.logger.warning("Using cached data!")
        odin = pd.read_pickle(cfg.odin_debug_file)
    else:
        cfg.logger.info("Database")
        odin = pd.read_sql(q.odin_debug, cfg.odin_engine)
        odin.to_pickle(cfg.odin_debug_file)
else:
    if cfg.odin_file.exists():
        cfg.logger.warning("Using cached data!")
        odin = pd.read_pickle(cfg.odin_file)
    else:
        cfg.logger.info("Using database")
        odin = pd.read_sql(q.odin, cfg.odin_engine)
        odin.to_pickle(cfg.odin_file)


# Some PC4 zones from 2019 did not exist in 2017, when the data mapping PC4 to MRDH is from, so we must replace those
# with their  2017 counterparts
replace_pc4 = {
    # New_pc4: old_pc4,
    5156: 5256,
    5255: 5256,
}
for pc4 in ['home_pc4', 'vertpc', 'aankpc']:
    odin[pc4].replace(replace_pc4, inplace=True)

# %% Translate PC4 zones to MRDH,
# Taking into account the purpose
odin[
    [
        "home_pc4",
        "work_pc4",
        "school_pc4",
        "orig_mrdh",
        "dest_mrdh",
        "home_mrdh",
        "work_mrdh",
        "school_mrdh",
    ]
] = fn.translate_zones(odin)

# The translated zones are mapped with base 0, they need to be mapped with base 1 because asim will offset it back
odin['offset'] = 1
for mrdh in ["orig_mrdh", "dest_mrdh", "home_mrdh", "work_mrdh", "school_mrdh"]:
    odin[mrdh] = odin[mrdh] + odin['offset']

# %% Convert types to Int64 (nullable integer type)
odin = odin.astype("Int64")


# Drop respondents if no MRDH zone has been mapped to either origin or destination PC4, often because the PC4 was not
# found on the clippings file
for od in ["orig_mrdh", "dest_mrdh"]:
    opids_for_removal = odin[(odin[od].isna()) & (~odin['verplid'].isna())].opid.unique()
    if list(opids_for_removal):
        cfg.logger.warning(f'Removing {len(opids_for_removal)} respondents, missing {od} (opid: {list(opids_for_removal)})')
    odin = odin.loc[~odin['opid'].isin(opids_for_removal)]

# # FIXME keep survey ids
# # Redefine opid to start from 1
# opid_keys = list(odin.opid.unique())
# # opid_values = [i for i in range(len(opid_keys))]
# opid_values = [i + 1 for i in range(len(opid_keys))]
# odin.opid = odin.opid.map(dict(zip(opid_keys, opid_values)))



# Sample ODiN
if SAMPLE:
    fraction = 0.1  # 10%
    cfg.logger.warning(f'Sampling {fraction*100}% of data')
    seed = 1
    opid_to_keep = pd.Series(odin.opid.unique()).sample(frac=fraction, random_state=seed).tolist()
    odin = odin.loc[odin.opid.isin(opid_to_keep)].reset_index(drop=True)

if DELFT_CASE:
    # Drop any respondents that traveled outside of the Delft MRDH zones
    cfg.logger.warning('Keeping only respondents in Delft area')
    for od in ["orig_mrdh", "dest_mrdh"]:
        opids_for_removal = odin.loc[~odin[od].isin(delft_mrdh_zones)].opid.unique()
        odin = odin.loc[~odin['opid'].isin(opids_for_removal)]

# FIXME keep survey ids
# Redefine opid to start from 1
opid_keys = list(odin.opid.unique())
# opid_values = [i for i in range(len(opid_keys))]
opid_values = [i + 1 for i in range(len(opid_keys))]
odin.opid = odin.opid.map(dict(zip(opid_keys, opid_values)))

# %% Generate tour_id, outbound and parent_tour_id columns
fn.generate_tour_id(odin)
fn.generate_outbound_marker(odin)
fn.generate_parent_tour_id(odin)

# Check if there are tours with no inbound trips (incomplete tours) and remove those respondents
# all_tours = set(odin.tour_id.unique())
tours_no_home = odin.dropna(subset=['outbound'])
all_tours = set(tours_no_home.tour_id.unique())
#complete_tours = tours_no_home.loc(~tours_no_home['outbound'])
complete_tours = set(tours_no_home[~tours_no_home.outbound].tour_id.unique())
incomplete_tours = list(all_tours - complete_tours)
incomplete_tours.sort()
opids_for_removal = odin[odin.tour_id.isin(incomplete_tours)].opid.unique()
odin = odin.loc[~odin['opid'].isin(opids_for_removal)]

# %% Households
hh = odin[
    ["opid", "home_mrdh", "hhgestinkg", "hhpers", "hhsam", "hhlft4", "betwerk", "hhauto", "geslacht", 'urbanized',
     "leeftijd"]
].copy()

hh.drop_duplicates(inplace=True)

# Number of workers in household based on adults in household and employment status of respondent
hh["unemployed"] = np.where((hh['leeftijd'] >= 18) & (hh['betwerk'] == 0), 1, 0)
# hh["num_workers"] = 1
hh["workers"] = hh['hhlft4'] - hh['unemployed']

# Remap values
# Median disposable income values for each decile are extracted from CBS
income_dict = {}
for i in range(10):
    income_dict[i+1] = pd.read_csv(Path(cfg.sourcedata, income_dist_file))['Mediaan inkomen'][i]
hh.hhgestinkg = hh.hhgestinkg.map(income_dict) * EURO_2_USDOLLAR


# Household composition
# HHT_NONFAMILY_MALE_ALONE: 4
hh.loc[(hh.hhsam == 1) & (hh.geslacht == 1), "temp_hhsam"] = 4
# HHT_NONFAMILY_MALE_NOTALONE: 5
hh.loc[(hh.hhsam.isin([2, 8])) & (hh.geslacht == 1), "temp_hhsam"] = 5
# HHT_NONFAMILY_FEMALE_ALONE: 6
hh.loc[(hh.hhsam == 1) & (hh.geslacht == 2), "temp_hhsam"] = 6
# HHT_NONFAMILY_FEMALE_NOTALONE: 7
hh.loc[(hh.hhsam.isin([2, 8])) & (hh.geslacht == 2), "temp_hhsam"] = 7
# HHT_FAMILY_MARRIED: 1
hh.loc[hh.hhsam.isin([3, 4, 5]), "temp_hhsam"] = 1
# HHT_FAMILY_MALE: 2
hh.loc[(hh.hhsam.isin([6, 7])) & (hh.geslacht == 1), "temp_hhsam"] = 2
# HHT_FAMILY_FEMALE: 3
hh.loc[(hh.hhsam.isin([6, 7])) & (hh.geslacht == 2), "temp_hhsam"] = 3

# Copy over to actual column
hh.hhsam = hh.temp_hhsam

# Rename columns
hh.rename(
    columns={
        "opid": "household_id",
        "home_mrdh": "home_zone_id",
        "hhgestinkg": "income",
        # "hhpers": "hhsize",
        "hhpers": "PERSONS",
        "hhsam": "HHT",
        "hhauto": "auto_ownership",
    },

    inplace=True,
)
hh.drop(columns=["geslacht", "temp_hhsam", "hhlft4", "betwerk", "leeftijd"], inplace=True)

hh_inferred = hh.copy()

# %% Persons
p = odin[
    [
        "opid",
        "leeftijd",
        "opleiding",
        "urbanized",
        "geslacht",
        "betwerk",
        "ovstkaart",
        "rijbewijs",
        "herkomst",
        "has_car",
        "type_car",
        "has_bike",
        "has_ebike",
        "car_sharing",
        "work_mrdh",
        "school_mrdh",
        "home_mrdh",
    ]
].copy()
p.drop_duplicates(inplace=True)
p["household_id"] = p.opid
p["PNUM"] = 1  # TODO is this correct?

# %%% Types
# PTYPE_FULL: 1
p["ptype"] = 1
p.loc[(p.leeftijd.between(15, 65)) & (p.betwerk == 3), "ptype"] = 1
# PTYPE_PART: 2
p.loc[(p.leeftijd.between(15, 65)) & (p.betwerk == 2), "ptype"] = 2
# PTYPE_UNIVERSITY: 3
p.loc[
    (p.leeftijd.between(15, 65))
    & ~(p.betwerk.isin([2, 3]))
    & (p.ovstkaart.isin([1, 2])),
    "ptype",
] = 3
# PTYPE_NONWORK: 4
p.loc[(p.leeftijd.between(19, 65)) & (p.betwerk == 0) & (p.ovstkaart == 0), "ptype"] = 4
# PTYPE_RETIRED: 5
p.loc[(p.leeftijd > 65), "ptype"] = 5
# PTYPE_DRIVING: 6
p.loc[
    (p.leeftijd.between(15, 18))
    & ~(p.betwerk.isin([2, 3]))
    & ~(p.ovstkaart.isin([1, 2])),
    "ptype",
] = 6
# PTYPE_SCHOOL: 7
p.loc[p.leeftijd.between(6, 14), "ptype"] = 7
# PTYPE_PRESCHOOL: 8
p.loc[p.leeftijd < 6, "ptype"] = 8

# %%% Student
# PSTUDENT_GRADE_OR_HIGH: 1
p.loc[p.leeftijd.between(6, 14), "pstudent"] = 1
p.loc[
    (p.leeftijd.between(15, 18))
    & ~(p.betwerk.isin([2, 3]))
    & ~(p.ovstkaart.isin([1, 2])),
    "pstudent",
] = 1
# PSTUDENT_UNIVERSITY: 2
p.loc[
    (p.leeftijd.between(15, 65))
    & ~(p.betwerk.isin([2, 3]))
    & (p.ovstkaart.isin([1, 2])),
    "pstudent",
] = 2
# PSTUDENT_NOT: 3
p["pstudent"] = 3

# # %%% Employ
# pemploy_dict = {
#     1: 'PEMPLOY_FULL',
#     2: 'PEMPLOY_PART',
#     3: 'PEMPLOY_NOT',
#     4: 'PEMPLOY_CHILD',  # not possible as children < 15 y/o are not asked about employment
# }

p["pemploy"] = p.betwerk.map({0: 3, 1: 2, 2: 2, 3: 1, 5: 3})

# Unknown, so assumed to be false for now
p["free_parking_at_work"] = False

# Change people who do not qualify for student ov to not having ov
p.loc[p.ovstkaart == 4, "ovstkaart"] = 0

# Assume children above 12 have finished primary school, and the rest have not
p.loc[((p.opleiding == 7) & (p.leeftijd < 12)), "opleiding"] = 0
p.loc[((p.opleiding == 7) & (p.leeftijd >= 12)), "opleiding"] = 0

# # Assumptions for possession of maas_subscription
# # At person level:
# p["maas_subscription"] = p["car_sharing"] # If op has shared car, then op has MaaS subscription
#
# def maas_from_no_possession(frequency_or_license_column, positive_value, vehicle_in_hh_column):
#     if frequency_or_license_column in list(positive_value) and vehicle_in_hh_column == 0:
#         return 1
#     else:
#         pass
# # Generalizing from a trip to person level
p["maas_subscription"] = 0  # FIXME possibly change to another method to derive this
# FIXME also make changes to have car sharing in the model (can_use_drt)


# %%% Clean-up
# Rename columns
p.rename(
    columns={
        "opid": "person_id",
        "leeftijd": "age",
        "geslacht": "sex",
        "rijbewijs": "driving_license",
        "herkomst": "roots_person",
        "work_mrdh": "workplace_zone_id",
        "school_mrdh": "school_zone_id",
        "ovstkaart": "student_pt",
        "opleiding": "education",
        "home_mrdh": "home_zone_id"
    },
    inplace=True,
)
p = p[
    [
        "person_id",
        "household_id",
        "age",
        "sex",
        "pemploy",
        "PNUM",
        "pstudent",
        "ptype",
        "education",
        "driving_license",
        "roots_person",
        "has_car",
        "type_car",
        "has_bike",
        "has_ebike",
        "student_pt",
        "urbanized",
        "maas_subscription",
        "school_zone_id",
        "workplace_zone_id",
        "home_zone_id",
        "free_parking_at_work",
    ]
]

p_inferred = p.copy()

# %% Trips
trips = odin[
    [
        "verplid",
        "opid",
        "tour_id",
        "doel",
        "ovstkaart",
        "dest_mrdh",
        "orig_mrdh",
        "vertuur",
        "aankuur",
        "hvm",
        "hvmrol",
        "outbound",
        "parent_tour_id"
    ]
].copy()

# Sort trips by opid and verplid
trips.sort_values(by=["opid", "verplid"], inplace=True)
trips["household_id"] = trips.opid
other = ["eatout", "social"]
# Distinguish between university and school
trips.loc[(trips.doel == 7) & (trips.ovstkaart.isin([1, 2])), "doel"] = 70
# Mapping of ODiN doel to trip purpose
trips["purpose"] = trips.doel.map(
    {
        1: "home",  # Naar huis
        2: "work",  # Werken
        3: "business",  # Zakelijk bezoek in werksfeer
        4: "",  # Vervoer als beroep (filtered)
        5: "escort",  # Afhalen/brengen personen
        6: "",  # Afhalen/brengen goederen (filtered)
        7: "school",  # Onderwijs/cursus volgen
        70: "university",  # Based on doel and student PT card possession
        8: "shopping",  # Winkelen/boodschappen doen
        9: "social",  # Visite/logeren
        10: "10",  # Toeren/wandelen
        11: "othmaint",  # Sport/hobby
        12: "othdiscr",  # Overige vrijetijdsbesteding
        13: "othmaint",  # Diensten/persoonlijke verzorging
        14: "14",  # Ander doel
        pd.NA: "home",
    }
)


# Do these separately as that draws a random choice for every row, not only once
# when the mapping is created
trips["purpose"] = trips.purpose.apply(lambda x: x.replace("10", random.choice(other)))
trips["purpose"] = trips.purpose.apply(lambda x: x.replace("14", random.choice(other)))

# TODO map subtour purposes properly

trips["trip_mode"] = trips.hvm.map(
    {
        1: "CAR",   # Personenauto
        2: "WALK_PT_WALK",  # Trein
        3: "WALK_PT_WALK",  # Bus
        4: "WALK_PT_WALK",  # Tram
        5: "WALK_PT_WALK",  # Metro
        6: "EBIKE",  # Speedpedelec (>45 km/h)
        7: "EBIKE",  # Elektrische fiets
        8: "BIKE",  # Niet-elektrische fiets
        9: "WALK",  # Te voet
        10: "CARPASSENGER",  # Touringcar
        11: "CAR",  # Bestelauto
        12: "CAR",  # Vrachtwagen
        13: "CAR",  # Camper
        14: "DRT",  # Taxi/Taxibusje
        15: "CAR",  # Landbouwvoertuig
        16: "CAR",  # Motor
        17: "EBIKE",  # Bromfiets
        18: "EBIKE",  # Snorfiets
        19: "WALK",  # Gehandicaptenvervoermiddel met motor
        20: "WALK",  # Gehandicaptenvervoermiddel zonder motor
        21: "BIKE",  # Skates/skeelers/step
        22: "WALK_PT_WALK",  # Boot
        23: "CAR",  # Anders met motor
        24: "WALK",  # Anders zonder motor
    }
)

# Distinguish car drivers from passengers
trips.loc[trips.hvmrol == 2, "trip_mode"] = "CARPASSENGER"

# Calculate trip_num

# # Copy to a df for tours
# tours = trips.copy()

# Remove stay at home "trips"
trips = trips.loc[~trips.verplid.isna()]

# Copy to a df for tours
tours = trips.copy()

# Rename columns
trips.rename(
    columns={
        "opid": "person_id",
        "verplid": "trip_id",
        "dest_mrdh": "destination",
        "orig_mrdh": "origin",
        "vertuur": "depart",
    },
    inplace=True,
)
trips = trips[
    [
        "trip_id",
        "person_id",
        "household_id",
        "tour_id",
        "outbound",
        "purpose",
        "destination",
        "origin",
        "depart",
        "trip_mode",
    ]
]

trips_inferred = trips.copy()

# %% Tours
# %%% Type
# Rename columns
tours.rename(
    columns={
        "opid": "person_id",
        "dest_mrdh": "destination",
        "orig_mrdh": "origin",
        "vertuur": "start",
        "aankuur": "end",
        "purpose": "tour_type",
        "trip_mode": "tour_mode",
    },
    inplace=True,
)

# Define tour_type sort order
# The tour type is the first of these trip types that appears in the tour trips
tour_type_order = {
    "work": 0,
    "university": 1,
    "school": 2,
    "escort": 3,
    "shopping": 4,
    "othmaint": 5,
    "eatout": 6,
    "social": 7,
    "othdiscr": 8,
    "business": 9,
    "home": 10,
}
# Per tour get the origin, destination, start and end time
tours_timeplace = tours.groupby(by="tour_id").agg(
    {"destination": "last", "origin": "first", "start": "first", "end": "last"}
)


# Within each group sort the tour_type using the custom order
tours.sort_values(by="tour_type", key=lambda x: x.map(tour_type_order), inplace=True)
# Now in the group by we just need to pick the first tour_type/hvm to get the corresponding value
tours_properties = tours.groupby(by="tour_id").agg(
    {
        "person_id": "first",
        "household_id": "first",
        "tour_type": "first",
        "tour_mode": "first",
        "parent_tour_id": "first"
    }
)


tours = tours_properties.merge(
    tours_timeplace, left_index=True, right_index=True
).reset_index()


# %%% Category
tours["tour_category"] = tours.tour_type.map(
    {
        "home": "home",
        "work": "mandatory",
        "business": "atwork",
        "escort": "non_mandatory",
        "school": "mandatory",
        "university": "mandatory",
        "shopping": "non_mandatory",
        "othdiscr": "non_mandatory",
        "othmaint": "non_mandatory",
        "social": "non_mandatory",
        "eatout": "non_mandatory"
    }
)



# %%% Clean-up
# tours["parent_tour_id"] = pd.NA


tours = tours[
    [
        "tour_id",
        "person_id",
        "household_id",
        "tour_type",
        "tour_category",
        "destination",
        "origin",
        "start",
        "end",
        "tour_mode",
        "parent_tour_id",
    ]
]

tours_inferred = tours.copy()

# Infer trip number in tour
trips_inferred['trip_num'] = \
            trips.sort_values(by=['tour_id', 'outbound', 'depart', 'trip_id']).\
            groupby(['tour_id', 'outbound']).\
            cumcount() + 1


# Infer stop frequency (one less than the number of trips in tour)
alts = pd.read_csv(os.path.join(configs_dir, 'stop_frequency_alternatives.csv'), comment='#')

assert 'alt' in alts
assert 'in' in alts
assert 'out' in alts

freq = pd.DataFrame(index=tours['tour_id'])

# number of trips is one less than number of stops
freq['out'] = trips_inferred.loc[trips_inferred['outbound']].groupby('tour_id').trip_num.max() - 1
freq['in'] = trips_inferred.loc[~trips_inferred['outbound']].groupby('tour_id').trip_num.max() - 1
if (freq['out'] > 3).any():
    cfg.logger.warning('Some tours contain more than 3 stops on outbound, substituting with only 3')
    freq.loc[freq['out'] > 3, 'out'] = 3

if (freq['in'] > 3).any():
    cfg.logger.warning('Some tours contain more than 3 stops on inbound, substituting with only 3')
    freq.loc[freq['in'] > 3, 'in'] = 3
freq = pd.merge(freq.reset_index(), alts, on=['out', 'in'], how='left')

assert (freq['tour_id'] == tours_inferred['tour_id']).all()

tours_inferred['stop_frequency'] = freq.alt

# Include tour departure and duration (tdd - tour scheduling)
tdd_alts = pd.read_csv(os.path.join(configs_dir, 'tour_departure_and_duration_alternatives.csv'))
tdd_alts['duration'] = tdd_alts.end - tdd_alts.start
tdd_alts = tdd_alts.astype(np.int8)  # - NARROW
tdd_alts.loc[tdd_alts.shape[0]] = [pd.NA, pd.NA, pd.NA]
tdd_alts['tdd'] = tdd_alts.index

if not tours_inferred.start.isin(tdd_alts.start).all():
    print(tours_inferred[~tours_inferred.start.isin(tdd_alts.start)])
assert tours_inferred.start.isin(tdd_alts.start).all(), "not all tour starts in tdd_alts"

assert tours_inferred.end.isin(tdd_alts.end).all(), "not all tour starts in tdd_alts"

tdds = pd.merge(tours_inferred[['start', 'end']], tdd_alts, left_on=['start', 'end'], right_on=['start', 'end'], how='left')

if tdds.tdd.isna().any():
    bad_tdds = tours_inferred[tdds.tdd.isna()]
    print("Bad tour start/end times:")
    print(bad_tdds)
    bug

tours_inferred['tdd'] = tdds.tdd
tours_inferred.loc[tours_inferred['tour_type'] == 'home', 'tdd'] = 0  # FIXME Should I just drop home tours???
# tours_inferred.loc[tours_inferred['tour_type'] == 'home', 'tdd'] = ''

# Infer at work subtour frequency
alts = pd.read_csv(os.path.join(configs_dir, 'atwork_subtour_frequency_alternatives.csv'), comment='#')
tour_types = list(alts.drop(columns=alts.columns[0]).columns)  # get trip_types, ignoring first column
alts['alt_id'] = alts.index

work_tours = tours_inferred[tours_inferred.tour_type == 'work']
work_tours = work_tours[['tour_id']]

subtours = tours_inferred[tours_inferred.tour_category == 'atwork']
subtours = subtours[['tour_id', 'tour_type', 'parent_tour_id']]


tour_counts = pd.DataFrame(index=work_tours['tour_id'])
for tour_type in tour_types:
    # count subtours of this type by parent_tour_id
    tour_type_count = subtours[subtours.tour_type == tour_type].groupby('parent_tour_id').size()
    # backfill with 0 count
    tour_counts[tour_type] = tour_type_count.reindex(tour_counts.index).fillna(0).astype(np.int8)

tour_counts = \
    pd.merge(tour_counts.reset_index(), alts,
             left_on=tour_types, right_on=tour_types, how='left').set_index(tour_counts.index.name)

atwork_subtour_frequency = tour_counts.alt

if atwork_subtour_frequency.isna().any():
    bad_tour_frequencies = atwork_subtour_frequency.isna()
    cfg.logger.warning("Bad atwork subtour frequencies for %s work tours" % bad_tour_frequencies.sum())
    cfg.logger.warning("Bad atwork subtour frequencies: num_tours\n%s" %
                   tour_counts[bad_tour_frequencies])
    cfg.logger.warning("Bad atwork subtour frequencies: num_tours\n%s" %
                   subtours[subtours.parent_tour_id.isin(tour_counts[bad_tour_frequencies].index)].
                   sort_values('parent_tour_id'))
    bug

atwork_subtour_frequency = reindex(atwork_subtour_frequency, tours_inferred['tour_id']).fillna('')

tours_inferred['atwork_subtour_frequency'] = atwork_subtour_frequency

tours_inferred['composition'] = ''


# Infer CDAP activity (persons table), M if the person does mandatory tours, N if only non mandatory, H if just home
num_mandatory_tours = \
    tours_inferred[tours_inferred['tour_category'] == 'mandatory'].\
    groupby('person_id').size().reindex(p_inferred.person_id).fillna(0).astype(np.int8)
# num_mandatory_tours = num_mandatory_tours.to_frame().rename(columns={0: 'num_mandatory_tours'})

num_non_mandatory_tours = \
    tours_inferred[tours_inferred['tour_category'] == 'non_mandatory'].\
    groupby('person_id').size().reindex(p_inferred.person_id).fillna(0).astype(np.int8)

cdap_activity = pd.Series('H', index=p_inferred.person_id)
cdap_activity = cdap_activity.where(num_non_mandatory_tours == 0, 'N')
cdap_activity = cdap_activity.where(num_mandatory_tours == 0, 'M')

cdap_activity = cdap_activity.to_frame()
cdap_activity.reset_index(inplace=True)
cdap_activity.rename(columns={'index': 'person_id', 0: 'cdap_activity'}, inplace=True)
p_inferred = p_inferred.merge(cdap_activity, how='left', on='person_id')

# p_inferred['cdap_activity'] = cdap_activity

# work_tours = tours_inferred[tours_inferred.tour_type == 'work'].groupby('person_id').size()
# school_tours = tours_inferred[tours_inferred.tour_type == 'school'].groupby('person_id').size()

# Infer mandatory tour frequency
num_work_tours = \
    tours_inferred[tours_inferred.tour_type == 'work'].\
    groupby('person_id').size()#.reindex(p_inferred.index).fillna(0).astype(np.int8)

num_school_tours = \
    tours_inferred[tours_inferred.tour_type.isin(['school', 'university'])].\
    groupby('person_id').size()#.reindex(p_inferred.index).fillna(0).astype(np.int8)

num_school_tours = num_school_tours * 10

# num_university_tours = \
#     tours_inferred[tours_inferred.tour_type == 'university'].\
#     groupby('person_id').size().reindex(p_inferred.index).fillna(0).astype(np.int8)
#
# num_school_tours += num_university_tours

mtf = {
    0: '',
    1: 'work1',
    2: 'work2',
    3: 'work2',  # Added myself
    # 3: 'work3',  # Added myself
    10: 'school1',
    20: 'school2',
    30: 'school2',  # Added myself
    # 30: 'school3',  # Added myself
    11: 'work_and_school',
    12: 'work_and_school',  # Added myself
    # 12: 'work_and_school2',  # Added myself
    21: 'work_and_school',  # Added myself
    # 21: 'work2_and_school',  # Added myself
}

mandatory_tour_frequency = num_work_tours.add(num_school_tours, fill_value=0).astype(int).map(mtf).to_frame()
mandatory_tour_frequency.reset_index(inplace=True)
mandatory_tour_frequency.rename(columns={'index': 'person_id', 0: 'mandatory_tour_frequency'}, inplace=True)
p_inferred = p_inferred.merge(mandatory_tour_frequency, how='left', on='person_id')
p_inferred['mandatory_tour_frequency'] = p_inferred['mandatory_tour_frequency'].fillna(value='')


# p_inferred['mandatory_tour_frequency'] = (num_work_tours + num_school_tours*10).map(mtf)

# Infer non mandatory tour frequency # FIXME mapping is bonkers
alts = pd.read_csv(os.path.join(configs_dir, 'non_mandatory_tour_frequency_alternatives.csv'), comment='#')
alts = alts.astype(np.int8)  # - NARROW

tours_nm = tours_inferred[tours_inferred.tour_category == 'non_mandatory']

tour_types = list(alts.columns.values)

alts['alt_id'] = alts.index

unconstrained_tour_counts = pd.DataFrame(index=p_inferred.person_id) # actual tour counts (may exceed counts envisioned by alts)
# unconstrained_tour_counts = pd.DataFrame(index=p_inferred.index) # actual tour counts (may exceed counts envisioned by alts)
for tour_type in tour_types:
    # unconstrained_tour_counts[tour_type] = \
    tour_counts = \
    tours_inferred[tours_inferred.tour_type == tour_type]. \
        groupby('person_id').size()#.reindex(p_inferred.index).fillna(0).astype(np.int8)
    tour_counts = tour_counts.to_frame().rename(columns={0: tour_type})
    # tour_counts = tour_counts.reindex(tour_counts.index + 1)
    # tour_counts.index.shift(1)
    unconstrained_tour_counts = unconstrained_tour_counts.merge(tour_counts, how='left', left_index=True, right_index=True)
    unconstrained_tour_counts[tour_type] = unconstrained_tour_counts[tour_type].fillna(value=0)

# unconstrained_tour_counts = unconstrained_tour_counts.shift(-1, axis=0)

    # mandatory_tour_frequency = num_work_tours.add(num_school_tours, fill_value=0).astype(int).map(mtf).to_frame()
    # mandatory_tour_frequency.reset_index(inplace=True)
    # mandatory_tour_frequency.rename(columns={'index': 'person_id', 0: 'mandatory_tour_frequency'}, inplace=True)
    # p_inferred = p_inferred.merge(mandatory_tour_frequency, how='left', on='person_id')
    # p_inferred['mandatory_tour_frequency'] = p_inferred['mandatory_tour_frequency'].fillna(value='')

# unextend tour counts
# activitysim extend tours counts based on a probability table
# counts can only be extended if original count is between 1 and 4
# and tours can only be extended if their count is at the max possible
max_tour_counts = alts[tour_types].max(axis=0)
constrained_tour_counts = pd.DataFrame(index=p_inferred.person_id)
for tour_type in tour_types:
    constrained_tour_counts[tour_type] = unconstrained_tour_counts[tour_type].clip(upper=max_tour_counts[tour_type])

# persons whose tours were constrained who aren't eligible for extension because they have > 4 constrained tours
has_constrained_tours = (unconstrained_tour_counts != constrained_tour_counts).any(axis=1)
print("%s persons with constrained tours" % (has_constrained_tours.sum()))
too_many_tours = has_constrained_tours & constrained_tour_counts.sum(axis=1) > 4
if too_many_tours.any():
    print("%s persons with too many tours" % (too_many_tours.sum()))
    print(constrained_tour_counts[too_many_tours])
    # not sure what to do about this. Throw out some tours? let them through?
    print("not sure what to do about this. Throw out some tours? let them through?")
    assert False

alt_id = pd.merge(constrained_tour_counts.reset_index(), alts,
                  left_on=tour_types, right_on=tour_types, how='left').set_index(p_inferred.person_id).alt_id

# did we end up with any tour frequencies not in alts?
if alt_id.isna().any():
    bad_tour_frequencies = alt_id.isna()
    logger.warning("WARNING Bad joint tour frequencies\n\n")
    logger.warning("\nWARNING Bad non_mandatory tour frequencies: num_tours\n%s" %
                   constrained_tour_counts[bad_tour_frequencies])
    logger.warning("\nWARNING Bad non_mandatory tour frequencies: num_tours\n%s" %
                   tours_nm[tours_nm.person_id.isin(p_inferred.person_id[bad_tour_frequencies])].sort_values('person_id'))
    bug

tf = unconstrained_tour_counts.rename(columns={tour_type: '_%s' % tour_type for tour_type in tour_types})
tf['non_mandatory_tour_frequency'] = alt_id
tf.reset_index(drop=True, inplace=True)

for c in tf.columns:
    print("assigning persons", c)
    p_inferred[c] = tf[c]

# Add survey id columns
trips_inferred['survey_trip_id'] = trips_inferred['trip_id']*10
trips_inferred['survey_tour_id'] = trips_inferred['tour_id']*10
tours_inferred['survey_tour_id'] = tours_inferred['tour_id']*10
tours_inferred['survey_parent_tour_id'] = tours_inferred['parent_tour_id']*10



# %% Export to CSV
# NaN values MUST be replaced with -1
if cfg.export_to_csv:
    hh.to_csv(Path(output_directory, "survey_households.csv"), na_rep="-1", index=False)
    p.to_csv(Path(output_directory, "survey_persons.csv"), na_rep="-1", index=False)
    tours.to_csv(Path(output_directory, "survey_tours.csv"), na_rep="-1", index=False)
    trips.to_csv(Path(output_directory, "survey_trips.csv"), na_rep="-1", index=False)
    hh_inferred.to_csv(Path(output_directory, "override_households.csv"), na_rep="-1", index=False)
    p_inferred.to_csv(Path(output_directory, "override_persons.csv"), na_rep="-1", index=False)
    tours_inferred.to_csv(Path(output_directory, "override_tours.csv"), na_rep="-1", index=False)
    trips_inferred.to_csv(Path(output_directory, "override_trips.csv"), na_rep="-1", index=False)
# Stop timer
toc = time.time()
cfg.logger.info(f"Runtime: {round(toc-tic, 2)} seconds")

# EOF
