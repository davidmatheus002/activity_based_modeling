"""
File to hold all functions used to convert the ODiN data to survey files.

Created on Wed Sep 15 16:00:54 2021

@author: asharib

Modified on Tue Apr 19 11:20 2022

@editor: David Matheus
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg
import sql_queries as q
import openmatrix as omx


def translate_zones(
    odin, pc4xmrdh=Path(cfg.sourcedata, "PC4_x_MRDH.v2020.csv")
):
    """
    Translate PC4 zones to MRDH zones, taking into account trip characteristics.

    Parameters
    ----------
    odin : DATAFRAME
        The whole ODiN dataset.
    pc4xmrdh (optional) : DATAFRAME
        Table that describes the PC4 and MRDH overlaid and clipped, with geo-demographic data.

    Returns
    -------
    odh_mrdh : TYPE
        DESCRIPTION.

    """
    if cfg.pc4_mrdh_pickle.exists():
        pc4 = pd.read_pickle(cfg.pc4_mrdh_pickle)
    else:
        # Read the lookup table that connects PC4 zone fragments with MRDH zones
        pc4_mrdh = pd.read_csv(pc4xmrdh)

        # Remove MRDH zones with less than 0.5% overlap with PC4 zone
        pc4_mrdh.drop(index=pc4_mrdh.loc[pc4_mrdh.FracPC4 < 0.005].index, inplace=True)

        # Calculate values in each clipped zone fragment
        clipped = pd.DataFrame()
        clipped["pc4"] = pc4_mrdh.PC4
        clipped["mrdh"] = pc4_mrdh.SUBZONE0
        clipped["edu_ge12"] = (
            pc4_mrdh.LLP12EO_5 * pc4_mrdh.FracMRDH
        )  # Edu >= 12 y/o places
        clipped["labor"] = pc4_mrdh.ARBEIDS_9 * pc4_mrdh.FracMRDH  # Labor places
        clipped["edu_lt12"] = (
            pc4_mrdh.LLP001_27 * pc4_mrdh.FracMRDH
        )  # Edu < 12 y/o places
        clipped["pop"] = pc4_mrdh.POP * pc4_mrdh.FracMRDH  # Inhabitants
        clipped["shop"] = pc4_mrdh.N_WINKELS * pc4_mrdh.FracMRDH  # Shops
        clipped["frac_pc4"] = pc4_mrdh.FracPC4  # PC4 area fraction
        clipped.reset_index(inplace=True)

        # For each PC4 zone and for each variable, find the MRDH zone that has the maximum value
        # First find the ID of the row that is maximal for each variable
        maxids = clipped.groupby(by="pc4")[
            ["edu_ge12", "labor", "edu_lt12", "pop", "shop", "frac_pc4"]
        ].idxmax()
        # Replace the row ID with the MRDH zone number of that row
        # Transform to numpy array for fast indexing
        mrdh = clipped.mrdh.values
        ids = maxids.values
        # Place everything back in a DataFrame
        # This is now a lookup table between a PC4 zone and MRDH zones
        # which have a certain maximal variable
        pc4 = pd.DataFrame(float("nan"), index=range(0, 10000), columns=maxids.columns)
        # Place the data at the correct index
        pc4.loc[maxids.index] = mrdh[ids]
        pc4.to_pickle(cfg.pc4_mrdh_pickle)

    # WORK
    # Add the work pc4 zone to each row in ODiN
    opid_work = find_first_pc4(odin, 2)
    assert opid_work.opid.is_unique  # Make sure there is only one value per OPID
    odin = odin.merge(opid_work, how="left", on="opid")

    # SCHOOL
    # Add the school pc4 zone to each row in ODiN
    opid_school = find_first_pc4(odin, 7)
    assert opid_school.opid.is_unique  # Make sure there is only one value per OPID
    odin = odin.merge(opid_school, how="left", on="opid")

    # Convert home, school and work PC4 to MRDH
    # Define a mapping what MRDH zone should be chosen
    # (e.g. for work pc4 look at the MRDH zone with most labor places)
    type_mapping = {
        "home_pc4": "pop",
        "work_pc4": "labor",
        "school_pc4": ["edu_lt12", "edu_ge12"],
    }

    for k, v in type_mapping.items():
        if k == "school_pc4":
            # Age < 12 years
            selector = (~odin[k].isna()) & (odin.leeftijd < 12)
            odin.loc[selector, k.replace("_pc4", "_mrdh")] = pc4.loc[
                odin.loc[selector, k], v[0]
            ].values
            # Age >= 12 years
            selector = (~odin[k].isna()) & (odin.leeftijd >= 12)
            odin.loc[selector, k.replace("_pc4", "_mrdh")] = pc4.loc[
                odin.loc[selector, k], v[1]
            ].values
        else:
            selector = ~odin[k].isna()
            odin.loc[selector, k.replace("_pc4", "_mrdh")] = pc4.loc[
                odin.loc[selector, k], v
            ].values
    # DESTINATION
    # Convert the aankpc to MRDH based on doel
    purpose_mapping = {
        1: {"source": "home_pc4", "maxvar": "pop"},  # going home
        2: {"source": "aankpc", "maxvar": "labor"},  # to work
        3: {"source": "aankpc", "maxvar": "labor"},  # business
        4: {"source": "aankpc", "maxvar": "pop"},  # professional transport
        5: {"source": "aankpc", "maxvar": "pop"},  # pick up people
        6: {"source": "aankpc", "maxvar": "pop"},  # pick up goods
        7: {"source": "aankpc", "maxvar": ["edu_lt12", "edu_ge12"]},  # education
        8: {"source": "aankpc", "maxvar": "shop"},  # shopping
        9: {"source": "aankpc", "maxvar": "pop"},  # visit
        10: {"source": "aankpc", "maxvar": "pop"},  # taking a walk
        11: {"source": "aankpc", "maxvar": "pop"},  # sport
        12: {"source": "aankpc", "maxvar": "pop"},  # other free time
        13: {"source": "aankpc", "maxvar": "shop"},  # services/personal care
        14: {"source": "aankpc", "maxvar": "pop"},  # other
    }

    purposes = odin.doel.unique()  # Get all purposes in the data

    for k, v in purpose_mapping.items():
        if k not in purposes:  # Skip if purpose is not present in the data
            continue
        if k == 7:
            odin.loc[(odin.doel == k) & (odin.leeftijd < 12), "aank_mrdh"] = pc4.loc[
                odin.loc[(odin.doel == k) & (odin.leeftijd < 12), v["source"]],
                v["maxvar"][0],
            ].values
            odin.loc[(odin.doel == k) & (odin.leeftijd >= 12), "aank_mrdh"] = pc4.loc[
                odin.loc[(odin.doel == k) & (odin.leeftijd >= 12), v["source"]],
                v["maxvar"][1],
            ].values
        else:
            odin.loc[odin.doel == k, "aank_mrdh"] = pc4.loc[
                odin.loc[odin.doel == k, v["source"]], v["maxvar"]
            ].values
    # ORIGIN
    # Condition indexers
    no_trip = odin.verplid.isna()
    first_trip = odin.op == 1
    # By default trips depart from the aank_mrdh of the previous trip
    odin["vert_mrdh"] = odin.aank_mrdh.shift(1)

    # The first trip departs from vertpc with the most inhabitants
    # This also corrects the error introduced above where the aank_mrdh of one person
    # is written as the vert_mrdh of another.
    odin.loc[(first_trip) & ~(no_trip), "vert_mrdh"] = pc4.loc[
        odin.loc[(first_trip) & ~(no_trip), "vertpc"], "pop"
    ].values

    # Set the vert_mrdh to NaN for stay at homes
    odin.loc[no_trip, "vert_mrdh"] = float("nan")

    # Get the relevant columns only
    mrdh = odin[
        [
            "home_pc4",
            "work_pc4",
            "school_pc4",
            "vert_mrdh",
            "aank_mrdh",
            "home_mrdh",
            "work_mrdh",
            "school_mrdh",
        ]
    ].copy()

    return mrdh


def find_first_pc4(odin, goal):
    """
    Find the likely work or school PC4 zone for each OPID.

    It looks at the first PC4 destination of a trip with a certain goal and
    assumes that is the goal related pc4 (e.g. work pc4 or school pc4).

    Parameters
    ----------
    odin : DATAFRAME
        The whole ODiN dataset.

    goal : INT
        The goal (doel) ID to filter on. 2 is going to work, 7 is going to school.

    Returns
    -------
    opid_first : DATAFRAME
        Lists the pc4 of the first trip with a certain goal per OPID.

    """
    # First find work/school trips only
    first = odin.loc[odin.doel == goal, ["opid", "verplid", "aankpc"]]

    # Now sort ascending by opid and verplid then group by opid and select the first row
    first = (
        first.sort_values(["opid", "verplid"], ascending=True).groupby("opid").head(1)
    )
    if goal == 2:
        first.rename(columns={"aankpc": "work_pc4"}, inplace=True)
    elif goal == 7:
        first.rename(columns={"aankpc": "school_pc4"}, inplace=True)
    # Drop the verplid column and leave only the pc4 per OPID.
    opid_first = first.drop(columns=["verplid"])

    return opid_first


def generate_tour_id(odin):
    """
    Generate a global tour id for each trip to mark home-based tours.

    Parameters
    ----------
    odin : DATAFRAME
        The OViN/ODiN data with at least the OP, AANKPC, HOME_PC4 and DOEL columns.

    Returns
    -------
    tour_id : SERIES
        An incrementing global ID that identifies tours in the OViN/ODiN dataset.

    """
    # Get previous and next purposes
    odin["prev_doel"] = odin["doel"].shift(1)
    odin["next_doel"] = odin["doel"].shift(-1)
    odin["sec_next_doel"] = odin["doel"].shift(-2)

    # Create a column that will account for the tour_id increment per person and per person tour id
    odin['op_incr'] = 0
    # odin['op_tour_num'] = 0

    # Mark all home trips
    odin['home_trip'] = 0
    odin.loc[odin.doel == 1, 'home_trip'] = 1

    # Column to distinguish all tours and parent tours of op based on home trips
    odin['main_tours'] = odin['home_trip'].shift(1).cumsum().fillna(0)

    odin["next_opid"] = odin["opid"].shift(-1)
    odin["sec_next_opid"] = odin["opid"].shift(-2)
    # iterate per op to come up with proper values

    # Increment when op goes to work again without going home first
    odin.loc[(odin.doel != 1) &  # make sure that if it goes home it does not count as subtour
             (odin.doel != 2) &  # make sure that it does not go to work again
             (odin.prev_doel == 2) &
             ((odin.next_doel == 2) | (odin.sec_next_doel == 2)) &
             ((odin.next_doel != 1) & (odin.sec_next_doel != 1)) &
             ((odin.next_opid == odin.opid) & (odin.sec_next_opid == odin.opid)),
             'op_incr'] = 1
    odin.loc[(odin.doel != 1) &  # make sure that if it goes home it does not count as subtour
             (odin.doel != 2) &  # make sure that it does not go to work again
             (odin.prev_doel == 2) &
             (odin.next_doel == 2) & (odin.next_opid == odin.opid),  # Previous one does not capture if last subtour before home
             'op_incr'] = 1

    # Decrease again to main tour when going home if there are subtours

    cumulatives = odin.groupby(by=['opid', 'main_tours'])['op_incr'].sum().reset_index()
    cumulatives.op_incr = cumulatives.op_incr * (-1)
    cumulatives.rename(columns={'op_incr': 'home_incr'}, inplace=True)

    odin = pd.merge(odin, cumulatives, how='left', left_on=['opid', 'main_tours'], right_on=['opid', 'main_tours'])

    odin.loc[odin.doel == 1, 'op_incr'] = odin.loc[odin.doel == 1, 'home_incr']

    # Create global increment
    odin["incr"] = odin["op_incr"]

    # It increments additionally for every new opid
    odin.loc[odin.op == 1, "incr"] = 1

    # After returning home the increase must reflect if there were subtours previously
    odin["prev_incr"] = odin["incr"].shift(1)
    odin.loc[odin.prev_doel == 1, "incr"] = 1 - odin.loc[odin.prev_doel == 1]["prev_incr"]

    odin["tour_id"] = odin.incr.cumsum()

    temp_columns = ["next_doel", "sec_next_doel", "home_trip", "main_tours", "op_incr", "incr", "prev_incr",
                    "next_opid", "sec_next_opid", "home_incr"]

    odin.drop(columns=temp_columns, inplace=True)

    return odin


def generate_parent_tour_id(odin):

    # Find subtours by finding when the tour_id decreases below the trip's tour_id
    # The tour_id of the decrease is then the parent_tour_id
    odin["parent_tour_id"] = ''

    tours_to_look_forward = 10

    for i in range(tours_to_look_forward):
        odin[f'next_tour_id_{i + 1}'] = odin['tour_id'].shift(-1-i)

    next_tours_columns = [f'next_tour_id_{i + 1}' for i in range(tours_to_look_forward)]

    for id_column in next_tours_columns:
        odin.loc[(odin[id_column] - odin['tour_id']) < 0, 'parent_tour_id'] = odin[id_column]

    odin.drop(columns=next_tours_columns, inplace=True)


def generate_outbound_marker(odin):

    # First deal with main tours
    tour_purposes = odin.loc[odin.parent_tour_id == ''].copy()[['tour_id', 'verplid', 'purpose']]
    purpose_order = {
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

    inv_purpose_order = {v: k for k, v in purpose_order.items()}

    tour_purposes['purpose'] = tour_purposes['purpose'].map(purpose_order)
    tour_purposes = tour_purposes.reset_index().sort_values(by=["purpose", 'index']).drop(['index'], axis=1)
                                            # key=lambda x: x.map(purpose_order),
                                            # inplace=True)
        #.drop(['index'], axis=1)
    tour_purposes['purpose'] = tour_purposes['purpose'].map(inv_purpose_order)
    tour_purposes.rename(columns={'verplid': 'purpose_trip_id'}, inplace=True)
    tour_purposes = tour_purposes.groupby(by="tour_id").agg({"purpose_trip_id": "first"})

    # # All the trips until the purpose trip are outbound
    # odin['outbound'] = odin['verplid'] <= odin.join(tour_purposes, on='tour_id')['purpose_trip_id']

    # Now do the equivalent for subtours
    subtour_purposes = odin.loc[odin.parent_tour_id != ''].copy()[['tour_id', 'verplid', 'purpose']]

    subtour_purpose_order = {
        "business": 0,
        "maint": 1,
        "eatout": 2,
        "work": 3,
    }

    subtour_purposes.sort_values(by="purpose", key=lambda x: x.map(subtour_purpose_order), inplace=True)
    subtour_purposes.rename(columns={'verplid': 'purpose_trip_id'}, inplace=True)
    subtour_purposes = subtour_purposes.groupby(by="tour_id").agg({"purpose_trip_id": "first"})

    tour_purposes = tour_purposes.append(subtour_purposes)

    # All the trips until the purpose trip are outbound
    odin['outbound'] = odin['verplid'] <= odin.join(tour_purposes, on='tour_id')['purpose_trip_id']

    # First trip, if not going home, is outbound
    odin.loc[
        (odin.op == 1) & (odin.verpl == 1) & (odin.doel != 1) & (odin.outbound.isna()),
        "outbound",
    ] = True
    # If going home it is not outbound
    odin.loc[(odin.doel == 1) & (odin.outbound.isna()), "outbound"] = False
    # If going away from home
    odin.loc[(odin.prev_doel == 1) & odin.outbound.isna(), "outbound"] = True

    # If staying at home it should be NA (set this last)
    odin.loc[odin.verplid.isna(), "outbound"] = pd.NA


def opid_filter_report():
    """
    Count the number of OPIDs that are filtered by every additional SQL condition.

    Returns
    -------
    df : DATAFRAME
        Contains per SQL condition the number of OPIDs filtered.

    """
    query = q.opid_filter.replace("DISTINCT o.opid", "count(distinct o.opid)")
    query_lines = query.splitlines()
    filter_counts = []
    for f in range(4, len(query_lines) + 1):
        print(f"Filter: {f-3}")
        current_query = " ".join(query_lines[0:f])
        current_filter = query_lines[f - 1].strip()
        opids = pd.read_sql(current_query, cfg.odin_engine)
        filter_counts.append([current_filter, opids.iloc[0, 0]])
    df = pd.DataFrame(data=filter_counts, columns=["filter", "filtered_opids"])
    df["additional"] = df.filtered_opids.diff()
    return df


def generate_pemploy(persons, work_column='betwerk', workplace_column='work_mrdh'):
    """
            Generate PEMPLOY tag on persons table.

            Parameters
            ----------
            persons : DATAFRAME
                The ODiN data for persons.
            work_column : str
                Dataframe column name where paid work information for the person can be found, 'betwerk' in ODiN
            workplace_column : str
                Dataframe column name where workplace location for the person can be found.

            Returns
            -------
            persons : DATAFRAME
                Persons table with new PEMPLOY column


            PEMPLOY values:
            1: Full-time worker   - 'PEMPLOY_FULL'
            2: Part-time worker   - 'PEMPLOY_PART'
            3: Not in labor force - 'PEMPLOY_NOT'
            4: Student under 15   - 'PEMPLOY_CHILD'
            """

    # We can almost get a perfect mapping from survey employment information
    persons["pemploy"] = persons[work_column].map({0: 3, 1: 2, 2: 2, 3: 1, 5: 4})

    # However, we see that some people still do work trips even if they are not employed. These are often people that
    # listed as unemployed (perhaps gigs or job interviews?), with disabilities, retirees, or just perform unpaid work.
    # We assume these are part time workers.
    persons.loc[(persons['pemploy'].isin([3, 4])) & (pd.notna(persons[workplace_column])), 'pemploy'] = 2


def generate_pstudent(persons, retirement_age, age_column='leeftijd', ov_column='ovstkaart',
                      schoolplace_column='school_mrdh'):
    """
            Generate PSTUDENT tag on persons table.

            Parameters
            ----------
            persons : DATAFRAME
                The ODiN data for persons.
            retirement_age : int or float
                Age of retirement, passed to the function because it is being gradually incremented in the Netherlands.
            age_column : str
                Dataframe column name where age information for the person can be found, 'leeftijd' in ODiN
            ov_column : str
                Dataframe column name where ov student subscription information for the person can be found,
                'ovstkaart' in ODiN
            schoolplace_column : str
                Dataframe column name where school location for the person can be found.

            Returns
            -------
            persons : DATAFRAME
                Persons table with new PSTUDENT column


            PSTUDENT values:
            1: Preschool, middle school, high school... - 'PSTUDENT_GRADE_OR_HIGH'
            2: University/professional school student   - 'PSTUDENT_UNIVERSITY'
            3: Not a student                            - 'PSTUDENT_NOT'
            """

    persons["pstudent"] = 3

    persons.loc[persons[age_column].between(6, 16), "pstudent"] = 1
    persons.loc[
        (persons[age_column].between(17, 18))
        & ~(persons[ov_column].isin([1, 2])),
        "pstudent",
    ] = 1

    persons.loc[
        (persons[age_column].between(17, retirement_age))
        & ~(persons.pemploy == 1)
        & (persons[ov_column].isin([1, 2])),
        "pstudent",
    ] = 2

    # Some people make school trips even if they do not fall under the school or university categories. Perhaps these
    # are courses of some kind. Not more information is given, but it must be fixed to prevent inconsistencies. We
    # assume they attend university
    persons.loc[(persons['pstudent'] == 3) & (pd.notna(persons[schoolplace_column])), 'pstudent'] = 2


def generate_ptype(persons, retirement_age, age_column='leeftijd'):
    """
            Generate PTYPE tag on persons table.

            Parameters
            ----------
            persons : DATAFRAME
                The ODiN data for persons.
            retirement_age : int or float
                Age of retirement, passed to the function because it is being gradually incremented in the Netherlands.
            age_column : str
                Dataframe column name where age information for the person can be found, 'leeftijd' in ODiN

            Returns
            -------
            persons : DATAFRAME
                Persons table with new PTYPE column


            PTYPE values:
            | PTYPE |                         Person type                   |  Age  | Work status | School status |
            |-------|-------------------------------------------------------|-------|-------------|---------------|
            |   1   | Full-time worker (30+ hrs/week)                       |  18+  | Full-time   | None          |
            |   2   | Part-time worker (<30 hrs but works on regular basis) |  18+  | Part-time   | None          |
            |   3   | College student                                       |  18+  | Any         | College       |
            |   4   | Non-working adult                                     | 18-66 | Unemployed  | None          |
            |   5   | Retired person                                        |  67+  | Unemployed  | None          |
            |   6   | Driving age student                                   | 17-18 | Any         | Pre-college   |
            |   7   | Non-driving student                                   | 06-16 | None        | Pre-college   |
            |   8   | Pre-school child                                      | 00-05 | None        | None          |
            """

    persons.loc[(persons[age_column] > retirement_age), "ptype"] = 5

    persons.loc[(persons[age_column] >= 18) & (persons.pemploy == 1), "ptype"] = 1

    persons.loc[(persons[age_column] >= 18) & (persons.pemploy == 2), "ptype"] = 2

    persons.loc[(persons[age_column] >= 18) & (persons.pstudent == 2), "ptype"] = 3

    persons.loc[
        (persons[age_column].between(18, retirement_age))
        & (persons.pemploy == 3)
        & (persons.pstudent == 3),
        "ptype"
    ] = 4

    persons.loc[
        (persons[age_column].between(17, 18))
        & (persons.pstudent.isin([1, 2])),
        "ptype"
    ] = 6

    persons.loc[
        (persons[age_column].between(6, 16))
        & (persons.pstudent == 1),
        "ptype"
    ] = 7

    persons.loc[persons[age_column] < 6, "ptype"] = 8


def enumerate_tour_types(tour_flavors):
    # tour_flavors: {'eat': 1, 'business': 2, 'maint': 1}
    # channels:      ['eat1', 'business1', 'business2', 'maint1']
    channels = [tour_type + str(tour_num)
                for tour_type, max_count in tour_flavors.items()
                for tour_num in range(1, max_count + 1)]
    return channels


def get_mandatory_presampling_mapping(persons, destination_column, home_column='home_zone_id'):
    """
        Generate a mapping of the most likely work or school locations based on home zone.

        Parameters
        ----------
        persons : DATAFRAME
            The ODiN data with at least the person_id, home_column, and destination_column as columns.
        destination_column : str
            Dataframe column name where the work or school zone can be found
        home_column : str
            Dataframe column name where the home zone can be found

        Returns
        -------
        presample_mapping : DATAFRAME
            Mapping of home zones to likely destination zones

        """

    # Get counts of every origin and destination combination
    presample_mapping = persons.copy().dropna(subset=[destination_column])

    presample_mapping = presample_mapping.groupby(
        [home_column, destination_column]).size().to_frame(name='pick_count').reset_index()
    presample_mapping.sort_values([home_column, destination_column, 'pick_count'],
                                  ascending=[True, True, False],
                                  inplace=True)

    # Now get probability/frequency of every pair
    for origin in presample_mapping[home_column].unique().tolist():
        total = presample_mapping.loc[presample_mapping[home_column] == origin, 'pick_count'].sum()
        presample_mapping.loc[presample_mapping[home_column] == origin, 'prob'] = \
            presample_mapping.loc[presample_mapping[home_column] == origin, 'pick_count'] / total

    presample_mapping.rename(
        columns={home_column: 'home_zone_id', destination_column: 'alt_dest'},
        inplace=True)

    presample_mapping = presample_mapping[[home_column, 'alt_dest', 'prob', 'pick_count']]

    return presample_mapping


def get_spatial_mapping(persons, matrix_file, destination_column, home_column='home_zone_id'):
    """
        Generate a mapping of the most likely work or school locations based on home zone.

        Parameters
        ----------
        persons : DATAFRAME
            The ODiN data with at least the person_id, home_column, and destination_column as columns.
        matrix_file : filepath
            Location of the skim matrices.
        destination_column : str
            Dataframe column name where the work or school zone can be found
        home_column : str
            Dataframe column name where the home zone can be found

        Returns
        -------
        presample_mapping : DATAFRAME
            Mapping of home zones to likely destination zones based on activity spaces

        """

    # Get distances per OD pair
    skim_matrices = omx.open_file(matrix_file, 'r')
    distance_matrix = pd.DataFrame(np.array(skim_matrices['PRIVE_NIETGEDEELD_DIST'])).stack().to_frame().reset_index()
    distance_matrix['level_0'] = distance_matrix['level_0'].apply(lambda x: x + 1)
    distance_matrix['level_1'] = distance_matrix['level_1'].apply(lambda x: x + 1)
    distance_matrix.rename(columns={'level_0': home_column, 'level_1': destination_column, 0: 'distance'}, inplace=True)

    # Find OD pairs in survey data and assign distances

    persons = persons[[home_column, destination_column]]\
        .dropna(subset=[destination_column])\
        .merge(distance_matrix, how='left', on=[home_column, destination_column])

    # Find maximum distances per origin, this will be the search radius for the sampling
    search_radius = persons.groupby(home_column)['distance'].max().dropna().to_frame().reset_index()
    search_radius.rename(columns={'distance': 'max_distance'}, inplace=True)

    # Sometimes an oddly long or short trip appears on the data and defines the search radius. When it is too short we
    # can get too few zones within the radius making the choice not accurate, and when it is too big the simulation gets
    # to a standstill. It is better to remove these outliers by comparing with a moving average, as they do not improve
    # simulation outputs.
    moving_avg_window = 7  # Number of zones to average
    repetitions = 2  # Perhaps more than once to remove the influence of subsequent outliers
    for _ in range(repetitions):
        search_radius['moving_average'] = search_radius['max_distance'].rolling(moving_avg_window, center=True).mean()
        search_radius['moving_average'] = search_radius['moving_average'].bfill()
        search_radius['moving_average'] = search_radius['moving_average'].ffill()
        search_radius['max_threshold'] = search_radius['moving_average'] * 1.5  # Large to really just remove if needed
        search_radius['min_threshold'] = search_radius['moving_average'] / 2  # Small to really just remove if needed
        search_radius = search_radius.loc[
            ~(search_radius['max_distance'] > search_radius['max_threshold'])
            & ~(search_radius['max_distance'] < search_radius['min_threshold'])
        ]
    search_radius.drop(columns=['moving_average', 'max_threshold', 'min_threshold'], inplace=True)

    # Fill in the data for those zones without survey information (or excluded because they were outliers)
    distance_matrix = distance_matrix.merge(search_radius, how='left', on=home_column)
    # distance_matrix['max_distance'] = distance_matrix['max_distance'].interpolate()
    distance_matrix['max_distance'] = distance_matrix['max_distance'].ffill()
    distance_matrix['max_distance'] = distance_matrix['max_distance'].bfill()


    # Select only the destinations within the sample radius
    presample_mapping = distance_matrix.loc[distance_matrix.distance <= distance_matrix.max_distance]
    presample_mapping.drop(columns=['distance', 'max_distance'], inplace=True)

    # Missing origins will be assumed to only take internal trips
    included = set(presample_mapping.home_zone_id.unique().tolist())
    needed = set([i for i in range(1, 7788)])
    to_include = list(needed - included)
    presample_mapping = presample_mapping.append(pd.DataFrame(zip(to_include, to_include),
                                                              columns=[home_column, destination_column]),
                                                 ignore_index=True)
    presample_mapping.sort_values(by=[home_column], inplace=True)

    # Add other necessary columns, uniform distribution is assumed
    presample_mapping['pick_count'] = 1
    totals = presample_mapping[[home_column, 'pick_count']].groupby(home_column).count().reset_index()
    totals.rename(columns={'pick_count': 'prob'}, inplace=True)
    totals['prob'] = 1 / totals['prob']
    presample_mapping = presample_mapping.merge(totals, how='left', on=home_column)

    presample_mapping.rename(columns={home_column: 'home_zone_id', destination_column: 'alt_dest'}, inplace=True)

    return presample_mapping


def get_non_mandatory_spatial_mapping(tours, matrix_file, destination_column='destination', origin_column='origin'):
    """
        Generate a mapping of the most non_mandatory tour destinations based on home zone.

        Parameters
        ----------
        tours : DATAFRAME
            The ODiN data with at least the tour_id, tour_category, origin_column, and destination_column as columns.
        matrix_file : filepath
            Location of the skim matrices.
        destination_column : str
            Dataframe column name where the work or school zone can be found
        origin_column : str
            Dataframe column name where the home zone can be found

        Returns
        -------
        presample_mapping : DATAFRAME
            Mapping of home zones to likely destination zones based on activity spaces

        """

    # Get distances per OD pair
    skim_matrices = omx.open_file(matrix_file, 'r')
    distance_matrix = pd.DataFrame(np.array(skim_matrices['PRIVE_NIETGEDEELD_DIST'])).stack().to_frame().reset_index()
    distance_matrix['level_0'] = distance_matrix['level_0'].apply(lambda x: x + 1)
    distance_matrix['level_1'] = distance_matrix['level_1'].apply(lambda x: x + 1)
    distance_matrix.rename(columns={'level_0': origin_column, 'level_1': destination_column, 0: 'distance'}, inplace=True)

    # Exclude mandatory tours
    tours = tours.loc[tours.tour_category != 'mandatory']

    # Find OD pairs in survey data and assign distances
    tours = tours[[origin_column, destination_column]]\
        .dropna(subset=[destination_column])\
        .merge(distance_matrix, how='left', on=[origin_column, destination_column])

    # Find maximum distances per origin, this will be the search radius for the sampling
    search_radius = tours.groupby(origin_column)['distance'].max().dropna().to_frame().reset_index()
    search_radius.rename(columns={'distance': 'max_distance'}, inplace=True)

    # Fill in the data for those zones without survey information
    distance_matrix = distance_matrix.merge(search_radius, how='left', on=origin_column)
    distance_matrix['max_distance'] = distance_matrix['max_distance'].bfill()
    distance_matrix['max_distance'] = distance_matrix['max_distance'].ffill()

    # Select only the destinations within the sample radius
    presample_mapping = distance_matrix.loc[distance_matrix.distance <= distance_matrix.max_distance]
    presample_mapping.drop(columns=['distance', 'max_distance'], inplace=True)

    # Missing origins will be assumed to only take internal trips
    included = set(presample_mapping.home_zone_id.unique().tolist())
    needed = set([i for i in range(1, 7788)])
    to_include = list(needed - included)
    presample_mapping = presample_mapping.append(pd.DataFrame(zip(to_include, to_include),
                                                              columns=[origin_column, destination_column]),
                                                 ignore_index=True)
    presample_mapping.sort_values(by=[origin_column], inplace=True)

    # Add other necessary columns, uniform distribution is assumed
    presample_mapping['pick_count'] = 1
    totals = presample_mapping[[origin_column, 'pick_count']].groupby(origin_column).count().reset_index()
    totals.rename(columns={'pick_count': 'prob'}, inplace=True)
    totals['prob'] = 1 / totals['prob']
    presample_mapping = presample_mapping.merge(totals, how='left', on=origin_column)

    presample_mapping.rename(columns={origin_column: 'home_zone_id', destination_column: 'alt_dest'}, inplace=True)

    return presample_mapping


def canonical_tours():
    """
        From ActivitySim code.
        create labels for every the possible tour by combining tour_type/tour_num.

    Returns
    -------
        list of canonical tour labels in alphabetical order
    """

    # FIXME we pathalogically know what the possible tour_types and their max tour_nums are
    # FIXME instead, should get flavors from alts tables (but we would have to know their names...)
    # alts = pipeline.get_table('non_mandatory_tour_frequency_alts')
    # non_mandatory_tour_flavors = {c : alts[c].max() for c in alts.columns}

    # - non_mandatory_channels
    MAX_EXTENSION = 2
    non_mandatory_tour_flavors = {'escort': 2 + MAX_EXTENSION,
                                  'shopping': 1 + MAX_EXTENSION,
                                  'othmaint': 1 + MAX_EXTENSION,
                                  'othdiscr': 1 + MAX_EXTENSION,
                                  'eatout': 1 + MAX_EXTENSION,
                                  'social': 1 + MAX_EXTENSION}
    non_mandatory_channels = enumerate_tour_types(non_mandatory_tour_flavors)

    # - mandatory_channels
    mandatory_tour_flavors = {'work': 2, 'school': 2}
    mandatory_channels = enumerate_tour_types(mandatory_tour_flavors)

    # - atwork_subtour_channels
    # we need to distinguish between subtours of different work tours
    # (e.g. eat1_1 is eat subtour for parent work tour 1 and eat1_2 is for work tour 2)
    atwork_subtour_flavors = {'eat': 1, 'business': 2, 'maint': 1}
    atwork_subtour_channels = enumerate_tour_types(atwork_subtour_flavors)
    max_work_tours = mandatory_tour_flavors['work']
    atwork_subtour_channels = ['%s_%s' % (c, i+1)
                               for c in atwork_subtour_channels
                               for i in range(max_work_tours)]

    # - joint_tour_channels
    joint_tour_flavors = {'shopping': 2, 'othmaint': 2, 'othdiscr': 2, 'eatout': 2, 'social': 2}
    joint_tour_channels = enumerate_tour_types(joint_tour_flavors)
    joint_tour_channels = ['j_%s' % c for c in joint_tour_channels]

    sub_channels = \
        non_mandatory_channels + mandatory_channels + atwork_subtour_channels + joint_tour_channels

    sub_channels.sort()

    return sub_channels


def set_tour_id(tours, is_joint=False):
    """
    Canonical tour IDs as expected from ActivitySim. Tour IDs must be in this format for estimation, otherwise the
    index of the override tours table will not match the index of the simulated tours table and override will not be
    possible.

    The new index values are stable based on the person_id, tour_type, and tour_num.
    The existing index is ignored and replaced.

    This gives us a stable (predictable) tour_id with tours in canonical order
    (when tours are sorted by tour_id, tours for each person
    of the same type will be adjacent and in increasing tour_type_num order)

    It also simplifies attaching random number streams to tours that are stable
    (even across simulations)

    Parameters
    ----------
    tours : DataFrame
        Tours dataframe to reindex.
    """

    tour_num_col = 'tour_type_num'

    grouped = tours.groupby(['person_id', 'tour_type'])
    tours[tour_num_col] = grouped.cumcount() + 1

    parent_tour_num_col = 'parent_tour_type_num'
    grouped = tours.groupby(['parent_tour_id', 'tour_type'])
    tours[parent_tour_num_col] = grouped.cumcount() + 1
    tours.loc[tours.parent_tour_id == '', parent_tour_num_col] = pd.NA

    possible_tours = canonical_tours()
    possible_tours_count = len(possible_tours)

    assert tour_num_col in tours.columns

    # create string tour_id corresponding to keys in possible_tours (e.g. 'work1', 'j_shopping2')
    tours['tour_id'] = tours.tour_type + tours[tour_num_col].map(str)

    if parent_tour_num_col:
        # we need to distinguish between subtours of different work tours
        # (e.g. eat1_1 is eat subtour for parent work tour 1 and eat1_2 is for work tour 2)

        parent_tour_num = tours.loc[~tours[parent_tour_num_col].isna(), parent_tour_num_col]
        if parent_tour_num.dtype != 'int64':
            # might get converted to float if non-subtours rows are None (but we try to avoid this)
            cfg.logger.error('parent_tour_num.dtype: %s' % parent_tour_num.dtype)
            parent_tour_num = parent_tour_num.astype(np.int64)

        tours.loc[~tours[parent_tour_num_col].isna(), 'tour_id'] = tours['tour_id'] + '_' + parent_tour_num.map(str)
        tours.loc[tours[parent_tour_num_col].isna(), parent_tour_num_col] = ''

    if is_joint:
        tours['tour_id'] = 'j_' + tours['tour_id']

    # Unrecognized strings possibly means the respondent has more tours than allowed, delete these respondents
    opids_for_removal = tours.loc[~tours.tour_id.isin(possible_tours), 'person_id'].unique().tolist()
    if opids_for_removal:
        tours = tours.loc[~tours.person_id.isin(opids_for_removal)]

    # map recognized strings to ints
    tours.tour_id = tours.tour_id.replace(to_replace=possible_tours,
                                          value=list(range(possible_tours_count)))


    # convert to numeric - shouldn't be any NaNs - this will raise error if there are
    tours.tour_id = pd.to_numeric(tours.tour_id, errors='raise').astype(np.int64)

    tours.tour_id = (tours.person_id * possible_tours_count) + tours.tour_id

    if tours.tour_id.duplicated().any():
        print("\ntours.tour_id not unique\n%s" % tours[tours.tour_id.duplicated(keep=False)])
        print(tours[tours.tour_id.duplicated(keep=False)][['survey_tour_id', 'tour_type', 'tour_category']])
    assert not tours.tour_id.duplicated().any()

    tours.drop(columns=[tour_num_col], inplace=True)

    # we modify tours in place, but return the dataframe for the convenience of the caller
    return tours, opids_for_removal


def set_trip_id(trips):

    MAX_TRIPS_PER_LEG = 4  # max number of trips per leg (inbound or outbound) of tour

    # canonical_trip_num: 1st trip out = 1, 2nd trip out = 2, 1st in = 5, etc.
    canonical_trip_num = (~trips.outbound * MAX_TRIPS_PER_LEG) + trips.trip_num
    trips['tour_id'] = trips['tour_id'].astype(int)
    trips['trip_id'] = trips.tour_id * (2 * MAX_TRIPS_PER_LEG) + canonical_trip_num
    trips.set_index('trip_id', inplace=True, verify_integrity=True)
    trips.reset_index(drop=False, inplace=True)

    # we modify trips in place, but return the dataframe for the convenience of the caller
    return trips


# EOF
