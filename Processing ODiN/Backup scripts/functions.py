"""
File to hold all functions used to convert the OViN/ODiN data to survey files.

Created on Wed Sep 15 16:00:54 2021

@author: asharib

Modified on Tue Apr 19 11:20 2022

@editor: David Matheus
"""

import pandas as pd
from pathlib import Path
import config as cfg
import sql_queries as q


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
    # HOME
    # Add the home pc4 zone to each row in ODiN
    # opid_home = find_home_pc4(odin, wogem_pc4)
    # assert opid_home.opid.is_unique  # Make sure there is only one value per OPID
     #odin = odin.merge(opid_home, how="left", on="opid")
    # odin.rename(columns={'wopc': 'home_pc4'}, inplace=True)

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

# Finding home is not needed in ODiN because the pc4 for home is already given as wopc
# def find_home_pc4(odin, wogem_pc4):
#     """
#     Determine the home pc4 for each OPID.
#
#     Parameters
#     ----------
#     odin : DATAFRAME
#         The whole ODiN dataset.
#     wogem_pc4 : DATAFRAME
#         Lookup table of wogem and PC4.
#
#     Returns
#     -------
#     opid_home : DATAFRAME
#         Lists the home_pc4 per OPID.
#
#     """
#     # Home location of OPID's: get first row of data for each OPID.
#     opid_home = odin.loc[
#         odin.op == 1, ["opid", "verplid", "vertpc", "aankpc", "doel", "wogem", "jaar"]
#     ]
#
#     # Condition indexers
#     no_trip = opid_home.verplid.isna()
#     to_home = opid_home.doel == 1
#
#     # For those who stay at home and have no trip, look at the wogem
#     opid_home.loc[no_trip, "home_pc4"] = (
#         opid_home.loc[no_trip]
#         .merge(
#             wogem_pc4,
#             how="left",
#             left_on=["wogem", "jaar"],
#             right_on=["code", "year"],
#         )["pc4"]
#         .values
#     )
#
#     # First trip is going home, choose aankpc
#     opid_home.loc[to_home, "home_pc4"] = opid_home.aankpc
#     # All others, choose vertpc
#     opid_home.loc[opid_home.home_pc4.isna(), "home_pc4"] = opid_home.vertpc
#     # Only get relevant columns
#     opid_home = opid_home[["opid", "home_pc4"]]
#
#     return opid_home


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


def find_pc4_wogem(
    wogem_files=[
        "wogem2013.csv",
        "wogem2014.csv",
        "wogem2015.csv",
        "wogem2016.csv",
        "wogem2017.csv",
    ],
    pc4plaatsgemregio=Path(cfg.sourcedata, "PC4plaatsgemvregio2017.csv"),
):
    """
    Create a lookup table between the gemcode and a pc4 that lies in that gemeente.

    If a wogem has multipl pc4's the first pc4 is chosen, so not entirely random

    Parameters
    ----------
    wogem_files : LIST, optional
        A list of filenames that have the gemeente code and name per year.
    pc4plaatsgemregio : PATH, optional
        The path to the CSV file that lists the PC4's in a gemeente.

    Returns
    -------
    wogem_pc4 : DATAFRAME
        A lookup table between wogem code and PC4 zone.

    """
    if cfg.wogem_pickle.exists():
        wogem_pc4 = pd.read_pickle(cfg.wogem_pickle)
        return wogem_pc4
    gem = pd.read_csv(pc4plaatsgemregio)
    wogem = pd.DataFrame()
    # Load all wogem data into single dataframe, indicate year
    for wogem_file in wogem_files:
        temp = pd.read_csv(Path(cfg.sourcedata, wogem_file))
        temp["year"] = int(wogem_file[-8:-4])
        wogem = wogem.append(temp)
    # Get lookup table based on gemcode, woonplaats and gemeente
    gemcode = gem.groupby(by="GemCode")["PC4"].first()
    gemplaats = gem.groupby(by="Woonplaats")["PC4"].first()
    gemgem = gem.groupby(by="Gemeente")["PC4"].first()

    # Merge it with the wogem info
    wogem = wogem.merge(gemcode, how="left", left_on="code", right_index=True)
    wogem = wogem.merge(gemplaats, how="left", left_on="gemeente", right_index=True)
    wogem = wogem.merge(gemgem, how="left", left_on="gemeente", right_index=True)

    # Create a single pc4 column that uses takes the value basd on the code, the plaats or the gemeentenaam
    wogem["pc4"] = wogem["PC4_x"]  # Based on gemcode
    wogem["pc4"] = wogem.pc4.fillna(wogem.PC4_y).fillna(wogem.PC4)  # Fill missing data
    wogem_pc4 = wogem.drop(columns=["PC4", "PC4_x", "PC4_y"])

    wogem_pc4.to_pickle(cfg.wogem_pickle)
    return wogem_pc4


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
    # Create a tour id increment column
    odin["incr"] = 0
    # It increments for every new opid
    odin.loc[odin.op == 1, "incr"] = 1
    # It increments the trip after (hence shift(1)) someone went home
    odin.loc[
        (odin.doel == 1).shift(1).fillna(False),
        # ((odin.aankpc == odin.home_pc4) & (odin.doel == 1)).shift(1).fillna(False),
        "incr",
    ] = 1

    # Add a column with the previous and next doel
    odin["prev_doel"] = odin.groupby(by="opid")["doel"].shift(1)
    odin["next_doel"] = odin.groupby(by="opid")["doel"].shift(-1)

    # for person in unique_opids
        # if any work trips
            # find next work and next home indices/verplid
            # if next work < next home
                    # parent_tour_id = 'parent' on first work
                    # subtours = 0
                    #

    # Increment whenever someone does a visit during work (a sub-tour, doel == 3)
    # Find the verplid where the doel differs from the previous doel
    # If the previous doel is not 3 and the current doel is 3 increment tour_id
    odin.loc[(odin.prev_doel != 3) & (odin.doel == 3), "incr"] = 1

    # Sum the increment cumulatively, creating a tour_id column
    # No need to return as the odin dataframe is mutable, changing it in the function
    # changes the outside scope as well
    odin["tour_id"] = odin.incr.cumsum()


def generate_parent_tour_id(odin):
    # Find all tours where there is more than 1 work trip

    # Separate those subtours
    # Set all to blank first
    odin["parent_tour_id"] = ''
    # Select rows where doel changes to 3
    odin.loc[(odin.prev_doel != 3) & (odin.doel == 3), "parent_tour_id"] = (
        odin.tour_id - 1
    )
    # df2 = df1.groupby(by="opid")["verplid"].head(1)


def generate_outbound_marker(odin):

    # Get trip purpose
    tour_purposes = odin.copy()[['tour_id', 'verplid', 'doel', 'ovstkaart']]
    tour_purposes['purpose'] = tour_purposes['doel']
    tour_purposes.loc[(tour_purposes.doel == 7) & (tour_purposes.ovstkaart.isin([1, 2])), "purpose"] = 70
    tour_purposes["purpose"] = tour_purposes.purpose.map(
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

    tour_purposes.sort_values(by="purpose", key=lambda x: x.map(purpose_order), inplace=True)
    tour_purposes.rename(columns={'verplid': 'purpose_trip_id'}, inplace=True)
    tour_purposes = tour_purposes.groupby(by="tour_id").agg({"purpose_trip_id": "first"})

    # # Add to odin table
    # odin.join(tour_purposes, on='tour_id')

    # All the trips until the purpose trip are outbound
    odin['outbound'] = odin['verplid'] <= odin.join(tour_purposes, on='tour_id')['purpose_trip_id']

    # # First set all to NA
    # odin["outbound"] = pd.NA



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


if __name__ == "__main__":
    c = opid_filter_report()

    # find_pc4_wogem()
    # odin = pd.read_sql(q.odin, cfg.odin_engine)
    # translate_zones(odin, find_pc4_wogem())
