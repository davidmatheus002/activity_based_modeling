from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import expressions
# from .util.expressions import skim_time_period_label
# from .util.expressions import annotate_preprocessors
from .util.trip_acc_egr_choice import *
from .util.trip_mode_chain_constraints_checker import *
from activitysim.core import chunk
# from activitysim.abm.tables.constants import *
from math import sqrt, pow
import logging
import pandas as pd
import numpy as np
import os
from ctypes import *

logger = logging.getLogger(__name__)
FINAL_MODE_COL_NAME = 'multi_modal_mode'
FINAL_MODE_COL_NAME_STR = 'final_trip_mode'
HUB_AUTO_BIKE = 'hubAutoFiets'
HUB_BIKE_AUTO = 'hubFietsAuto'
HUB_AUTO_EBIKE = 'hubAutoEbike'
HUB_EBIKE_AUTO = 'hubEbikeAuto'
HUB_AUTO_PT = 'hubAutoOv'
HUB_PT_AUTO = 'hubOvAuto'
MULTIMODAL_CAR_AS_MAIN_WITH_BIKE = [2004001, 1004002]
MULTIMODAL_CAR_AS_MAIN_WITH_EBIKE = [3004001, 1004003]
MULTIMODAL_CAR_AS_MAIN_WITH_OTHER = [6004001, 7004001, 1004006, 1004007]
MULTIMODAL_CAR_AS_MAIN = [(MULTIMODAL_CAR_AS_MAIN_WITH_BIKE, [HUB_AUTO_BIKE, HUB_BIKE_AUTO]),
                          (MULTIMODAL_CAR_AS_MAIN_WITH_EBIKE, [HUB_AUTO_EBIKE, HUB_EBIKE_AUTO]),
                          (MULTIMODAL_CAR_AS_MAIN_WITH_OTHER, [HUB_AUTO_PT, HUB_PT_AUTO])]

# def skim_time_period_label(time):
#     """
#     convert time period times to skim time period labels (e.g. 9 -> 'AM')
#
#     Parameters
#     ----------
#     time : pandas Series
#
#     Returns
#     -------
#     pandas Series
#         string time period labels
#     """
#
#     skim_time_periods = config.setting('skim_time_periods')
#
#     # FIXME - eventually test and use np version always?
#     if np.isscalar(time):
#         bin = np.digitize([time % 24], skim_time_periods['hours'])[0] - 1
#         return skim_time_periods['labels'][bin]
#
#     return np.array(skim_time_periods['labels'])[pd.cut(time, skim_time_periods['hours'], labels=False)]


def replace_parallel(args):
    df_data, trip_mode_alt_utility = args
    for dictKey, dict_val in trip_mode_alt_utility.items():
        df_data[dictKey] = df_data[dictKey].map(dict_val)
    return df_data


def convert_mode_id2name(df_data, utility_col_name2id, use_feathers_output=False):
    # replace -1 to None, -1 is created by GPU dll
    df_data.loc[df_data[FINAL_MODE_COL_NAME] == -1, FINAL_MODE_COL_NAME] = None

    mode_id2name_map = {value: key for key, value in utility_col_name2id.items()}
    df_data[FINAL_MODE_COL_NAME_STR] = df_data[FINAL_MODE_COL_NAME]
    selection = df_data[FINAL_MODE_COL_NAME_STR].notnull()
    if use_feathers_output:
        df_data.loc[~selection, FINAL_MODE_COL_NAME] = df_data.loc[~selection, 'trip_mode'] * 1E3
        df_data.loc[~selection, FINAL_MODE_COL_NAME_STR] = df_data.loc[~selection, 'trip_mode'] * 1E3

    # replace with name
    df_data[FINAL_MODE_COL_NAME_STR].replace(mode_id2name_map, inplace=True)
    if not use_feathers_output:
        # for None value, fill in the trip_mode name and integer representation instead
        df_data.loc[~selection, FINAL_MODE_COL_NAME] = df_data.loc[~selection, 'main_mode_id'] * 1E3
        df_data.loc[~selection, FINAL_MODE_COL_NAME_STR] = df_data.loc[~selection, 'trip_mode']
        df_data[FINAL_MODE_COL_NAME_STR].replace({'PT': 'WALK_PT_WALK'}, inplace=True)
    # set to lower case
    df_data[FINAL_MODE_COL_NAME_STR] = df_data[FINAL_MODE_COL_NAME_STR].str.lower()


def determine_person_mode_ownership(df_pop, maas_enabled, private_car_high_priority):
    """
    get the person's ownership in order to load the correct mode chain sets, if
    :param df_pop:
    :param maas_enabled:
    :param private_car_high_priority: traveller having private car and driving licence, even he/she owns the MaaS subscription, still choose the private vehicles
    :return:
    """
    df_pop['total_ownership'] = (df_pop[FIELD_AUTO_OWNERSHIP] & df_pop[FIELD_DRIVING_LICENCE]) * 1 + df_pop[FIELD_BIKE_OWNERSHIP] * 2 + df_pop[FIELD_EBIKE_OWNERSHIP] * 4
    df_pop.loc[(df_pop[FIELD_MAAS_SUBSCRIPTION] > 0) & maas_enabled, 'total_ownership'] = 'MaaS'

    if private_car_high_priority:
        selection = (df_pop[FIELD_AUTO_OWNERSHIP] > 0) & (df_pop[FIELD_DRIVING_LICENCE] > 0)
        df_pop.loc[selection, 'total_ownership'] = df_pop.loc[selection, FIELD_AUTO_OWNERSHIP] * 1 + df_pop.loc[selection, FIELD_BIKE_OWNERSHIP] * 2 + df_pop.loc[selection, FIELD_EBIKE_OWNERSHIP] * 4
        # tmp['total_ownership'] = (tmp[FIELD_AUTO_OWNERSHIP] & tmp[FIELD_DRIVING_LICENCE]) * 1 + tmp[FIELD_BIKE_OWNERSHIP] * 2 + tmp[FIELD_EBIKE_OWNERSHIP] * 4
        # df_pop.loc[(df_pop[FIELD_AUTO_OWNERSHIP]) & (df_pop[FIELD_DRIVING_LICENCE]), 'total_ownership'] = (df_pop[FIELD_AUTO_OWNERSHIP] & df_pop[FIELD_DRIVING_LICENCE]) * 1 + \
        #                                                                                                   df_pop[FIELD_BIKE_OWNERSHIP] * 2 + df_pop[FIELD_EBIKE_OWNERSHIP] * 4


def valid_chain_set_with_main_mode(df_chainsets, trip_num, main_mode_id):
    df_chainsets["mode_set_valid"] = (df_chainsets["mode_set_valid"]) & \
                                     ((df_chainsets[
                                           'trip' + str(trip_num) + '_mode'] % 10 ** 6) // 10 ** 3 == main_mode_id)


def valid_chain_set_with_ownership(df_data, trip_num, person):
    col_name = 'trip' + str(trip_num) + '_mode'
    df_data['tmp_acc'] = df_data[col_name] // 10 ** 6
    df_data['tmp_main'] = df_data[col_name] % 10 ** 6 // 10 ** 3
    df_data['tmp_egr'] = df_data[col_name] % 10 ** 3
    # select multimodal modes require car
    selection = (df_data['tmp_acc'] == MODE_CAR) | (df_data['tmp_main'] == MODE_CAR) | (df_data['tmp_egr'] == MODE_CAR)
    df_data.loc[selection, "mode_set_valid"] &= person['can_use_carmode']
    # # bike
    selection = (df_data['tmp_acc'] == MODE_BIKE) | (df_data['tmp_main'] == MODE_BIKE) | (
                df_data['tmp_egr'] == MODE_BIKE)
    df_data.loc[selection, "mode_set_valid"] &= person['can_use_bikemode']
    # # ebike
    selection = (df_data['tmp_acc'] == MODE_EBIKE) | (df_data['tmp_main'] == MODE_EBIKE) | (
                df_data['tmp_egr'] == MODE_EBIKE)
    df_data.loc[selection, "mode_set_valid"] &= person['can_use_ebikemode']
    # # drt
    selection = (df_data['tmp_acc'] == MODE_DRT) | (df_data['tmp_main'] == MODE_DRT) | (df_data['tmp_egr'] == MODE_DRT)
    df_data.loc[selection, "mode_set_valid"] &= person['can_use_drtmode']


def valid_chain_set(val1, mode_id, main_mode_id):
    """
    DEPRECATED: VERY SLOW
    compare the main mode id with the main mode in a mode-combination
    :param val1: current validation value of previous trips in a tour
    :param mode_id: mode combination id having integer format as xxx,yyy,zzz (xxx for access, yyy-main, zzz-egress)
    :param main_mode_id: predicted main mode id
    :return:
    """
    return val1 and split_combined_id(mode_id)[1] == main_mode_id


def valid_ownership(val1, person, mode_id, maas_enabled):
    """
    DEPRECATED: VERY SLOW
    Check if the agent's ownership and travel resource subscription satisfy a mode combination
    :param val1: current validation value for the mode_id for previous trips
    :param person:
    :param mode_id: An integer represents the multi-modal: xxx,yyy,zzz (xxx for access, yyy-main, zzz-egress)
    :return:
    """
    # replace MaaS subscription to mode-specific subscription
    acc_mode, main_mode, egr_mode = split_combined_id(mode_id)
    # check what resource ownership is required
    if acc_mode == MODE_CAR or main_mode == MODE_CAR or egr_mode == MODE_CAR:
        # when a person has a car and driving license or a carshr-subscription, he/she can use CAR mode
        if not (person[FIELD_DRIVING_LICENCE] and person[FIELD_AUTO_OWNERSHIP]) and not (
                person[FIELD_MAAS_SUBSCRIPTION] and maas_enabled):
            return False
    elif acc_mode == MODE_BIKE or main_mode == MODE_BIKE or egr_mode == MODE_BIKE:
        if not person[FIELD_BIKE_OWNERSHIP] and not (person[FIELD_MAAS_SUBSCRIPTION] and maas_enabled):
            return False
    elif acc_mode == MODE_EBIKE or main_mode == MODE_EBIKE or egr_mode == MODE_EBIKE:
        if not person[FIELD_EBIKE_OWNERSHIP] and not (person[FIELD_MAAS_SUBSCRIPTION] and maas_enabled):
            return False
    elif acc_mode == MODE_DRT or main_mode == MODE_DRT or egr_mode == MODE_DRT:
        if not (person[FIELD_MAAS_SUBSCRIPTION] and maas_enabled):
            return False
    return True and val1


def filter_mode_chain_sets(df_data, main_modes):
    df_data = df_data.copy()
    for i in range(len(main_modes)):
        trip_nr = i + 1
        mode = main_modes[i]
        # df_data = df_data[df_data["trip{}_mode".format(trip_nr)].astype(str).str[-4].astype(int) == mode]
        df_data = df_data[df_data["trip{}_mode".format(trip_nr)] % 10 ** 6 // 10 ** 3 == mode]
    return df_data


def make_access_egress_mode_choice(output_dir, mode_chain_set_folder, utility_col_name2id, trips, maas_enabled, private_car_high_priority, seed_eta_offset,
                                   sigma, use_feather_input=False, use_gpu=False, dll_path='', use_CRN=1):
    """
    Make access and egress mode choice depends on the predicted Main mode done by earlier models
    :param output_dir
    :param mode_chain_set_folder:
    :param utility_col_name2id:
    :param trips:
    :param maas_enabled:
    :param private_car_high_priority: traveller having private car and driving licence, even he/she owns the MaaS subscription, still choose the private vehicles
    :param seed_eta_offset: seed offset for Eta random number
    :param sigma: std.deviation of normal distribution
    :param use_feather_input:
    :param use_gpu: Use GPU version for multimodal mode choice
    :param dll_path: full path of the GPU model
    :param use_CRN: use common random numbers
    :return:
    """
    t0 = tracing.print_elapsed_time()
    utility_col_names = list(utility_col_name2id.values())
    if use_feather_input:
        df_feathers_tours = trips.groupby(['tour_new_id']).size().to_frame('count')
        df_feathers_tours.index.rename('feathers_tour_id', inplace=True)
        pipeline.get_rn_generator().add_channel('feathers_tours', df_feathers_tours)
        trips['main_mode_id'] = trips['trip_mode']
    else:
        trips['trip_mode'].replace({'WALK_PT_WALK': 'PT', 'CARPASSENGER': 'CP'}, inplace=True)
        # convert trip mode (made by other models) name to ID
        trips['main_mode_id'] = trips['trip_mode'].apply(lambda x: TrafficMode.mode_name_map[x.lower()])

    # calculate the ownership number by check the person's ownership to determine if he/she can use certain mode
    determine_person_mode_ownership(trips, maas_enabled, private_car_high_priority)

    # initial the final multi modal mode
    trips[FINAL_MODE_COL_NAME] = None

    '''Read all the mode chain sets and save in dictionary using (ownership, tour type) as key.
    Open stores, each of which represents different type of ownership combinations: e.g. agent owns car,bike,ebike 
    having name like ModeChainSet_7_ownership_%_totalModes.h5, here 7 (111 in binary) means those combinations taken
    car,bike and ebike in generation.  If agent only owns car, the store's name is *_1_ownership_*.h5
       VALUE   has_car has_bike has_ebike
         0   =   0       0       0
         1   =   1       0       0
         2   =   0       1       0
         3   =   1       1       0
         ...

    Within each store, it contains different type of tours. Currently we generate mode combination for a tour consist 
    of 2/3/4/5/6 trips and tour+subtour(4trips). 
    '''
    # this dictionary will contain pairs (ownership, num_trips) that need to be processed later
    process_combinations = {}

    # TODO: 3 means there are 3 modes requires travel resource ownership: car,bike and ebike;
    #  Try to define 3 as constant somewhere

    # read the mode chain sets for different ownership combinations from 0 ~ 7
    t1 = tracing.print_elapsed_time()
    mode_chain_sets = {}

    # initial the GPU dll
    if use_gpu:
        dll_gpu_acc_egr = cdll.LoadLibrary(dll_path)
        # create instance
        dll_gpu_acc_egr.create.restype = c_void_p
        acc_egr_chooser = dll_gpu_acc_egr.create()
        # read the mode chain sets
        data_dir = os.getcwd()
        print("GPU data directory: " + data_dir)
        dll_gpu_acc_egr.read_mode_chain_sets.argtypes = [c_void_p, c_char_p]
        dll_gpu_acc_egr.read_mode_chain_sets.restype = c_bool
        dll_gpu_acc_egr.read_mode_chain_sets(acc_egr_chooser, data_dir.encode("ascii"))

    for i in range(2 ** 3):
        filename = mode_chain_set_folder + 'split_ModeChainSet_' + str(i) + '_ownership_' + str(
            len(utility_col_names)) + "_TotalModes.h5"
        # i represents the ownership combination number
        store = pd.HDFStore(filename, mode='r')
        for key in store.keys():
            key_set = key.split('/')
            if not use_gpu:
                mode_chain_sets[(i, key)] = pd.read_hdf(store, key)
            process_combinations[(i, key_set[1])] = 1

    if use_gpu:
        t1 = tracing.print_elapsed_time("\nload mode chain sets keys...", t1)
    else:
        t1 = tracing.print_elapsed_time("\nload mode chain sets...", t1)

    # read the mode chain sets when MaaS is enabled
    if maas_enabled:
        store = pd.HDFStore(
            mode_chain_set_folder + "split_ModeChainSet_MaaS_" + str(len(utility_col_names)) + "_TotalModes.h5",
            mode='r')
        for key in store.keys():
            key_set = key.split('/')
            # todo: currently we ignore the MaaS with 5 trips
            if key_set[1] == "5_trips":
                continue
            if not use_gpu:
                mode_chain_sets[('MaaS', key)] = pd.read_hdf(store, key)
            process_combinations[('MaaS', key_set[1])] = 1
        tracing.print_elapsed_time("loading mode chain sets (ignore tour with 5 trips) of MaaS...", t1)

    # add column to trips containing number of trips in the tours
    df_size = trips.groupby(['tour_new_id']).size().to_frame('num_trips_in_tour')
    trips = pd.merge(trips, df_size['num_trips_in_tour'], left_on='tour_new_id', right_index=True, how='left')

    # deal with X trips including a sub-tour, first select those tour ids, then assign full name, currently only tour
    # with maximum 8 trips are included
    if not use_feather_input:
        for i in range(4, 8):
            tour_with_subtour = trips.loc[(trips.num_trips_in_tour == i) & (~trips.parent_tour_id.isnull()),
                                          'tour_new_id'].drop_duplicates().tolist()
            # Then assign a special value
            trips.loc[trips.tour_new_id.isin(tour_with_subtour), 'num_trips_in_tour'] = str(i) + '_trips_subtour_from_2nd'

    # only take relevant columns of trips
    subset_cols = ['person_id', 'tour_new_id', 'trip_seq', 'main_mode_id', 'num_trips_in_tour', 'total_ownership',
                   HUB_AUTO_BIKE, HUB_BIKE_AUTO, HUB_AUTO_EBIKE, HUB_EBIKE_AUTO, HUB_AUTO_PT, HUB_PT_AUTO]
    column_utility_offset = len(subset_cols)
    subset_cols.extend(utility_col_names)
    trips_subsetcols = trips[subset_cols].sort_values(by=['person_id', 'tour_new_id', 'trip_seq'])

    # rename tour_new_id to tour_id
    trips_subsetcols.rename(columns={'tour_new_id': 'tour_id'}, inplace=True)

    print(process_combinations)

    # add an extra col: to be used by choice
    trips_subsetcols['trip_id'] = trips_subsetcols.index
    # set tour id as index
    trips_subsetcols.set_index('tour_id', inplace=True)
    column_utility_offset -= 1  # -1 because tour_new_id will be used as index

    # trips_subsetcols = trips_subsetcols.sort_index()
    # trips_subsetcols = trips_subsetcols

    t1 = tracing.print_elapsed_time()
    # execute for each <ownership, tour type> combination
    for k in process_combinations.keys():
        time_start = tracing.print_elapsed_time()
        person_ownership, tour_type = k
        num_trips_per_tour = int(tour_type[0])
        # special case:
        if tour_type[1:] == '_trips_subtour_from_2nd':
            num_trips_per_tour = tour_type
        # logger.info("Start to execute for ownership {0}, tour type {1}".format(person_ownership, tour_type))
        # select rows from trips based on person_ownership and num_trips_per_tour
        trips_subset = trips_subsetcols.loc[(trips_subsetcols['num_trips_in_tour'] == num_trips_per_tour)
                                            & (trips_subsetcols['total_ownership'] == person_ownership)]
        num_trips = trips_subset.shape[0]
        # logger.info("  Number of trips: {0}".format(num_trips))

        # skip if there are no trips to consider
        if num_trips == 0:
            continue

        # now just get the number of trips in tour
        num_trips_per_tour = int(tour_type[0])

        # deal with the hub location consistency!!
        # a. for tour of >= 3 trips, check the hub location in the first and last trip of the tour:
        # if they are NOT the same and person has no MaaS:
        #   the car mode can NOT be used in a multimodal mode, set the Utility of car-related to -9999
        t2 = tracing.print_elapsed_time()
        for mm_car_modes, hubs in MULTIMODAL_CAR_AS_MAIN:
            df_hub_not_consistency = trips_subset.groupby(trips_subset.index).nth(0)[hubs[0]] != \
                                     trips_subset.groupby(trips_subset.index).nth(-1)[hubs[1]]
            trips_subset = pd.merge(trips_subset, pd.DataFrame(df_hub_not_consistency, columns=['mm_no_car']),
                                    left_index=True, right_index=True, how='left')
            # b. get the nth([0,-1])'s trip id
            selection = trips_subset['trip_id'].isin(trips_subset.groupby(trips_subset.index).nth([0, -1])['trip_id'])
            # c. then set the utility of those car related multimodal to -9999 when people don't have a MaaS
            trips_subset.loc[selection & (trips_subset['mm_no_car'] == 1) & (
                        trips_subset['total_ownership'] != 'MaaS'), mm_car_modes] = -9999
            trips_subset.drop(columns='mm_no_car', axis=1, inplace=True)
        t2 = tracing.print_elapsed_time("check hub location consistency", t2)

        # create list of separate DataFrames containing 1st, 2nd, ..., nth trip of each tour, each DataFrame contains
        # the utility of nth trip of this specified tour type (2trips, 3trips, etc.). We add normal distributed error
        # term to each DataFrame!
        utilities_trip_n = []
        # a. set the seed in order to produce the same random number by every run
        np.random.seed(num_trips_per_tour + seed_eta_offset)
        for i in range(num_trips_per_tour):
            trips_selection = trips_subset.groupby(trips_subset.index).nth(i)

            # b. generate normal distribution random for each row, each row represents a trip of a different tour
            if not use_CRN:
                total_count = len(utility_col_names) * len(trips_selection)  # count = number of utility * number of rows
                eta = np.random.normal(0, sigma, total_count).reshape(len(trips_selection), len(utility_col_names))
            else:
                # logger.info("For tours {0} th trips, generate error term with std. dev {1}".format(i+1, sigma))
                eta = pipeline.get_rn_generator().normal_random_for_df(trips_selection, 0, sigma, len(utility_col_names))
            # df_eta = pd.DataFrame(eta, columns=list(trips_selection.columns), index=trips_selection.index)
            trips_selection.iloc[:, column_utility_offset: -1] += eta
            utilities_trip_n.append(trips_selection)
        t2 = tracing.print_elapsed_time("add Eta error term to each trip in tour type {0}".format(tour_type), t2)

        num_tours = utilities_trip_n[0].shape[0]
        # logger.info("  Number of tours: {0}".format(num_tours))

        """
        Use GPU based multi modal mode choice 
        """
        if use_gpu:
            #
            # FROM here to use the GPU-based multimodal mode chain choice
            #
            # step 1. export the Nth utilities of all tours having the same tour type (2,3,4 or etc) to ONE HDF5 files
            export_name = 'utilities_' + str(person_ownership) + '_ownership_' + str(tour_type) + '.h5'
            for i, trip_n_utilities in enumerate(utilities_trip_n, 1):
                tmp = trip_n_utilities.drop(['total_ownership', 'num_trips_in_tour', HUB_AUTO_BIKE, HUB_BIKE_AUTO, HUB_AUTO_EBIKE, HUB_EBIKE_AUTO, HUB_AUTO_PT, HUB_PT_AUTO], axis=1)
                # change the column name typs to string to avoid mixed type
                tmp.columns = tmp.columns.astype(str)
                tmp.to_hdf(output_dir + '/' + export_name, str(i))
                #logger.info("Export utilities for GPU kernel as HDF5 file {0}".format(export_name))

            # step 2. Get the results from DLL
            ny_trip_mode = np.zeros(len(trips_subset), dtype=c_int32)  # must allocate memory here
            dll_gpu_acc_egr.make_choice.argtypes = [c_void_p, c_void_p, c_int, c_char_p, c_char_p, c_bool]
            dll_gpu_acc_egr.make_choice.restype = c_bool
            data_dir = os.getcwd() + '/' + output_dir
            if person_ownership == 'MaaS':
                succeed = dll_gpu_acc_egr.make_choice(acc_egr_chooser, c_void_p(ny_trip_mode.ctypes.data), 999, tour_type.encode("ascii"), data_dir.encode("ascii"), False)
            else:
                succeed = dll_gpu_acc_egr.make_choice(acc_egr_chooser, c_void_p(ny_trip_mode.ctypes.data), person_ownership, tour_type.encode("ascii"), data_dir.encode("ascii"), False)

            if use_gpu and (not succeed):
                logger.error("GPU version to make choice FAILED. Check!")
                return

            # step 3. merge the result to trip
            df_trip_mode_choice = pd.DataFrame({FINAL_MODE_COL_NAME: ny_trip_mode})
            df_trip_mode_choice.index = trips_subset['trip_id']
            # assigned the mode in the selected mode chain sets to each trip of this tour
            trips.loc[df_trip_mode_choice.index, FINAL_MODE_COL_NAME] = df_trip_mode_choice
        else:
            # compute main mode combinations for which we need to filter
            main_mode_combinations = {}
            main_mode_combinations_per_tour = []
            main_mode_combinations_tourIDs = {}
            for i in range(num_tours):
                main_modes = []
                for j in range(num_trips_per_tour):
                    main_mode = utilities_trip_n[j].iloc[i]['main_mode_id']
                    main_modes.append(main_mode)
                main_modes = tuple(main_modes)
                main_mode_combinations_per_tour.append(main_modes)
                if not (main_modes in main_mode_combinations):
                    main_mode_combinations[main_modes] = 1
                    main_mode_combinations_tourIDs[main_modes] = [i]
                else:
                    main_mode_combinations[main_modes] = 1
                    main_mode_combinations_tourIDs[main_modes].append(i)
            t2 = tracing.print_elapsed_time("identify {} main mode combinations in trips".format(len(main_mode_combinations)), t2)

            # for each main mode combination we process the relevant tours
            for combination in main_mode_combinations:
                combination_str = '_'.join([str(c) for c in combination])

                # load mode chain sets
                combination_key = (person_ownership, '/{}/{}'.format(tour_type, combination_str))
                if not combination_key in mode_chain_sets.keys():
                    continue
                tour_mode_chain_sets = mode_chain_sets[combination_key]

                # remove _name columns if they exist
                num_chains = tour_mode_chain_sets.shape[0]
                # tour_mode_chain_sets = tour_mode_chain_sets[tour_mode_chain_sets.columns[~tour_mode_chain_sets.columns.str.endswith('_name')]]

                # get list of tour IDs (in range 0,...,n with number of tours n)
                tourIDs = main_mode_combinations_tourIDs[combination]
                logger.info(
                    "  Processing {0} tours having {1} Main_Mode_Combination with {2} ModeChainSets".format(len(tourIDs),
                                                                                                            combination,
                                                                                                            num_chains))

                # skip if there are no chains
                if num_chains == 0:
                    continue

                # flatten mode chain sets to a single row, each column has name with format C10_trip2_mode (10 is chain number, trip2 refers to 2nd trip)
                tour_mode_chain_sets_flat = tour_mode_chain_sets.unstack().to_frame().sort_index(level=1).T
                tour_mode_chain_sets_flat.columns = tour_mode_chain_sets_flat.columns.map('C{0[1]}_{0[0]}'.format)
                tour_mode_chain_sets_flat_array = tour_mode_chain_sets_flat.to_numpy()[0]

                # reduce dataframes in utilities_trip_n to relevant tours only
                utilities_trip_n_relevant = []
                for i in range(len(utilities_trip_n)):
                    df_tour = utilities_trip_n[i]
                    df_tour = df_tour.iloc[tourIDs, :]
                    df_tour = df_tour.copy()
                    utilities_trip_n_relevant.append(df_tour)

                # create dataframe with a row for each tour containing the columns of df_mode_chain_sets_flat as separate column
                df_tour = utilities_trip_n_relevant[0].copy()
                df_tour = df_tour[['person_id']]

                # create column for each chain (e.g., C1 for chain 1)
                final_utilities_cols = []
                for i in range(len(tour_mode_chain_sets_flat.columns)):
                    col = tour_mode_chain_sets_flat.columns[i]
                    if str(col).endswith("_mode") and i % num_trips_per_tour == 0:
                        newcol = col.split("_")[0]
                        df_tour[newcol] = 0
                        final_utilities_cols.append(newcol)

                # use utilities_trip_n to add utility columns to df_tour
                for i in range(len(tour_mode_chain_sets_flat.columns)):
                    col = tour_mode_chain_sets_flat.columns[i]

                    # get index representing number of this trip in the tour
                    trip_nr_in_tour = i % num_trips_per_tour

                    # get utilities for this trip nr
                    utilities_for_trip_mode = utilities_trip_n_relevant[trip_nr_in_tour]

                    # get the trip_mode for this column according to mode chain sets
                    trip_mode = tour_mode_chain_sets_flat_array[i]

                    # take the utilities column corresponding to the trip_mode
                    utilities = utilities_for_trip_mode[[trip_mode]]
                    df_tour[col] = utilities

                    # insert the utilities as a column in df_tour
                    sumcol = col.split("_")[0]
                    df_tour[sumcol] = df_tour[sumcol] + df_tour[col]

                # create dataframe with chain utilities for each tour
                df_tour = df_tour[final_utilities_cols]

                """make a choice"""
                df_tour.columns = np.arange(0, len(df_tour.columns))

                # if combination_str == '7_7' and person_ownership == 2:
                #     df_tour.to_csv('df_tour_utility_7_7_own2.csv')

                # deal with Total utilities which are really small, if each mode in chainset is real small, we should not
                # choose this chain set  because the alternatives are not suitable
                bad_utility = (df_tour < -900).sum(axis=1).sub(np.full((len(df_tour.index),), len(df_tour.columns))) == np.zeros(len(df_tour.index))
                if bad_utility.any():
                    # if total utility of tours are BAD, those trips belong to those tours won't make choice
                    no_choice_trip_idx = trips_subset.loc[df_tour[bad_utility].index, 'trip_id']
                    trips.loc[no_choice_trip_idx, FINAL_MODE_COL_NAME] = None
                    # logger.info("Bad utility for main mode combination " + combination_str)
                    # if combination_str == '7_7' and person_ownership == 2:
                    #     bad_utility.to_csv('df_bad_util_7_7_own2.csv')
                # deal with not-bad utility
                df_tour = df_tour[~bad_utility]
                choices = df_tour.idxmax(axis=1)
                # if combination_str == '7_7' and person_ownership == 2:
                #     choices.to_csv('df_choice_7_7_own2.csv')

                # retrieve the trip_id of those tours of specified tour type (2trips, 3trips, etc.)
                trip_idx = trips_subset.loc[df_tour.index, 'trip_id']

                #  assign the selected mode chain set to each trip
                alts = tour_mode_chain_sets.iloc[choices, 0:num_trips_per_tour].stack()
                alts.index = trip_idx

                # if combination_str == '7_7' and person_ownership == 2:
                #     alts.to_csv('df_alts_7_7_own2.csv')

                # assigned the mode in the selected mode chain sets to each trip of this tour
                trips.loc[alts.index, FINAL_MODE_COL_NAME] = alts

                # if combination_str == '7_7' and person_ownership == 2:
                #     trips.loc[alts.index, ].to_csv('df_trip_7_7_own2_choice.csv')

                """ Deprecated: MNL approach to make choice 
                probs = logit.utils_to_probs(df_tour, allow_zero_probs=True)
                BAD_PROB_THRESHOLD = 0.001
                bad_probs = probs.sum(axis=1).sub(np.ones(len(probs.index))).abs() > BAD_PROB_THRESHOLD * np.ones(len(probs.index))
                if bad_probs.any():
                     trips.loc[trip_idx, FINAL_MODE_COL_NAME] = None
                else:
    
                if use_feather_input:
                    probs.index.rename('feathers_tour_id', inplace=True)
                choices, rands = logit.make_choices(probs)
                # assign the selected mode chain set to each trip
                alts = tour_mode_chain_sets.iloc[choices, 0:num_trips_per_tour].stack()
                alts.index = trip_idx
                assigned the mode in the selected mode chain sets to each trip of this tour
                trips.loc[alts.index, FINAL_MODE_COL_NAME] = alts
                """
                logger.info("  Processed {0} tours having {1} Main_Mode_Combination and {2} ModeChainSet".format(
                    len(tourIDs), combination, len(tour_mode_chain_sets)))

            tracing.print_elapsed_time("make choice for each main mode combination", t2)
        # print runtime for this <ownership, tour type> combination

        if use_gpu:
            tracing.print_elapsed_time("GPU version make mode chain choice for {2} tours with ownership {0}, tour type {1}".format(person_ownership, tour_type, num_tours), time_start)
        else:
            tracing.print_elapsed_time("make mode chain choice for {2} tours with ownership {0}, tour type {1}".format(person_ownership, tour_type, num_tours), time_start)

    convert_mode_id2name(trips, utility_col_name2id, use_feather_input)
    tracing.print_elapsed_time("Make mode choice (exclude loading)", t1)
    tracing.print_elapsed_time("Total make_access_egress_mode_choice", t0)

    if use_gpu:
        dll_gpu_acc_egr.destroy.argtypes = [c_void_p]
        dll_gpu_acc_egr.destroy(acc_egr_chooser)
    return trips


@inject.step()
def trip_access_egress_mode_choice(output_dir, settings, trips, tours_merged, persons, households, network_los, chunk_size):#skim_dict, skim_stack
    """
    trip_access_egress_mode_choice - make access and egress mode choice
    The trip mode made in trip_mode_choice step is used as main mode, this model will add access mode and egress mode
    choice.
    :param settings: global settings
    :param trips:
    :param tours_merged:
    :param persons:
    :param households:
    :param skim_dict
    :param skim_stack
    :return:
    """
    print(output_dir)
    trace_label = 'trip_access_egress_mode_choice'
    # configs_dir = inject.get_injectable('configs_dir')
    model_settings = config.read_model_settings('trip_access_egress_choice.yaml')
    T_terminal_carbike = model_settings['CONSTANTS']['TRANSFER_TIME_CAR_BIKE']
    T_terminal_carpt = model_settings['CONSTANTS']['TRANSFER_TIME_CAR_PT']
    T_terminal_ptbike = model_settings['CONSTANTS']['TRANSFER_TIME_PT_BIKE']
    export_utility_to_csv = settings.get("export_utility", 0)
    export_trips_2_hdf = settings.get("EXPORT_TRIPS_2_HDF", 0)
    trips_file_name = settings.get('EXPORT_TRIPS_NAME')
    delft_area = settings.get('DELFT_AREA', 0)
    private_car_high_priority = settings.get('PRIVATE_CAR_HIGH_PRIORITY', 0) # default: use shared-car even if owns a private car
    
    # use common random number
    use_CRN = settings.get('USE_CRN', 1)
    use_input_sigma = (use_CRN == 1)
    sigma_crn = settings.get('SIGMA_FOR_CRN', None)
    if use_CRN and sigma_crn is None:
        logger.error("Must specify Std.deviation when use CRN approach.")
        return

    if use_CRN:
        if use_input_sigma:
            logger.info("Use Common Random Numbers across simulations, the standart deviation for Normal distribution is {0}".format(sigma_crn))
        else:
            logger.info("Use Common Random Numbers across simulations, the standart deviation for Normal distribution will be determined dynamically")
    else:
        logger.info("Not Use Common Random Numbers across simulations")

    # GPU based dll for multimodal mode choice
    use_gpu_version = settings.get('USE_GPU', 0)
    dll_path = settings.get('PATH_DLL_ACC_EGR_MODEL')
    if use_gpu_version and dll_path is None:
        logger.error('Enabled GPU model but the dll location is not specified')
        return
    logger.info("Use GPU model and DLL path: " + dll_path)

    # get seed for random generator for Mu
    seed_mu = settings.get('SEED_FOR_MU', 42)
    seed_eta_offset = settings.get('SEED_OFFSET_ETA', 0)
    if not use_CRN:
        logger.info("The seed for Mu random error is {0}".format(seed_mu))
        logger.info("The seed offset for Eta random error is {0}".format(seed_eta_offset))

    # get the enable_maas from settings.yaml
    enable_maas = settings.get('enable_maas', 0)
    logger.info("MaaS enabled = {0}, Parking factor = {1}".format(enable_maas, settings.get('parking_factor')))
    sigma_percent = settings.get('SIGMA_PERCENT', 0.1)

    # get the model specific settings
    model_settings = config.read_model_settings('trip_access_egress_choice.yaml')

    # get the list of mode names (lowercase) specified in the YAML file
    multimodes_list = read_modes_from_settings(config.read_model_settings('trip_access_egress_choice.yaml'))
    print('{} mode names: {}'.format(len(multimodes_list), multimodes_list))

    # get the utility expression from *.csv file
    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients = simulate.get_segment_coefficients(model_settings, 'general')
    # some utility parameters
    # omnibus_coefficients = assign.read_constant_spec(config.config_file_path(model_settings['COEFFS']))
    # print('Modes with utility function:')
    # print(model_spec.info())

    trips_df = trips.to_frame()

    # check if we need to use FEATHERS output
    use_feather_trips = settings.get('use_FEATHERS_trips', 0)
    df_origin = None
    if use_feather_trips:
        file_name = settings.get('feathers_input_store')
        trips_df = pd.read_hdf('data/' + file_name, 'feathers')

        # DEBUG
        # trips_df = trips_df[trips_df.person_id.isin([2066, 2120, 586627])]

        df_origin = trips_df.copy()
        trips_df = trips_df[trips_df.trip_id > 0]

        logger.info("Use FEATHERS trips {} as the input for the access-egress model.".format(file_name))
    else:
        logger.info("Use ActivitySim  as the input for the access-egress model.")
        # load persons, the index name is person_id
        persons_df = persons.to_frame()
        hh_df = households.to_frame()
        tours_merged_df = tours_merged.to_frame()
        persons_df = pd.merge(persons_df, hh_df[['auto_ownership', 'hhsize', 'income']], left_on='household_id', right_index=True, how='left')
        # merge with person and tour dataframe
        trips_df = pd.merge(trips_df, persons_df[model_settings['TOURS_MERGED_CHOOSER_COLUMNS']], left_on='person_id',
                            right_index=True, how='left')
        trips_df['is_joint'] = False
        trips_df = pd.merge(trips_df, tours_merged_df[['density_index', 'parent_tour_id', 'duration']],
                            left_on='tour_id', right_index=True, how='left')

    logger.info("Running %s with %d trips", trace_label, trips_df.shape[0])

    # print attributes in dfs
    # print("== PERSONS ==")
    # print(persons_df.info())
    # print("== TRIPS ==")
    # print(trips_df.info())
    # print("== HOUSEHOLDS ==")
    # print(hh_df.info())

    #
    # setup skim keys
    #
    trips_df['trip_period'] = network_los.skim_time_period_label(trips_df.depart)
    orig_col = 'origin'
    dest_col = 'destination'
    period_col = 'trip_period'
    # determine the intra zone column
    trips_df['is_intra_zone'] = trips_df[orig_col] == trips_df[dest_col]

    skim_dict = network_los.get_default_skim_dict()

    # prepare the skims dictionary
    odt_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col, dim3_key=period_col)
    od_skim_wrapper = skim_dict.wrap(orig_col, dest_col)
    # wrapper for AutoFiets hub
    o_autoBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_AUTO_BIKE, dim3_key=period_col)
    autoBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_AUTO_BIKE, dest_key=dest_col, dim3_key=period_col)
    # wrapper for bike-Auto hub
    o_bikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_BIKE_AUTO, dim3_key=period_col)
    bikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_BIKE_AUTO, dest_key=dest_col, dim3_key=period_col)

    # wrapper for Auto ebike hub
    o_autoEbikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_AUTO_EBIKE, dim3_key=period_col)
    autoEbikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_AUTO_EBIKE, dest_key=dest_col, dim3_key=period_col)
    # wrapper for ebike-Auto hub
    o_ebikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_EBIKE_AUTO, dim3_key=period_col)
    ebikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_EBIKE_AUTO, dest_key=dest_col, dim3_key=period_col)

    # wrapper for AutoOV Hub
    o_carPTt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_AUTO_PT, dim3_key=period_col)
    carPTHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_AUTO_PT, dest_key=dest_col, dim3_key=period_col)

    # wrapper for OVAuto Hub
    o_PTcart_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_PT_AUTO, dim3_key=period_col)
    PTCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_PT_AUTO, dest_key=dest_col, dim3_key=period_col)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
        "o_carbikehubt_skims": o_autoBikeHubt_skim_wrapper,
        "carbikehub_dt_skims": autoBikeHub_dt_skim_wrapper,
        "o_bikecarhubt_skims": o_bikeCarHubt_skim_wrapper,
        "bikecarhub_dt_skims": bikeCarHub_dt_skim_wrapper,
        "o_carEbikehubt_skims": o_autoEbikeHubt_skim_wrapper,
        "carEbikehub_dt_skims": autoEbikeHub_dt_skim_wrapper,
        "o_ebikecarhubt_skims": o_ebikeCarHubt_skim_wrapper,
        "ebikecarhub_dt_skims": ebikeCarHub_dt_skim_wrapper,
        "o_carpthubt_skims": o_carPTt_skim_wrapper,
        "carpthub_dt_skims": carPTHub_dt_skim_wrapper,
        "o_ptcarhubt_skims": o_PTcart_skim_wrapper,
        "ptcarhub_dt_skims": PTCarHub_dt_skim_wrapper,
    }
    constants = config.get_model_constants(model_settings)
    constants.update({'ORIGIN': orig_col, 'DESTINATION': dest_col})
    # locals_dict = assign.evaluate_constants(omnibus_coefficients["general"], constants=constants)
    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(coefficients)
    locals_dict.update(skims)
    locals_dict['enable_maas'] = enable_maas
    expressions.annotate_preprocessors(trips_df, locals_dict, skims, model_settings, trace_label)

    _FAKE_ZONE_NR = 7787
    if delft_area:
        print("\nSet the fake hub location to 239. (Delft Area)")
        logger.info(" Set the hub location to 239. (Delft case)")
        # #####################################################################################################
        """             NEED TO be disabled when calculate the full MRDH area
        # temperarily process the hub locations:
        # for all 0 < hub < 2251: set hub= 2483 (delft Hub)
        # for all hub > 2483: hub= 2483 (delft Hub)
        # now it holds that all hubs are in delft, or 0
        # for all hub>0: set hub = hub-2251+1  convert to Delft zone system
        # for all hub==0: hub = MAX_ZONE_NR+1 = 239
        # """
        _DELFT_START_ZONE = 2251
        _DELFT_END_ZONE = 2483
        _DELFT_HUB_ZONE = 2483
        _FAKE_ZONE_NR = 239
        for col in [HUB_AUTO_BIKE, HUB_BIKE_AUTO, HUB_AUTO_EBIKE, HUB_EBIKE_AUTO, HUB_AUTO_PT, HUB_PT_AUTO]:
            selection = ((trips_df[col] > 0) & (trips_df[col] < _DELFT_START_ZONE)) | (trips_df[col] > _DELFT_END_ZONE)
            trips_df.loc[selection, col] = _DELFT_HUB_ZONE
            # trips_df[col] += -_DELFT_START_ZONE + 1
            trips_df.loc[selection, col] += -_DELFT_START_ZONE + 1
            trips_df.loc[trips_df[col] == 0, col] = _FAKE_ZONE_NR
        # # #####################################################################################################
    else:
        print("\nSet the fake hub location to 7787")
        logger.info(" Set the fake hub location to 7787")
        _FAKE_ZONE_NR = 7787  # there are 1-7786 zones
        for col in [HUB_AUTO_BIKE, HUB_BIKE_AUTO, HUB_AUTO_EBIKE, HUB_EBIKE_AUTO, HUB_AUTO_PT, HUB_PT_AUTO]:
            trips_df.loc[trips_df[col] == 0, col] = _FAKE_ZONE_NR

    if use_feather_trips:
        # 0. determine the use of shared BIKE, use share bike whenever possible even having own bike
        # TT shared bike, tt_shr_bike1 is the total travel time by bike
        trips_df['tt_shr_bike1'] = odt_skim_stack_wrapper['MICRO15_GEDEELD_TIME']
        trips_df['can_use_bikeshare'] = (trips_df.new_mode_2 > 0) & (trips_df.tt_shr_bike1 < 9999)
        # TT shared bike from car-bike hub to destination, tt_shr_bike2, carbikehub_dt_skims['MICRO15_GEDEELD_TIME']
        trips_df['tt_shr_bike2'] = autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_TIME']
        trips_df['can_use_bikeshare_at_carbikehubs'] = (trips_df.new_mode_2 > 0) & (trips_df.tt_shr_bike2 < 9999)
        # TT shared bike from origin to bike-car hub, tt_shr_bike3, o_bikecarhubt_skims['MICRO15_GEDEELD_TIME']
        trips_df['tt_shr_bike3'] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_TIME']
        trips_df['can_use_bikeshare_at_bikecarhubs'] = (trips_df.new_mode_2 > 0) & (trips_df.tt_shr_bike3 < 9999)

        # # determine the use of shared EBIKE, use share EBIKE whenever possible even having own EBIKE
        # TT shared ebike, tt_shr_ebike1, odt_skims['MICRO25_GEDEELD_TIME']
        trips_df['tt_shr_ebike1'] = odt_skim_stack_wrapper['MICRO25_GEDEELD_TIME']
        trips_df['can_use_ebikeshare'] = (trips_df.new_mode_3 > 0) & (trips_df.tt_shr_ebike1 < 9999)
        # TT shared bike from car-bike hub to destination, _tt_shr_ebike2, carEbikehub_dt_skims['MICRO25_GEDEELD_TIME']
        trips_df['tt_shr_ebike2'] = autoEbikeHub_dt_skim_wrapper['MICRO25_GEDEELD_TIME']
        trips_df['can_use_ebikeshare_at_carebikehubs'] = (trips_df.new_mode_3 > 0) & (trips_df.tt_shr_ebike2 < 9999)
        # TT shared bike from origin to bike-car hub, _tt_shr_ebike3, o_ebikecarhubt_skims['MICRO25_GEDEELD_TIME']
        trips_df['tt_shr_ebike3'] = o_ebikeCarHubt_skim_wrapper['MICRO25_GEDEELD_TIME']
        trips_df['can_use_ebikeshare_at_ebikecarhubs'] = (trips_df.new_mode_3 > 0) & (trips_df.tt_shr_ebike3 < 9999)

        # remove the unused columns
        trips_df.drop(['tt_shr_bike1', 'tt_shr_bike2', 'tt_shr_bike3', 'tt_shr_ebike1', 'tt_shr_ebike2', 'tt_shr_ebike3'], axis=1, inplace=True)
    else:
        trips_df['can_use_bikeshare_at_carbikehubs'] = trips_df['can_use_bikeshare']
        trips_df['can_use_bikeshare_at_bikecarhubs'] = trips_df['can_use_bikeshare']
        trips_df['can_use_ebikeshare_at_carebikehubs'] = trips_df['can_use_ebikeshare']
        trips_df['can_use_ebikeshare_at_ebikecarhubs'] = trips_df['can_use_ebikeshare']


    #
    # 1. calculate the Utility using the utility functions in the csv file,
    #    e.g U(Walk_Car_bike_Acc), U(walk_car_bike_M), U(walk_car_bike_Egr), ...
    # print('== TRIP CHOOSERS ==')
    # print(trips_df.columns)
    chunk.adaptive_chunked_choosers(trips_df, chunk_size, trace_label)
    utilities = simulate.eval_utilities(spec=model_spec, choosers=trips_df, locals_d=locals_dict,
                                        trace_label='test_mode_choice', have_trace_targets=False)

    # FREE the skim from memory
    # del skim_dict
    # skim_dict.skim_data.clear()
    # print("\nSkims in the memory is freed.")

    #
    # 2. Sum the Utility per multi-modal in extra columns: such as
    #    U(walk_car_bike) = U(Walk_Car_bike_Acc) + U(walk_car_bike_M) + U(walk_car_bike_Egr)
    t0 = tracing.print_elapsed_time()
    for col in utilities.columns:
        if col[-2:] == '_M':
            # this column represents a main mode, so we compute the utility using access, main and egress
            mode_main = col[:-2]
            utilities[mode_main] = utilities[mode_main + '_M']
            utilities.drop(mode_main + '_M', axis=1, inplace=True)

            # add access utility if available
            if mode_main + '_ACC' in utilities.columns:
                utilities[mode_main] = utilities[mode_main] + utilities[mode_main + '_ACC']
                # remove those _M, _ACC and _EGR columns
                utilities.drop(mode_main + '_ACC', axis=1, inplace=True)

            # add egress utility if available
            if mode_main + '_EGR' in utilities.columns:
                utilities[mode_main] = utilities[mode_main] + utilities[mode_main + '_EGR']
                utilities.drop(mode_main + '_EGR', axis=1, inplace=True)
            # TODO 'normalize' if only 2 terms have been added? otherwise total utility lower than cases with 3 terms
            # idea: add a constant that represents a minimum utility (e.g., for ACC, minimum of all ACC utilities?)
    t0 = tracing.print_elapsed_time("Sum utilities of access, main and egress of each multimodal modes.", t0)

    # prepare the column names for each mode alternative, their name are integers
    utility_col_names = {name.upper(): convert_mode_name2id(name, TrafficMode.mode_name_map) for name in
                         multimodes_list}
    # change the column names https://note.nkmk.me/en/python-pandas-dataframe-rename/
    utilities.rename(columns=utility_col_names, inplace=True)

    # 2a. Add error-term Mu, which is for each person each mode. I.e. for the same person same mode, use the same
    # mu-sample. SAMPLE Mu term
    person_ids = trips_df['person_id'].unique()
    num_cols = len(utilities.columns)
    # Calculate the average(abs(Utility)), take K% of that value as sigma (std. deviation) for a Normal
    # distribution, the original X as sigma for eta leads to nice results. However, because we introduce anther
    # normal distributions Mu, to keep the variance the same, the new sigma is 2Y^2 = X^2 --> y = sqrt(x^2 * 0.5)
    # get all the unique person_id
    X = sigma_percent * np.nansum(np.absolute(utilities[utilities > -200].values)) / (utilities.values > -200).sum()
    sigma = round(sqrt(pow(X, 2) * 0.5), 3)
    t0 = tracing.print_elapsed_time("dynamic determined std. Deviation for Normal Distribution N(0, {0}), "
                                    "sigma=SQRT(pow({1}% of avg. utilities, 2) * 0.5).".format(sigma, sigma_percent * 100), t0)
    print("dynamic determined std. Deviation for Normal Distribution N(0, {0}), sigma=SQRT(pow({1}% of avg. utilities, 2) * 0.5).".format(sigma, sigma_percent * 100))

    if use_input_sigma:
        sigma = sigma_crn

    if not use_CRN:
        with open('sigma.txt', 'w') as f:
            f.write("dynamic determined std. Deviation for Normal Distribution N(0, {0}), sigma=SQRT(pow({1}% of avg. utilities, 2) * 0.5).".format(sigma, sigma_percent * 100))
        num_persons = len(person_ids)
        np.random.seed(seed_mu)
        mu = np.random.normal(0, sigma, num_persons * num_cols).reshape(num_persons, num_cols)
    else:
        print("Common Random Number Approach: Std. Deviation for normal error distribution = {0}".format(sigma))
        df_persons = pd.DataFrame(data=person_ids, columns=['person_id'])
        df_persons.set_index('person_id', inplace=True)
        mu = pipeline.get_rn_generator().normal_random_for_df(df_persons, 0, sigma, n=num_cols)
        # df_mu_export = pd.merge(trips_df[['person_id', 'household_id', 'tour_id','trip_num','trip_count']], df_mu,
        # left_on='person_id', right_index=True, how='left')
        # df_mu_export.to_csv('output/hh_600/df_mu_export_600.csv')

    if export_utility_to_csv:
        utilities.to_csv(output_dir + '/utilities_trips_no_mu.csv')
        t0 = tracing.print_elapsed_time("Export determinstic utilities to csv", t0)

    # merge personal and mode related error term to trips, trips having the same person_id and mode name get the
    # same Mu-sample
    df_mu = pd.DataFrame(mu, columns=utilities.columns, index=person_ids)
    df_mu = pd.merge(trips_df['person_id'], df_mu, left_on='person_id', right_index=True, how='left')
    df_mu.drop(['person_id'], axis=1, inplace=True)
    # print('== UTILITIES ==')
    # print(utilities)

    # add the Mu-term into the utilities
    utilities += df_mu
    t0 = tracing.print_elapsed_time("Add error term mu (personal preference on modes) to utilities.", t0)

    if export_utility_to_csv:
        utilities.to_csv(output_dir + '/utilities_trips.csv')
        tracing.print_elapsed_time("Export V + mu utilities to csv", t0)

    trips_df = pd.concat([trips_df, utilities], axis=1)

    # print('== TRIPS ==')
    # print(trips_df.info())
    # print(trips_df.iloc[0])

    #
    # 3. Save in the pipeline
    #
    # pipeline.replace_table("access_egress_utilities", utilities)
    # currently this saves in the folder 'access_egress_utilities' in the pipeline
    # TODO: do we want it like this, or should we just replace trips after merging??

    # # if __debug__:
    # # generate random numbers to represent utility values for modes in multimodes_list
    # utility_col_names = {name.upper(): convert_mode_name2id(name, TrafficMode.mode_name_map) for name in
    #                      multimodes_list}
    # np.random.seed(42)
    # df_rand_util = pd.DataFrame(np.random.random((trips_df.shape[0], len(multimodes_list))),
    # columns=list(utility_col_names.values()), index=trips_df.index)
    # trips_df = pd.concat([trips_df, df_rand_util], axis=1)

    #
    # 4. get the mode combinations  # configs_dir + '/data/chainset_' + str(len(multimodes_list)) + '_modes'
    #
    mode_chain_set_folder = 'chainset_' + str(len(multimodes_list)) + '_modes/'

    # make multi-modal mode choice
    # if use_feather_trips:
    #     tour_ids = trips_df.loc[trips_df.trip_mode.isin([7]), 'tour_new_id']  # ,1007001,1007002,2007001,2007002
    #     tour_ids = tour_ids.unique()
    #     trips_df = trips_df[trips_df.tour_new_id.isin(tour_ids)]

    '''we could export the trips when next time just call trip_access_egress_mode_choice_load to continue to avoid 
    long loading time of LOS matrices'''
    if export_trips_2_hdf:
        trips_df.to_hdf(output_dir + "/" + trips_file_name, 'trips')
        return

    trips_df = make_access_egress_mode_choice(output_dir, mode_chain_set_folder, utility_col_names, trips_df, enable_maas, private_car_high_priority,
                                              seed_eta_offset, sigma, use_feather_trips, use_gpu_version, dll_path, use_CRN)

    # save into the pipeline (drop the utility columns)
    trips_df.drop(list(utility_col_names.values()), axis=1, inplace=True)

    # determine which hub location is actually used
    logger.info("Create hub_location column in the trips")
    determine_hub_location(trips_df, _FAKE_ZONE_NR)

    print("Calculate CAR Travel distance (assume all trips using car) and actual car distance")
    calculate_car_distance(trips_df, skim_dict, orig_col, dest_col, period_col, "final_trip_mode")

    print("Calculate DRT actual distance")
    calculate_drt_distance(trips_df, skim_dict, orig_col, dest_col, period_col, "final_trip_mode")

    print("Calculate Bike and Ebike actual distance")
    calculate_bike_ebike_distance(trips_df, skim_dict, orig_col, dest_col, period_col, "final_trip_mode")

    # determine the actual travel distance of the selected mode
    print("Calculate Travel distance of final_trip_mode")
    calculate_travel_distance(trips_df, "travel_distance_MM", skim_dict, orig_col, dest_col, period_col, "final_trip_mode")

    print("Calculate Travel distance of trip_mode")
    calculate_travel_distance(trips_df, "travel_distance", skim_dict, orig_col, dest_col, period_col, "trip_mode")

    print("Calculate Travel time of final trip mode and private-car travel time (assume all trips using car)")
    car_time_pct = settings.get('CAR_TT_10_PCT_HIGHER', 1.0)  # 0.8 mean 80% of original travel time
    pt_time_pct = settings.get('PT_TT_10_PCT_HIGHER', 1.0)  # 0.8 mean 80% of original travel time
    bike_time_pct = settings.get('BIKE_TT_PCT', 1.0)  # 0.8 mean 80% of original travel time
    ebike_time_pct = settings.get('EBIKE_TT_PCT', 1.0)  # 0.8 mean 80% of original travel time
    drt_time_pct = settings.get('DRT_TT_PCT', 1.0)  # 0.8 mean 80% of original travel time
    print("Time factor on CAR = {0}, PT = {1}, Bike = {2}, Ebike = {3}, DRT = {4}".format(car_time_pct, pt_time_pct, bike_time_pct, ebike_time_pct, drt_time_pct))
    print("Terminal time at Car-Bike = {0} min, Car-PT = {1} min, PT-bike = {2} min".format(T_terminal_carbike, T_terminal_carpt, T_terminal_ptbike))
    calculate_travel_time(trips_df, "travel_time", skim_dict, orig_col, dest_col, period_col, "final_trip_mode", car_time_pct, pt_time_pct, bike_time_pct, ebike_time_pct, drt_time_pct, T_terminal_carbike, T_terminal_carpt, T_terminal_ptbike)

    pipeline.replace_table("trips", trips_df)

    if use_feather_trips:
        t0 = tracing.print_elapsed_time()

        df_origin = pd.merge(df_origin, trips_df[[FINAL_MODE_COL_NAME, 'hub_location']], left_index=True,
                             right_index=True, how='left')
        # trips with trip id == -2 is not in the calculation but in the output
        df_origin.loc[df_origin.trip_id == -2, FINAL_MODE_COL_NAME] = -2

        # by preparing the origin and destination are change to 1-based
        df_origin['origin'] -= 1
        df_origin['destination'] -= 1

        column_mapping = {
            'trip_origin': 'origin',
            'trip_destination': 'destination',
            # 'trip_start_time': 'depart',
            'hh_income': 'income',
            'hh_nr_of_cars': 'auto_ownership',
            'trip_transport_mode': 'trip_mode',
            'paid_work': 'pemploy',
            'gender': 'male',
            'age_person': 'age',
            'student_pt': 'student_pt',
            'urbanized': 'urbanized',
            'hh_composition': 'hhsize'
        }
        ori_colum_mapping = {value: key for key, value in column_mapping.items()}
        df_origin.rename(columns=ori_colum_mapping, inplace=True)
        df_origin.to_csv(output_dir + '/final_UTN_trips.csv')
        tracing.print_elapsed_time("Merge model choice and hub location to FEATHERS output.", t0)


def calculate_travel_distance(df_trips, distance_column, skim_dict, orig_col, dest_col, period_col, mode_col):
    """
    Total travel distance
    """
    MILE2KM = 1.609
    HUB_COL = "hub_location"
    # prepare the skims dictionary
    odt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col, dim3_key=period_col)
    # od_skim_wrapper = skim_dict.wrap(orig_col, dest_col)

    # wrapper for AutoFiets hub
    o_autoBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    autoBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)
    # wrapper for bike-Auto hub
    o_bikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    bikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for Auto ebike hub
    o_autoEBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    autoEBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for ebike-Auto hub
    o_EbikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    EbikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for AutoOV Hub
    o_carPTt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    carPTHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for OVAuto Hub
    o_PTCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    PTCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # set_df
    odt_skim_wrapper.set_df(df_trips)
    # origin -> car-bike hub -> destination
    o_autoBikeHubt_skim_wrapper.set_df(df_trips)
    autoBikeHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> car-Ebike hub -> destination
    o_autoEBikeHubt_skim_wrapper.set_df(df_trips)
    autoEBikeHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> bike-car hub -> destination
    o_bikeCarHubt_skim_wrapper.set_df(df_trips)
    bikeCarHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> Ebike-car hub -> destination
    o_EbikeCarHubt_skim_wrapper.set_df(df_trips)
    EbikeCarHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> car-PT hub -> destination
    o_carPTt_skim_wrapper.set_df(df_trips)
    carPTHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> PT-Auto Hub -> destination
    o_PTCarHubt_skim_wrapper.set_df(df_trips)
    PTCarHub_dt_skim_wrapper.set_df(df_trips)

    df_trips[mode_col] = df_trips[mode_col].str.lower()
    df_trips[distance_column] = 0
    # assume all trips is using private car
    # df_trips['car_distance_full'] = odt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips["car_distance"] = 0

    # 1. distance: walk
    df_trips.loc[(df_trips[mode_col] == 'walk'), distance_column] = odt_skim_wrapper['MICRO5_NIETGEDEELD_DIST']
     # 2. distance: bike
    df_trips.loc[(df_trips[mode_col] == 'bike') & (df_trips['can_use_bikeshare'] == 0), distance_column] = odt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike') & (df_trips['can_use_bikeshare'] == 1), distance_column] = odt_skim_wrapper['MICRO15_GEDEELD_DIST']
    # 3. distance: ebike
    df_trips.loc[(df_trips[mode_col] == 'ebike') & (df_trips['can_use_ebikeshare'] == 0), distance_column] = odt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike') & (df_trips['can_use_ebikeshare'] == 1), distance_column] = odt_skim_wrapper['MICRO25_GEDEELD_DIST']
    # 4. distance: car mode
    df_trips.loc[(df_trips[mode_col] == 'car') & (df_trips['can_use_carshare'] == 0), distance_column] = odt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'car') & (df_trips['can_use_carshare'] == 1), distance_column] = odt_skim_wrapper['PRIVE_GEDEELD_DIST']
    # specially store the car distance
    # df_trips.loc[(df_trips[mode_col] == 'car') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = odt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'car') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = odt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 5. distance: car passengier
    df_trips.loc[(df_trips[mode_col] == 'cp') & (df_trips['can_use_cpshare'] == 0), distance_column] = odt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'cp') & (df_trips['can_use_cpshare'] == 1), distance_column] = odt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']
    # 6. distance: share on demand
    df_trips.loc[(df_trips[mode_col] == 'drt'), distance_column] = odt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    # 7. distance: W PT W
    df_trips.loc[(df_trips[mode_col] == 'walk_pt_walk'), distance_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    # 8. distance: W PT W
    df_trips.loc[(df_trips[mode_col] == 'walk_pt_bike'), distance_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENFIETS_GEDEELD_DIST']
    # 9. distance: W PT W
    df_trips.loc[(df_trips[mode_col] == 'bike_pt_walk'), distance_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_FIETSLOPEN_GEDEELD_DIST']
    # 10. distance: W PT W
    df_trips.loc[(df_trips[mode_col] == 'bike_pt_bike'), distance_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_FIETSFIETS_GEDEELD_DIST']

    # 7. distance: W PT W
    df_trips.loc[(df_trips[mode_col] == 'pt'), distance_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']

    # 11. bike_car_walk
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = bikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = bikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 12. bike_cp_walk
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']

    # 13 bike-drt-walk
    df_trips.loc[(df_trips[mode_col] == 'bike_drt_walk') & (df_trips['can_use_bikeshare'] == 0), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_drt_walk') & (df_trips['can_use_bikeshare'] == 1), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 14 drt-car-walk
    df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 0), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 1), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 15
    df_trips.loc[(df_trips[mode_col] == 'drt_cp_walk') & (df_trips['can_use_cpshare'] == 0), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'drt_cp_walk') & (df_trips['can_use_cpshare'] == 1), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']

    # 16
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = EbikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = EbikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 17
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']

    # 18
    df_trips.loc[(df_trips[mode_col] == 'ebike_drt_walk') & (df_trips['can_use_ebikeshare'] == 0), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_drt_walk') & (df_trips['can_use_ebikeshare'] == 1), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 19
    df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 0), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 1), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 20
    df_trips.loc[(df_trips[mode_col] == 'pt_cp_walk') & (df_trips['can_use_cpshare'] == 0), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'pt_cp_walk') & (df_trips['can_use_cpshare'] == 1), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']

    # 21
    df_trips.loc[(df_trips[mode_col] == 'pt_drt_walk'), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 22
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = o_autoBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = o_autoBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 23
    df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 0), distance_column] = o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 1), distance_column] = o_carPTt_skim_wrapper['PRIVE_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = o_carPTt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 24
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = o_autoEBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = o_autoEBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 25
    df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 0), distance_column] = o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 1), distance_column] = o_carPTt_skim_wrapper['PRIVE_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = o_carPTt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 26
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']

    # 27
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_drt') & (df_trips['can_use_cpshare'] == 0), distance_column] = o_carPTt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_drt') & (df_trips['can_use_cpshare'] == 1), distance_column] = o_carPTt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 28
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']

    # 29
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_pt') & (df_trips['can_use_cpshare'] == 0), distance_column] = o_carPTt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_pt') & (df_trips['can_use_cpshare'] == 1), distance_column] = o_carPTt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    # 30
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_bikeshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_bikeshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    # 31
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_ebikeshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_ebikeshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    # 32
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_pt'), distance_column] = o_carPTt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']

    df_trips[distance_column] *= MILE2KM
    #if distance_column == "travel_distance_MM":
    # df_trips['car_distance'] *= MILE2KM
    # df_trips['car_distance_full'] *= MILE2KM
    #return df_trips[distance_column]


def calculate_car_distance(df_trips, skim_dict, orig_col, dest_col, period_col, mode_col):
    """
    Car distance in the trip and car distance if the trip is fully by car
    """
    MILE2KM = 1.609
    HUB_COL = "hub_location"
    # prepare the skims dictionary
    odt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col, dim3_key=period_col)
    # od_skim_wrapper = skim_dict.wrap(orig_col, dest_col)

    # wrapper for AutoFiets hub
    o_autoBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    autoBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)
    # wrapper for bike-Auto hub
    o_bikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    bikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for Auto ebike hub
    o_autoEBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    autoEBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for ebike-Auto hub
    o_EbikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    EbikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for AutoOV Hub
    o_carPTt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    carPTHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for OVAuto Hub
    o_PTCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    PTCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # set_df
    odt_skim_wrapper.set_df(df_trips)
    # origin -> car-bike hub -> destination
    o_autoBikeHubt_skim_wrapper.set_df(df_trips)
    autoBikeHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> car-Ebike hub -> destination
    o_autoEBikeHubt_skim_wrapper.set_df(df_trips)
    autoEBikeHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> bike-car hub -> destination
    o_bikeCarHubt_skim_wrapper.set_df(df_trips)
    bikeCarHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> Ebike-car hub -> destination
    o_EbikeCarHubt_skim_wrapper.set_df(df_trips)
    EbikeCarHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> car-PT hub -> destination
    o_carPTt_skim_wrapper.set_df(df_trips)
    carPTHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> PT-Auto Hub -> destination
    o_PTCarHubt_skim_wrapper.set_df(df_trips)
    PTCarHub_dt_skim_wrapper.set_df(df_trips)

    df_trips[mode_col] = df_trips[mode_col].str.lower()
    # assume all trips is using private car
    df_trips['car_distance_full'] = odt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips["car_distance"] = 0

    # 4. distance: car mode
    df_trips.loc[(df_trips[mode_col] == 'car') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = odt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'car') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = odt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 11. bike_car_walk
    # df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), distance_column] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST'] + bikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = bikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = bikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 14 drt-car-walk
    # df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 0), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 1), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 15
    # df_trips.loc[(df_trips[mode_col] == 'drt_cp_walk') & (df_trips['can_use_cpshare'] == 0), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'drt_cp_walk') & (df_trips['can_use_cpshare'] == 1), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']

    # 16
    # df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), distance_column] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST'] + EbikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = EbikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = EbikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 19
    # df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 0), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 1), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 20
    # df_trips.loc[(df_trips[mode_col] == 'pt_cp_walk') & (df_trips['can_use_cpshare'] == 0), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'pt_cp_walk') & (df_trips['can_use_cpshare'] == 1), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST']

    # 21
    # df_trips.loc[(df_trips[mode_col] == 'pt_drt_walk'), distance_column] = o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST'] + PTCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 22
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = o_autoBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = o_autoBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 23
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 0), distance_column] = o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 1), distance_column] = o_carPTt_skim_wrapper['PRIVE_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = o_carPTt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 24
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = o_autoEBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = o_autoEBikeHubt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 25
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 0), distance_column] = o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 1), distance_column] = o_carPTt_skim_wrapper['PRIVE_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 0), 'car_distance'] = o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 1), 'car_distance'] = o_carPTt_skim_wrapper['PRIVE_GEDEELD_DIST']

    # 26
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']

    # 27
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_drt') & (df_trips['can_use_cpshare'] == 0), distance_column] = o_carPTt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_drt') & (df_trips['can_use_cpshare'] == 1), distance_column] = o_carPTt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 28
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    #
    # 29
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_pt') & (df_trips['can_use_cpshare'] == 0), distance_column] = o_carPTt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_cp_pt') & (df_trips['can_use_cpshare'] == 1), distance_column] = o_carPTt_skim_wrapper['GEDEELDPRIVE_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']
    # 30
    # df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_bikeshare'] == 0), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_bikeshare'] == 1), distance_column] = o_autoBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    # 31
    # df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_ebikeshare'] == 0), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    # df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_ebikeshare'] == 1), distance_column] = o_autoEBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    # 32
    # df_trips.loc[(df_trips[mode_col] == 'walk_drt_pt'), distance_column] = o_carPTt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_DIST']

    df_trips['car_distance'] *= MILE2KM
    df_trips['car_distance_full'] *= MILE2KM


def calculate_drt_distance(df_trips, skim_dict, orig_col, dest_col, period_col, mode_col):
    """
    DRT travel distance
    """
    MILE2KM = 1.609
    HUB_COL = "hub_location"
    # prepare the skims dictionary
    odt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col, dim3_key=period_col)
    # od_skim_wrapper = skim_dict.wrap(orig_col, dest_col)

    # wrapper for AutoFiets hub
    o_autoBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    autoBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)
    # wrapper for bike-Auto hub
    o_bikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    bikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for Auto ebike hub
    o_autoEBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    autoEBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for ebike-Auto hub
    o_EbikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    EbikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for AutoOV Hub
    o_carPTt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    carPTHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for OVAuto Hub
    o_PTCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    PTCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # set_df
    odt_skim_wrapper.set_df(df_trips)
    # origin -> car-bike hub -> destination
    o_autoBikeHubt_skim_wrapper.set_df(df_trips)
    autoBikeHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> car-Ebike hub -> destination
    o_autoEBikeHubt_skim_wrapper.set_df(df_trips)
    autoEBikeHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> bike-car hub -> destination
    o_bikeCarHubt_skim_wrapper.set_df(df_trips)
    bikeCarHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> Ebike-car hub -> destination
    o_EbikeCarHubt_skim_wrapper.set_df(df_trips)
    EbikeCarHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> car-PT hub -> destination
    o_carPTt_skim_wrapper.set_df(df_trips)
    carPTHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> PT-Auto Hub -> destination
    o_PTCarHubt_skim_wrapper.set_df(df_trips)
    PTCarHub_dt_skim_wrapper.set_df(df_trips)

    df_trips[mode_col] = df_trips[mode_col].str.lower()

    df_trips["drt_distance"] = 0

    # 4. distance: car mode
    df_trips.loc[(df_trips[mode_col] == 'drt') & (df_trips['can_use_drt'] == 1), 'drt_distance'] = odt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 11. WALK_DRT_BIKE BIKE-DRT-WALK
    df_trips.loc[(df_trips[mode_col] == 'bike_drt_walk') & (df_trips['can_use_drt'] == 1), 'drt_distance'] = bikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_drt'] == 1), 'drt_distance'] = o_autoBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 14 drt-car-walk  walk-car-drt
    df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_drt'] == 1), 'drt_distance'] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_drt'] == 1), 'drt_distance'] = carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 15 drt-cp-walk  walk-cp-drt
    df_trips.loc[(df_trips[mode_col] == 'drt_cp_walk') & (df_trips['can_use_drt'] == 1), 'drt_distance'] = o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_drt') & (df_trips['can_use_drt'] == 1), 'drt_distance'] = carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 16 WALK_DRT_PT / PT-DRT-WALK
    df_trips.loc[(df_trips[mode_col] == 'pt_drt_walk'), 'drt_distance'] = PTCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_pt'), 'drt_distance'] = o_carPTt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    # 17 walk_drt_ebike / ebike-drt-walk
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_drt'] == 0), 'drt_distance'] = o_autoEBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_drt_walk') & (df_trips['can_use_drt'] == 1), 'drt_distance'] = EbikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_DIST']

    df_trips['drt_distance'] *= MILE2KM


def calculate_bike_ebike_distance(df_trips, skim_dict, orig_col, dest_col, period_col, mode_col):
    """
    ebike travel distance
    """
    MILE2KM = 1.609
    HUB_COL = "hub_location"
    # prepare the skims dictionary
    odt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col, dim3_key=period_col)
    # od_skim_wrapper = skim_dict.wrap(orig_col, dest_col)

    # wrapper for AutoFiets hub
    o_autoBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    autoBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)
    # wrapper for bike-Auto hub
    o_bikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    bikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for Auto ebike hub
    o_autoEBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    autoEBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # wrapper for ebike-Auto hub
    o_EbikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_COL, dim3_key=period_col)
    EbikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_COL, dest_key=dest_col, dim3_key=period_col)

    # set_df
    odt_skim_wrapper.set_df(df_trips)

    # origin -> car-bike hub -> destination
    o_autoBikeHubt_skim_wrapper.set_df(df_trips)
    autoBikeHub_dt_skim_wrapper.set_df(df_trips)

    # origin -> car-Ebike hub -> destination
    o_autoEBikeHubt_skim_wrapper.set_df(df_trips)
    autoEBikeHub_dt_skim_wrapper.set_df(df_trips)

    # origin -> bike-car hub -> destination
    o_bikeCarHubt_skim_wrapper.set_df(df_trips)
    bikeCarHub_dt_skim_wrapper.set_df(df_trips)

    # origin -> Ebike-car hub -> destination
    o_EbikeCarHubt_skim_wrapper.set_df(df_trips)
    EbikeCarHub_dt_skim_wrapper.set_df(df_trips)

    df_trips[mode_col] = df_trips[mode_col].str.lower()
    df_trips['ebike_distance'] = 0
    df_trips['bike_distance'] = 0

    # 1. distance: walk
     # 2. distance: bike
    df_trips.loc[(df_trips[mode_col] == 'bike') & (df_trips['can_use_bikeshare'] == 0), 'bike_distance'] = odt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike') & (df_trips['can_use_bikeshare'] == 1), 'bike_distance'] = odt_skim_wrapper['MICRO15_GEDEELD_DIST']
    # 3. distance: ebike
    df_trips.loc[(df_trips[mode_col] == 'ebike') & (df_trips['can_use_ebikeshare'] == 0), 'ebike_distance'] = odt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike') & (df_trips['can_use_ebikeshare'] == 1), 'ebike_distance'] = odt_skim_wrapper['MICRO25_GEDEELD_DIST']
    # 4. distance: car mode

    # 5. distance: car passengier
    # 6. distance: share on demand
    # 7. distance: W PT W
    # 8. distance: W PT W
    # 9. distance: W PT W
    # 10. distance: W PT W

    # 11. bike_car_walk
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 0), 'bike_distance'] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 1), 'bike_distance'] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST']

    # 12. bike_cp_walk
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 0), 'bike_distance'] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 1), 'bike_distance'] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST']

    # 13 bike-drt-walk
    df_trips.loc[(df_trips[mode_col] == 'bike_drt_walk') & (df_trips['can_use_bikeshare'] == 0), 'bike_distance'] = o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'bike_drt_walk') & (df_trips['can_use_bikeshare'] == 1), 'bike_distance'] = o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_DIST']

    # 15 drt-cp-walk

    # 16 ebike_car_walk
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 0), 'ebike_distance'] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 1), 'ebike_distance'] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST']

    # 17 ebike_cp_walk
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 0), 'ebike_distance'] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 1), 'ebike_distance'] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST']

    # 18 ebike_drt_walk
    df_trips.loc[(df_trips[mode_col] == 'ebike_drt_walk') & (df_trips['can_use_ebikeshare'] == 0), 'ebike_distance'] = o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'ebike_drt_walk') & (df_trips['can_use_ebikeshare'] == 1), 'ebike_distance'] = o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_DIST']

    # 19 pt-car-walk

    # 20 pt-cp-walk

    # 21 pt-drt-walk

    # 22 walk-car-bike
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 0), 'bike_distance'] = autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 1), 'bike_distance'] = autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']

    # 23 walk-car-drt

    # 24 walk-car-ebike
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 0), 'ebike_distance'] = autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 1), 'ebike_distance'] = autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']

    # 25 walk_car_pt

    # 26 walk-cp-bike
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 0), 'bike_distance'] = autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 1), 'bike_distance'] = autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']

    # 27 walk_cp_drt

    # 28
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 0), 'ebike_distance'] = autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 1), 'ebike_distance'] = autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']

    # 29 walk_cp_pt
    # 30 walk_drt_bike
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_bikeshare'] == 0), 'bike_distance'] = autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_bikeshare'] == 1), 'bike_distance'] = autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_DIST']
    # 31 walk_drt_ebike
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_ebikeshare'] == 0), 'ebike_distance'] = autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_DIST']
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_ebikeshare'] == 1), 'ebike_distance'] = autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_DIST']
    # 32 walk_drt_pt

    df_trips['ebike_distance'] *= MILE2KM
    df_trips['bike_distance'] *= MILE2KM


def calculate_travel_time(df_trips, tt_column, skim_dict, orig_col, dest_col, period_col, mode_col,
                          car_time_pct=1.0, pt_time_pct=1.0, bike_time_pct=1.0, ebike_time_pct=1.0, drt_time_pct=1.0,
                          T_terminal_carbike=0, T_terminal_carpt=0, T_terminal_ptbike=0):
    """
    Calculate the total travel time and also travel time if the trip is done fully by car
    """
    # prepare the skims dictionary
    odt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col, dim3_key=period_col)
    # od_skim_wrapper = skim_dict.wrap(orig_col, dest_col)

    # wrapper for AutoFiets hub
    o_autoBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_AUTO_BIKE, dim3_key=period_col)
    autoBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_AUTO_BIKE, dest_key=dest_col, dim3_key=period_col)
    # wrapper for bike-Auto hub
    o_bikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_BIKE_AUTO, dim3_key=period_col)
    bikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_BIKE_AUTO, dest_key=dest_col, dim3_key=period_col)

    # wrapper for Auto ebike hub
    o_autoEBikeHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_AUTO_EBIKE, dim3_key=period_col)
    autoEBikeHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_AUTO_EBIKE, dest_key=dest_col, dim3_key=period_col)

    # wrapper for ebike-Auto hub
    o_EbikeCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_EBIKE_AUTO, dim3_key=period_col)
    EbikeCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_EBIKE_AUTO, dest_key=dest_col, dim3_key=period_col)

    # wrapper for AutoOV Hub
    o_carPTt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_AUTO_PT, dim3_key=period_col)
    carPTHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_AUTO_PT, dest_key=dest_col, dim3_key=period_col)

    # wrapper for OVAuto Hub
    o_PTCarHubt_skim_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=HUB_PT_AUTO, dim3_key=period_col)
    PTCarHub_dt_skim_wrapper = skim_dict.wrap_3d(orig_key=HUB_PT_AUTO, dest_key=dest_col, dim3_key=period_col)

    # set_df
    odt_skim_wrapper.set_df(df_trips)
    # origin -> car-bike hub -> destination
    o_autoBikeHubt_skim_wrapper.set_df(df_trips)
    autoBikeHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> car-Ebike hub -> destination
    o_autoEBikeHubt_skim_wrapper.set_df(df_trips)
    autoEBikeHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> bike-car hub -> destination
    o_bikeCarHubt_skim_wrapper.set_df(df_trips)
    bikeCarHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> Ebike-car hub -> destination
    o_EbikeCarHubt_skim_wrapper.set_df(df_trips)
    EbikeCarHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> car-PT hub -> destination
    o_carPTt_skim_wrapper.set_df(df_trips)
    carPTHub_dt_skim_wrapper.set_df(df_trips)
    # origin -> PT-Auto Hub -> destination
    o_PTCarHubt_skim_wrapper.set_df(df_trips)
    PTCarHub_dt_skim_wrapper.set_df(df_trips)

    df_trips[mode_col] = df_trips[mode_col].str.lower()
    df_trips[tt_column] = 0

    # Get the travel time assume each trip is using private car
    df_trips['car_tt_full'] = odt_skim_wrapper['PRIVE_NIETGEDEELD_TIME']

    # 1. time: walk
    df_trips.loc[(df_trips[mode_col] == 'walk'), tt_column] = odt_skim_wrapper['MICRO5_NIETGEDEELD_TIME']
    # 2. time: bike
    df_trips.loc[(df_trips[mode_col] == 'bike') & (df_trips['can_use_bikeshare'] == 0), tt_column] = odt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'bike') & (df_trips['can_use_bikeshare'] == 1), tt_column] = odt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct
    # 3. time: ebike
    df_trips.loc[(df_trips[mode_col] == 'ebike') & (df_trips['can_use_ebikeshare'] == 0), tt_column] = odt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'ebike') & (df_trips['can_use_ebikeshare'] == 1), tt_column] = odt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct
    # 4. time: car mode
    df_trips.loc[(df_trips[mode_col] == 'car') & (df_trips['can_use_carshare'] == 0), tt_column] = odt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'car') & (df_trips['can_use_carshare'] == 1), tt_column] = odt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct

    # 5. time: car passengier
    df_trips.loc[(df_trips[mode_col] == 'cp') & (df_trips['can_use_cpshare'] == 0), tt_column] = odt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'cp') & (df_trips['can_use_cpshare'] == 1), tt_column] = odt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME']
    # 6. time: share on demand
    df_trips.loc[(df_trips[mode_col] == 'drt'), tt_column] = odt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct
    # 7. time: W PT W
    df_trips.loc[(df_trips[mode_col] == 'walk_pt_walk'), tt_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct
    # 8. time: W PT W
    df_trips.loc[(df_trips[mode_col] == 'walk_pt_bike'), tt_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENFIETS_GEDEELD_TIME'] * pt_time_pct + T_terminal_ptbike
    # 9. time: W PT W
    df_trips.loc[(df_trips[mode_col] == 'bike_pt_walk'), tt_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_FIETSLOPEN_GEDEELD_TIME'] * pt_time_pct + T_terminal_ptbike
    # 10. time: W PT W
    df_trips.loc[(df_trips[mode_col] == 'bike_pt_bike'), tt_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_FIETSFIETS_GEDEELD_TIME'] * pt_time_pct + 2.0 * T_terminal_ptbike

    # 7. time: W PT W
    df_trips.loc[(df_trips[mode_col] == 'pt'), tt_column] = odt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct

    # 11. bike_car_walk
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'bike_car_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct

    # 12. bike_cp_walk
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'bike_cp_walk') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME']

    # 13 bike-drt-walk
    df_trips.loc[(df_trips[mode_col] == 'bike_drt_walk') & (df_trips['can_use_bikeshare'] == 0), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct
    df_trips.loc[(df_trips[mode_col] == 'bike_drt_walk') & (df_trips['can_use_bikeshare'] == 1), tt_column] = T_terminal_carbike + o_bikeCarHubt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct + bikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct

    # 14 drt-car-walk
    df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'drt_car_walk') & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct

    # 15
    df_trips.loc[(df_trips[mode_col] == 'drt_cp_walk') & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'drt_cp_walk') & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME']

    # 16
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'ebike_car_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct

    # 17
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'ebike_cp_walk') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME']

    # 18
    df_trips.loc[(df_trips[mode_col] == 'ebike_drt_walk') & (df_trips['can_use_ebikeshare'] == 0), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct
    df_trips.loc[(df_trips[mode_col] == 'ebike_drt_walk') & (df_trips['can_use_ebikeshare'] == 1), tt_column] = T_terminal_carbike + o_EbikeCarHubt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct + EbikeCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct

    # 19
    df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct + PTCarHub_dt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct
    df_trips.loc[(df_trips[mode_col] == 'pt_car_walk') & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct + PTCarHub_dt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct

    # 20
    df_trips.loc[(df_trips[mode_col] == 'pt_cp_walk') & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME']
    df_trips.loc[(df_trips[mode_col] == 'pt_cp_walk') & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct + PTCarHub_dt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME']

    # 21
    df_trips.loc[(df_trips[mode_col] == 'pt_drt_walk'), tt_column] = T_terminal_carpt + o_PTCarHubt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct + PTCarHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct

    # 22
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_car_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct

    # 23
    df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_car_drt') & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct

    # 24
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_car_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct

    # 25
    df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 0), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['PRIVE_NIETGEDEELD_TIME'] * car_time_pct + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_car_pt') & (df_trips['can_use_carshare'] == 1), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['PRIVE_GEDEELD_TIME'] * car_time_pct + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct

    # 26
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME'] + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_bike') & (df_trips['can_use_bikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME'] + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct

    # 27
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_drt') & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_drt') & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME'] + carPTHub_dt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct

    # 28
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 0) & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME'] + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_ebike') & (df_trips['can_use_ebikeshare'] == 1) & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME'] + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct

    # 29
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_pt') & (df_trips['can_use_cpshare'] == 0), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['GEDEELDPRIVE_NIETGEDEELD_TIME'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_cp_pt') & (df_trips['can_use_cpshare'] == 1), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['GEDEELDPRIVE_GEDEELD_TIME'] + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct
    # 30
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_bikeshare'] == 0), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + autoBikeHub_dt_skim_wrapper['MICRO15_NIETGEDEELD_TIME'] * bike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_bike') & (df_trips['can_use_bikeshare'] == 1), tt_column] = T_terminal_carbike + o_autoBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + autoBikeHub_dt_skim_wrapper['MICRO15_GEDEELD_TIME'] * bike_time_pct
    # 31
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_ebikeshare'] == 0), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + autoEBikeHub_dt_skim_wrapper['MICRO25_NIETGEDEELD_TIME'] * ebike_time_pct
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_ebike') & (df_trips['can_use_ebikeshare'] == 1), tt_column] = T_terminal_carbike + o_autoEBikeHubt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + autoEBikeHub_dt_skim_wrapper['MICRO25_GEDEELD_TIME'] * ebike_time_pct
    # 32
    df_trips.loc[(df_trips[mode_col] == 'walk_drt_pt'), tt_column] = T_terminal_carpt + o_carPTt_skim_wrapper['GEDEELDONDEMAND_GEDEELD_TIME'] * drt_time_pct + carPTHub_dt_skim_wrapper['GEDEELDTRADITIONEEL_LOPENLOPEN_GEDEELD_TIME'] * pt_time_pct


def determine_hub_location(df_output, fake_hub_nr):
    """
    make final hub-location depends on the multi-modal-mode, this step happens after the multi-mmodal choice
    has been made.
    :param df_output:
    :param is_delft_area: indicate if it is a Delft area or not
    :return:
    """
    # The value in the Hub skims is 1-based!
    offset = 0
    df_output['hub_location'] = fake_hub_nr

    # hubAutoFiets
    mode_list = ['walk_car_bike', 'walk_cp_bike', 'walk_drt_bike']
    df_output.loc[df_output[FINAL_MODE_COL_NAME_STR].isin(mode_list), 'hub_location'] = df_output[HUB_AUTO_BIKE] - offset

    # hub BikeCar
    mode_list = ['bike_car_walk', 'bike_cp_walk', 'bike_drt_walk']
    df_output.loc[df_output[FINAL_MODE_COL_NAME_STR].isin(mode_list), 'hub_location'] = df_output[HUB_BIKE_AUTO] - offset

    # hubAutoEbike
    mode_list = ['walk_car_ebike', 'walk_cp_ebike', 'walk_drt_ebike']
    df_output.loc[df_output[FINAL_MODE_COL_NAME_STR].isin(mode_list), 'hub_location'] = df_output[HUB_AUTO_EBIKE] - offset

    # hub EbikeCar
    mode_list = ['ebike_car_walk', 'ebike_cp_walk', 'ebike_drt_walk']
    df_output.loc[df_output[FINAL_MODE_COL_NAME_STR].isin(mode_list), 'hub_location'] = df_output[HUB_EBIKE_AUTO] - offset

    # hub Car OV
    mode_list = ['walk_car_drt', 'walk_car_pt', 'walk_cp_drt', 'walk_cp_pt', 'drt_pt_walk', 'walk_drt_pt']
    df_output.loc[df_output[FINAL_MODE_COL_NAME_STR].isin(mode_list), 'hub_location'] = df_output[HUB_AUTO_PT] - offset

    # hub OV-car
    mode_list = ['drt_car_walk', 'pt_car_walk', 'drt_cp_walk', 'pt_cp_walk', 'walk_pt_drt', 'pt_drt_walk']
    df_output.loc[df_output[FINAL_MODE_COL_NAME_STR].isin(mode_list), 'hub_location'] = df_output[HUB_PT_AUTO] - offset

# if __name__ == "__main__":
#     df_data = pd.DataFrame({"primary_purpose": ['atwork', 'atwork', 'eat', 'work', 'work', 'shop'],
#                             "person_id": [1, 1, 1, 1, 1, 1], "tour_id": [33, 33, 11, 11, 11, 11]})
#     merge_atwork_tour_with_worktour(df_data)

# https://kanoki.org/2019/07/17/pandas-how-to-replace-values-based-on-conditions/


# An alternative for faster debug: however, large dataset is really SLOW to debug
@inject.step()
def trip_access_egress_mode_choice_load(output_dir, settings, trips, tours_merged, persons, households):
    """
    trip_access_egress_mode_choice - make access and egress mode choice
    The trip mode made in trip_mode_choice step is used as main mode, this model will add access mode and egress mode
    choice.

    This model first load the utilities which are calculated by previous run, the user need to specify the Sigma value
    :param settings: global settings
    :return:
    """
    trace_label = 'trip_access_egress_mode_choice_load'
    # configs_dir = inject.get_injectable('configs_dir')

    # get the enable_maas from settings.yaml
    enable_maas = settings.get('enable_maas', 0)
    # check if we need to use FEATHERS output
    use_feather_trips = settings.get('use_FEATHERS_trips', 0)
    seed_eta_offset = settings.get('SEED_OFFSET_ETA', 0)
    delft_area = settings.get('DELFT_AREA', 0)

    # GPU based dll for multimodal mode choice
    use_gpu_version = settings.get('USE_GPU', 0)
    trips_file_name = settings.get('EXPORT_TRIPS_NAME')
    dll_path = settings.get('PATH_DLL_ACC_EGR_MODEL')
    logger.info("DLL path" + dll_path)
    logger.info("MaaS enabled = {0}, Parking factor = {1}".format(enable_maas, settings.get('parking_factor')))
    sigma = 0.193
    logger.info("The specified sigma for error term per trip-person-mode is {0}".format(sigma))

    # get the list of mode names (lowercase) specified in the YAML file
    multimodes_list = read_modes_from_settings(config.read_model_settings('trip_access_egress_choice.yaml'))

    utility_col_names = {name.upper(): convert_mode_name2id(name, TrafficMode.mode_name_map) for name in
                         multimodes_list}

    #
    # 4. get the mode combinations  # configs_dir + '/data/chainset_' + str(len(multimodes_list)) + '_modes'
    #
    mode_chain_set_folder = 'chainset_' + str(len(multimodes_list)) + '_modes/'

    trips_df = pd.read_hdf(output_dir + '/' + trips_file_name, 'trips')
    logger.info("Load " + trips_file_name)
    # tour_ids = trips_df.loc[trips_df.trip_mode.isin([7]), 'tour_new_id']  # ,1007001,1007002,2007001,2007002
    # tour_ids = tour_ids.unique()
    # trips_df = trips_df[trips_df.tour_new_id.isin(tour_ids)]

    # make multi-modal mode choice
    trips_df = make_access_egress_mode_choice(mode_chain_set_folder, utility_col_names, trips_df, enable_maas,
                                              seed_eta_offset, sigma, use_feather_trips, use_gpu_version, dll_path)

    # save into the pipeline (drop the utility columns)
    trips_df.drop(list(utility_col_names.values()), axis=1, inplace=True)

    # determine which hub location is actually used
    logger.info("Create hub_location column in the trips")
    determine_hub_location(trips_df, fake_hub_nr=7787)
    pipeline.replace_table("trips", trips_df)
    if use_feather_trips:
        t0 = tracing.print_elapsed_time()

        # df_origin = pd.merge(df_origin, trips_df[[FINAL_MODE_COL_NAME, 'hub_location']], left_index=True,
        #                      right_index=True, how='left')
        # trips with trip id == -2 is not in the calculation but in the output
        # df_origin.loc[df_origin.trip_id == -2, FINAL_MODE_COL_NAME] = -2

        # by preparing the origin and destination are change to 1-based
        # df_origin['origin'] -= 1
        # df_origin['destination'] -= 1

        column_mapping = {
            'trip_origin': 'origin',
            'trip_destination': 'destination',
            # 'trip_start_time': 'depart',
            'hh_income': 'income',
            'hh_nr_of_cars': 'auto_ownership',
            'trip_transport_mode': 'trip_mode',
            'paid_work': 'pemploy',
            'gender': 'male',
            'age_person': 'age',
            'student_pt': 'student_pt',
            'urbanized': 'urbanized',
            'hh_composition': 'hhsize'
        }
        ori_colum_mapping = {value: key for key, value in column_mapping.items()}
        # df_origin.rename(columns=ori_colum_mapping, inplace=True)
        trips_df.to_csv(output_dir + '/final_UTN_trips.csv')
        tracing.print_elapsed_time("Merge model choice and hub location to FEATHERS output.", t0)