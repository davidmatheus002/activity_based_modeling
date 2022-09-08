# ActivitySim
# See full license in LICENSE.txt.

import logging

import pandas as pd
import numpy as np

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import expressions

from activitysim.core import assign
from activitysim.core import los

from activitysim.core.util import assign_in_place

from .util.mode import mode_choice_simulate
from .util import estimation


logger = logging.getLogger(__name__)


@inject.step()
def trip_mode_choice(
        trips,
        tours_merged,
        network_los,
        chunk_size, trace_hh_id):
    """
    Trip mode choice - compute trip_mode (same values as for tour_mode) for each trip.

    Modes for each primary tour putpose are calculated separately because they have different
    coefficient values (stored in trip_mode_choice_coefficients.csv coefficient file.)

    Adds trip_mode column to trip table
    """
    trace_label = 'trip_mode_choice'
    model_settings_file_name = 'trip_mode_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get('MODE_CHOICE_LOGSUM_COLUMN_NAME')
    mode_column_name = 'trip_mode'

    trips_df = trips.to_frame()
    logger.info("Running %s with %d trips", trace_label, trips_df.shape[0])

    tours_merged = tours_merged.to_frame()
    tours_merged = tours_merged[model_settings['TOURS_MERGED_CHOOSER_COLUMNS']]

    tracing.print_summary('primary_purpose',
                          trips_df.primary_purpose, value_counts=True)

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips_df,
        tours_merged,
        left_on='tour_id',
        right_index=True,
        how="left")
    assert trips_merged.index.equals(trips.index)

    # setup skim keys
    assert ('trip_period' not in trips_merged)
    trips_merged['trip_period'] = network_los.skim_time_period_label(trips_merged.depart)

    orig_col = 'origin'
    dest_col = 'destination'

    constants = {}
    constants.update(config.get_model_constants(model_settings))
    constants.update({
        'ORIGIN': orig_col,
        'DESTINATION': dest_col
    })

    skim_dict = network_los.get_default_skim_dict()

    odt_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col,
                                               dim3_key='trip_period')
    dot_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=dest_col, dest_key=orig_col,
                                               dim3_key='trip_period')
    od_skim_wrapper = skim_dict.wrap('origin', 'destination')

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?
        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(orig_key=orig_col, dest_key=dest_col,
                                           tod_key='trip_period', segment_key='demographic_segment',
                                           cache_choices=True,
                                           trace_label=trace_label, tag='tvpb_logsum_odt')
        skims.update({
            'tvpb_logsum_odt': tvpb_logsum_odt,
            # 'tvpb_logsum_dot': tvpb_logsum_dot
        })

        # TVPB constants can appear in expressions
        constants.update(network_los.setting('TVPB_SETTINGS.tour_mode_choice.CONSTANTS'))

    estimator = estimation.manager.begin_estimation('trip_mode_choice')
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_coefficients_template(model_settings=model_settings)
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    nest_spec = config.get_logit_model_settings(model_settings)

    choices_list = []
    for primary_purpose, trips_segment in trips_merged.groupby('primary_purpose'):

        segment_trace_label = tracing.extend_trace_label(trace_label, primary_purpose)

        logger.info("trip_mode_choice tour_type '%s' (%s trips)" %
                    (primary_purpose, len(trips_segment.index), ))

        # name index so tracing knows how to slice
        assert trips_segment.index.name == 'trip_id'

        if network_los.zone_system == los.THREE_ZONE:
            tvpb_logsum_odt.extend_trace_label(primary_purpose)
            # tvpb_logsum_dot.extend_trace_label(primary_purpose)

        coefficients = simulate.get_segment_coefficients(model_settings, primary_purpose)

        locals_dict = {}
        locals_dict.update(constants)
        locals_dict.update(coefficients)

        expressions.annotate_preprocessors(
            trips_segment, locals_dict, skims,
            model_settings, segment_trace_label)

        if estimator:
            # write choosers after annotation
            estimator.write_choosers(trips_segment)

        locals_dict.update(skims)

        choices = mode_choice_simulate(
            choosers=trips_segment,
            spec=simulate.eval_coefficients(model_spec, coefficients, estimator),
            nest_spec=simulate.eval_nest_coefficients(nest_spec, coefficients, segment_trace_label),
            skims=skims,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            mode_column_name=mode_column_name,
            logsum_column_name=logsum_column_name,
            trace_label=segment_trace_label,
            trace_choice_name='trip_mode_choice',
            estimator=estimator)

        if trace_hh_id:
            # trace the coefficients
            tracing.trace_df(pd.Series(locals_dict),
                             label=tracing.extend_trace_label(segment_trace_label, 'constants'),
                             transpose=False,
                             slicer='NONE')

            # so we can trace with annotations
            assign_in_place(trips_segment, choices)

            tracing.trace_df(trips_segment,
                             label=tracing.extend_trace_label(segment_trace_label, 'trip_mode'),
                             slicer='tour_id',
                             index_label='tour_id',
                             warn_if_empty=True)

        choices_list.append(choices)

    choices_df = pd.concat(choices_list)

    # add cached tvpb_logsum tap choices for modes specified in tvpb_mode_path_types
    if network_los.zone_system == los.THREE_ZONE:

        tvpb_mode_path_types = model_settings.get('tvpb_mode_path_types')
        for mode, path_type in tvpb_mode_path_types.items():

            skim_cache = tvpb_logsum_odt.cache[path_type]

            for c in skim_cache:
                dest_col = c
                if dest_col not in choices_df:
                    choices_df[dest_col] = np.nan if pd.api.types.is_numeric_dtype(skim_cache[c]) else ''
                choices_df[dest_col].where(choices_df[mode_column_name] != mode, skim_cache[c], inplace=True)

    if estimator:
        estimator.write_choices(choices_df.trip_mode)
        choices_df.trip_mode = estimator.get_survey_values(choices_df.trip_mode, 'trips', 'trip_mode')
        estimator.write_override_choices(choices_df.trip_mode)
        estimator.end_estimation()

    # update trips table with choices (and potionally logssums)
    trips_df = trips.to_frame()
    assign_in_place(trips_df, choices_df)

    tracing.print_summary('trip_modes',
                          trips_merged.tour_mode, value_counts=True)

    tracing.print_summary('trip_mode_choice choices',
                          trips_df[mode_column_name], value_counts=True)

    assert not trips_df[mode_column_name].isnull().any()

    t1 = tracing.print_elapsed_time()
    # assign new tour id and trip_seq
    trips_df['tour_new_id'] = trips_df['tour_id']
    trips_df['trip_seq'] = trips_df.groupby(["tour_id"]).cumcount() + 1
    merge_atwork_tour_with_worktour(tours_merged, trips_df)
    tracing.print_elapsed_time("merge the at-work sub-tour into its parent tour", t1)

    pipeline.replace_table("trips", trips_df)

    if trace_hh_id:
        tracing.trace_df(trips_df,
                         label=tracing.extend_trace_label(trace_label, 'trip_mode'),
                         slicer='trip_id',
                         index_label='trip_id',
                         warn_if_empty=True)

def merge_atwork_tour_with_worktour(tours, trips):
    """
     first prepare a dataframe like
     (index:subtour's tour_id) |  parent_tour_id  | subtour_count | trip_seq (the first trip in parent tour) | origin(of the subtour)  | destination(of the parent tour having the same destiation as subtour's origin)
     31533                        31568             2               1                                          1948                      1948
     7589                         7624              3               3 (the third trip in the parent tour)      166                       166
     """
    start = tracing.print_elapsed_time()
    # get the sub-tours' parent tour id: index is the subtours' tour_id
    parent_tour_ids = tours.loc[~tours.parent_tour_id.isnull(), 'parent_tour_id']
    num_subtours = len(parent_tour_ids)
    start = tracing.print_elapsed_time("# of tours having parent tour is %s " % num_subtours, start)

    # get the number of trips in each sub tours (atwork tours), index is tour_id
    trip_count_per_atwork_tour = trips[trips.tour_id.isin(parent_tour_ids.index.tolist())].groupby(
        'tour_id').size().to_frame('subtour_count')
    # print("time to get sub-tour trip-count is %s seconds" % (time() - start))

    # merge to get their parent_tour_id
    trip_count_per_atwork_tour = pd.merge(trip_count_per_atwork_tour, parent_tour_ids, on='tour_id')
    # print("time to get trips' parent tour id  is %s seconds" % (time() - start))

    #
    # a. get the parent tour's first trip's index, whose destination = subtour's origin, the index is the (sub-tour) tour_id
    #
    # get the subtour trips origin
    trip_count_per_atwork_tour = pd.merge(trip_count_per_atwork_tour, trips.loc[
        trips.tour_id.isin(parent_tour_ids.index.tolist()), ['tour_id', 'origin', 'destination']], left_index=True,
                                          right_on='tour_id', how='left')
    trip_count_per_atwork_tour = trip_count_per_atwork_tour.groupby('tour_id').nth(0)
    trip_count_per_atwork_tour.reset_index(inplace=True)
    # get parent tour's destination
    trip_count_per_atwork_tour = pd.merge(trip_count_per_atwork_tour, trips.loc[
        trips.tour_id.isin(parent_tour_ids.tolist()), ['tour_id', 'destination', 'trip_seq']], left_on='parent_tour_id',
                                          right_on='tour_id', suffixes=[None, '_parent'])
    # match the subtour's trip origin to parent tour's trip destination, keep only the first match in each subtour
    trip_count_per_atwork_tour = trip_count_per_atwork_tour[
        trip_count_per_atwork_tour.origin == trip_count_per_atwork_tour.destination_parent].drop_duplicates(
        subset=['tour_id'])
    trip_count_per_atwork_tour.set_index('tour_id', inplace=True)
    assert num_subtours == len(trip_count_per_atwork_tour)
    start = tracing.print_elapsed_time("get parent tours first trip sequence id", start)

    #
    # b. now start to update trip_seq for both subtour trips and parent tour trips
    #

    # only the trips having parent tour are remained in the dataframe, index=trip_id
    subtours_trips = pd.merge(trips, trip_count_per_atwork_tour['trip_seq'], left_on='tour_id', right_index=True,
                              suffixes=[None, '_parent'])

    # update subtours trips index
    subtours_trips['trip_seq'] = subtours_trips['trip_seq'] + subtours_trips['trip_seq_parent']
    start = tracing.print_elapsed_time("update sub-tours trip_seq", start)

    # update parent trips' index, except the trips before the sub-tour
    # keep the trip_id as a column, will recover as index later
    parent_tours_trips = trips[trips.tour_id.isin(parent_tour_ids.tolist())].reset_index()
    parent_tours_trips = pd.merge(parent_tours_trips,
                                  trip_count_per_atwork_tour[['subtour_count', 'parent_tour_id', 'trip_seq']],
                                  left_on='tour_id', right_on='parent_tour_id', suffixes=[None, '_parent'])
    # mark only parent tour's trips having trip_seq after the sub-tour trips
    mask = parent_tours_trips.trip_seq > parent_tours_trips.trip_seq_parent
    # print("time to create mask to ignore the parent tour's first trip is %s seconds" % (time() - start))
    parent_tours_trips.loc[mask, 'trip_seq'] = parent_tours_trips.loc[mask, 'trip_seq'] + parent_tours_trips.loc[
        mask, 'subtour_count']
    parent_tours_trips.set_index('trip_id', inplace=True)
    start = tracing.print_elapsed_time("update parent-tours trip_seq", start)

    #
    #  c. finally update trips
    #
    trips.loc[subtours_trips.index, 'trip_seq'] = subtours_trips['trip_seq']
    trips.loc[parent_tours_trips.index, 'trip_seq'] = parent_tours_trips['trip_seq']
    tracing.print_elapsed_time("update original trips dataframe triq_seq", start)


    # DEPRECATED: VERY SLOW
    # # select all the atwork subtour and their work tour
    # # 1. get all the sub-tours: having a valid parent_tour_id
    # subtours_and_its_parent_tour = tours_df.loc[~tours_df.parent_tour_id.isnull(), ['parent_tour_id']]
    # logger.info("# of tours having parent tour is %s " % (len(subtours_and_its_parent_tour)))
    #
    # # 2. loop through each sub tour and its parent tour
    # for subtour_id, o_row in subtours_and_its_parent_tour.iterrows():
    #     parent_tour_id = o_row['parent_tour_id']
    #     atwork_tours = trips_df.tour_id == subtour_id
    #     work_tours = trips_df.tour_id == parent_tour_id
    #
    #     # a. get the trip index of the first work trip in work tour
    #     work_trip_idx = trips_df[work_tours].trip_seq.iloc[0]
    #
    #     # b. assign work tour id to atwork-subtour
    #     trips_df.loc[atwork_tours, "tour_new_id"] = parent_tour_id
    #
    #     # c. get number of trips in atwork tour
    #     offset = len(trips_df[atwork_tours])
    #
    #     # d. update atwork trip's trip_idx
    #     trips_df.loc[atwork_tours, "trip_seq"] = range(work_trip_idx + 1, work_trip_idx + 1 + offset)
    #
    #     # e. update parent tour trips after the first 'work' trip
    #     trips_df.loc[work_tours & (trips_df.trip_seq > work_trip_idx), "trip_seq"] += offset
