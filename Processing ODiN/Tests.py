import pandas as pd
import openmatrix as omx
import numpy as np
# import functions as fn
import random

# print('LOS_AS_2016_basisjaar columns:')
# df = pd.read_csv('L:/UserData/David/ABM/Processing LOS/raw/LOS_AS_2016_basisjaar.csv')
# print(df.columns)
# print('')
# print('LOS_newmodalities_AS_modified columns:')
# df = pd.read_csv('L:/UserData/Erwin/LOS/LOS_newmodalities_AS_modified.csv')
# print(df.columns)

# for col in cols:
##     df = pd.read_csv('L:/UserData/David/ABM/Processing LOS/base_case_los_reformatted/LOS_newmodalities_RD_gedeeldondemand_gedeeld_afstand.csv')
#     df = pd.read_csv(f'L:/UserData/David/ABM/Processing LOS/base_case_los_reformatted/LOS_newmodalities_RD_{col}.csv')
#     print(col)
#     print(df.shape)
#     print('')
# source_file_name = 'L:/UserData/David/ABM/Processing LOS/raw/LOS_newmodalities_RD_modified.csv'
# # a_skim = pd.read_csv(source_file_name, header=None, delimiter=",", comment="#", decimal='.')
# df = pd.read_csv(source_file_name, delimiter=",", comment="#", decimal='.')
# print(df.origin.unique())
# df = df[['origin', 'destination', 'gedeeldondemand_gedeeld_afstand']]
#print(df)

# a_skim = df[]
# print(a_skim.shape)
# print(a_skim)

# # Test matrices
# test_skim = omx.open_file("L:/UserData/David/ABM/data_base_case/skims_MRDH2_new.omx", 'r')
# for matrix in test_skim.list_matrices():
#     print()
#     print(matrix)
#     print('----------------------------------------------------')
#     df = pd.DataFrame(np.array(test_skim[matrix]))
#     print(f'Columns: {df.columns.values}')
#     # df.index = np.arange(1, len(df) + 1)
#     print(f'Indices: {df.index.values}')
#     # print(df)
#     print(f'The minimum value in skim is {df.to_numpy().min()}')
#     print(f'The maximum value in skim is {df.to_numpy().max()}')
#     print(f'NaN values in skim: {df.isnull().sum().sum()}')

# df = pd.read_csv("L:/UserData/David/ABM/data_base_case/persons.csv")
# df = pd.read_csv("L:/UserData/David/case_delft/data/persons.csv")
# print(df.info())
# df2 = pd.read_csv("L:/UserData/David/ABM/data_base_case/persons.csv")
# print(df2.info())
# print(df2.home_zone_id.unique())

# n_zones = 7786
# # Test matrices
# test_skim = omx.open_file("L:/UserData/David/ABM/data_base_case/skims_MRDH2_new.omx", 'r')
# df = pd.DataFrame(np.array(test_skim[test_skim.list_matrices()[0]]))
# print()
# print(f'MRDH SKIMS')
# print('-------------------------------------------------------------------------------------------')
# print(f' Complete zones: {(df.columns.values == [i for i in range(n_zones + 1)]).all()}')
# print(f'Skims columns: {df.columns.values}')
#
# # df.index = np.arange(1, len(df) + 1)
# print(f'Skims indices: {df.index.values}')
# print(f'Number of origins and destinations: {len(df.columns.values) - 1}')
# print(df)
#
# print(df[614][0])
# # print(df)


#
# test_skim = omx.open_file("L:/UserData/David/case_delft/data/skims_delft_new.omx", 'r')
# df = pd.DataFrame(np.array(test_skim[test_skim.list_matrices()[0]]))
# print(df)

# # Test survey data
# df = pd.read_csv("L:/UserData/David/ABM/data_base_case/survey_data/survey_tours.csv")
# origins = df.origin.unique().tolist()
# origins.sort()
# destinations = df.destination.unique().tolist()
# destinations.sort()
# print()
# print('SURVEY TOURS')
# print('-------------------------------------------------------------------------------------------')
# print(f'Survey tour origins: {origins}')
# print(f'Number of origins: {len(origins)}')
# print(f'Survey tour destinations: {destinations}')
# print(f'Number of destinations: {len(destinations)}')
#
# df = pd.read_csv("L:/UserData/David/ABM/data_base_case/survey_data/survey_trips.csv")
# origins = df.origin.unique().tolist()
# origins.sort()
# destinations = df.destination.unique().tolist()
# destinations.sort()
# print()
# print('SURVEY TRIPS')
# print('-------------------------------------------------------------------------------------------')
# print(f'Survey trip origins: {origins}')
# print(f'Number of origins: {len(origins)}')
# print(f'Survey trip destinations: {destinations}')
# print(f'Number of destinations: {len(destinations)}')

# # Test matrices
# test_skim = omx.open_file("L:/UserData/David/ABM/data_base_case/skims_MRDH2_new.omx", 'r')
# matrix = test_skim.list_matrices().index('PRIVE_NIETGEDEELD_DIST')
# df = pd.DataFrame(np.array(test_skim['PRIVE_NIETGEDEELD_DIST']))
#
# print(df.iloc[0,0])
#
#
#
# print()
# print(f'SF TEST SKIMS')
# print('-------------------------------------------------------------------------------------------')
# print(f'Skims: {test_skim.list_matrices()}')
# print()
# print(f'Skims columns: {df.columns.values}')
# print()
# # df.index = np.arange(1, len(df) + 1)
# print(f'Skims indices: {df.index.values}')
# print()
# print(f'Number of origins and destinations: {len(df.columns.values) - 1}')
# print()
# print(f'Skim preview {test_skim.list_matrices()[matrix]}:')
# print(df)

tours = pd.read_csv("L:/UserData/David/ABM/data_base_case_sampled/survey_data/override_tours.csv")
trips = pd.read_csv("L:/UserData/David/ABM/data_base_case_sampled/survey_data/override_trips.csv")

# Some respondents make work or school trips to different locations, so they have more than one school or workplace loc,
# ActivitySim cannot handle this and they must be removed
double_destination = trips[['person_id', 'purpose', 'destination']]
double_destination = double_destination.loc[
    (double_destination.duplicated(subset=['person_id', 'purpose'], keep=False)) &
    (double_destination['purpose'].isin(['home']))]
double_destination.drop_duplicates(inplace=True)
opids_for_removal = double_destination.loc[
    (double_destination.duplicated(subset=['person_id', 'purpose'], keep=False)),
    'person_id'].unique().tolist()
print(opids_for_removal)

# school_tours = set(tours.loc[tours.tour_type == 'school', 'tour_id'].tolist())
# school_trips_tours = set(tours.loc[trips.purpose == 'school', 'tour_id'].to_list())
# weird = school_trips_tours - school_tours
# print(weird)
# print(len(weird))

# EOF
