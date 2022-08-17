"""
File to hold functions to easily reestimate models and set up the files for simulation

Created on Mon Jun 27 13:00:54 2022

@author: David Matheus
"""

import pandas as pd
import time
# import numpy as np
# import yaml
# import larch.util.excel
import os
import datetime
import shutil
import logging
from pathlib import Path

from activitysim.estimation.larch import component_model, update_coefficients, update_size_spec
# from activitysim.estimation.larch.location_choice import school_location_model

# Important: run from parent folder from output folders

# Set up logger
logger = logging.getLogger()
consoleHandler = logging.StreamHandler()
loglevel = logging.DEBUG
consoleHandler.setLevel(loglevel)
logger.setLevel(loglevel)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
consoleHandler.setFormatter(formatter)
if len(logger.handlers) == 0:
    logger.debug("Adding new log handler, none exists yet.")
    logger.addHandler(consoleHandler)
filename = 'estimation.log'
file_handler = logging.FileHandler(filename)
file_handler.setLevel(loglevel)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def rename_to_output(case):
    """
        There is little flexibility in the directory name of the outputs, hence we rename temporarily

        """
    if case == '':
        return
    current_directory = os.getcwd()
    source_directory = Path(current_directory, f'output_{case}')
    target_directory = Path(current_directory, 'output')
    os.rename(source_directory, target_directory)


def rename_to_output_case(case):
    """
        Reverts rename_to_output function before final output
        """
    if case == '':
        return
    current_directory = os.getcwd()
    source_directory = Path(current_directory, 'output')
    target_directory = Path(current_directory, f'output_{case}')
    os.rename(source_directory, target_directory)


def correct_output(modelname):
    """
        Brute force method to correct outputs that contain NaN values in override column

        Parameters
        ----------
        modelname : string
            Name of the model whose outputs need correction

        Output
        -------
        corrected output : csv files (tables)
            Updated output files in Estimated Data Bundles (EDB)
        """

    alternatives_file = f"L:/UserData/David/ABM/output/estimation_data_bundle/{modelname}/" \
                        f"{modelname}_alternatives_combined.csv"
    choosers_file = f"L:/UserData/David/ABM/output/estimation_data_bundle/{modelname}/{modelname}_choosers_combined.csv"

    alternatives = pd.read_csv(alternatives_file)
    choosers = pd.read_csv(choosers_file)

    bad_person_id = choosers.loc[choosers.override_choice < 0, 'person_id'].unique().tolist()

    if bad_person_id:
        logger.warning(f'Missing survey data for {modelname}, correcting entries: {bad_person_id}')

    choosers = choosers.loc[~choosers.override_choice < 0]
    alternatives = alternatives.loc[~alternatives.person_id.isin(bad_person_id)]

    choosers.to_csv(choosers_file, index=False)
    alternatives.to_csv(alternatives_file, index=False)


def estimate_model(modelname, case=''):
    """
        Reestimate model after running ActivitySim in estimation mode and obtaining Estimation Data Bundles (EDBs).

        Parameters
        ----------
        modelname : string
            Name of the model to estimate
        case : string
            Name of the case (suffix of output folder)

        Output
        -------
        revised coefficients : csv file (table)
            Updated coefficients for utility functions in ActivitySim, must be swapped in configs directory

        size terms : csv file (table)
            Updated size terms for ActivitySim, must be swapped in configs directory. Note that all the location and
            destination choice models share the same destination_choice_size_terms.csv input file, so if you are
            updating all these models, you'll need to ensure that updated sections of this file for each model are
            joined together correctly.

        model estimation report : xlsx file (table)
            Estimation report, including coefficient t-statistic and log likelihood
        """

    rename_to_output(case)

    if modelname in ['school_location', 'workplace_location']:
        correct_output(modelname)

    model, data = component_model(modelname, return_data=True)

    # FIXME use edb_directory="output/estimation_data_bundle/{name}/"
    # rename_to_output_case(case)  # Prevents the directory from remaining renamed if error is raised

    model.estimate(method='BHHH', options={'maxiter': 1000})

    # print(model.parameter_summary())

    result_dir = data.edb_directory / "estimated"

    # print(result_dir)

    # rename_to_output(case)

    update_coefficients(model, data, result_dir, output_file=f"{modelname}_coefficients.csv",)

    update_size_spec(model, data, result_dir, output_file=f"{modelname}_size_terms.csv")

    model.to_xlsx(result_dir / f"{modelname}_model_estimation.xlsx", data_statistics=False,)

    rename_to_output_case(case)


def now():
    current_datetime = datetime.datetime.now()
    return current_datetime.strftime("%d/%m/%Y %H:%M:%S")


def copy_files(models, cases=''):
    """
                Convenience function to put estimated files on a single directory per case, and copies them
                automatically to the appropriate configs directory (ensuring there is a copy of the original config
                files) to easily set up to run model on simulation mode, otherwise files are scattered.

                Parameters
            ----------
            models : string or list
                Names of the models to estimate for each case
            case : string or list
                Name of the case (suffix of output folder)

                """
    if isinstance(models, str):
        models = [models]

    if isinstance(cases, str):
        cases = [cases]

    current_directory = os.getcwd()
    configs_directory = Path(current_directory, 'configs')

    for case in cases:
        if case != '':
            case = '_' + case

        # Use alternative configs in case they are specified per case
        specific_configs_directory = Path(current_directory, f'configs_{case}')
        if os.path.exists(specific_configs_directory):
            configs_directory = specific_configs_directory

        asim_output_directory = Path(current_directory, f'output{case}')
        target_directory = Path(asim_output_directory, 'consolidated_estimation_files')

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        for model in models:
            estimation_directory = Path(asim_output_directory, f'estimation_data_bundle/{model}/estimated')

            files = os.listdir(estimation_directory)
            # Size terms need to be dealt with in a different way, as the files need to be unified
            files = [file for file in files if file not in
                     [f'{model}_size_terms.csv', f'{model}_model_estimation.xlsx']]
            for file in files:
                source_file_dir = Path(estimation_directory, file)
                target_file_dir = Path(target_directory, file)
                # Check if the file already exists in target directory, and delete it if does
                if os.path.exists(target_file_dir):
                    os.remove(target_file_dir)
                shutil.move(source_file_dir, target_file_dir)

        # Dealing with size terms, that only appear on destination choice
        configs_file_dir = Path(configs_directory, 'destination_choice_size_terms.csv')
        target_file_dir = Path(target_directory, 'destination_choice_size_terms.csv')

        if os.path.exists(target_file_dir):
            os.remove(target_file_dir)
        shutil.copy2(configs_file_dir, target_file_dir)

        destination_models = ['school_location', 'workplace_location', 'non_mandatory_tour_destination',
                              'atwork_subtour_destination', 'trip_destination']
        models_subset = [model for model in models if model in destination_models]

        model_mapping = {
            'school_location': 'school',
            'workplace_location': 'workplace',
            'non_mandatory_tour_destination': 'non_mandatory',
            'atwork_subtour_destination': 'atwork',
            'trip_destination': 'trip',
        }

        for model in models_subset:
            estimation_directory = Path(asim_output_directory, f'estimation_data_bundle/{model}/estimated')
            source_file_dir = Path(estimation_directory, f'{model}_size_terms.csv')
            model_size_terms = pd.read_csv(source_file_dir)
            global_size_terms = pd.read_csv(target_file_dir)

            for column in global_size_terms.columns.values.tolist():
                global_size_terms.loc[global_size_terms.model_selector == model_mapping[model], column] = \
                    model_size_terms.loc[model_size_terms.model_selector == model_mapping[model], column]

            global_size_terms.to_csv(target_file_dir, index=False)
            os.remove(source_file_dir)

        # for model in models:
        #     estimation_directory = Path(asim_output_directory, f'estimation_data_bundle/{model}/estimated')
        #     os.rmdir(estimation_directory)

        backup_directory = Path(configs_directory, 'backup_configs')
        if not os.path.exists(backup_directory):
            os.makedirs(backup_directory)

        new_config_files = os.listdir(target_directory)

        for file in new_config_files:
            configs_file_dir = Path(configs_directory, file)
            backup_file_dir = Path(backup_directory, file)
            estimated_file_dir = Path(target_directory, file)
            if not os.path.exists(backup_file_dir): # Only backs up the original, do not overwrite backup
                shutil.move(configs_file_dir, backup_file_dir)
            else:
                os.remove(configs_file_dir)
            shutil.copy2(estimated_file_dir, configs_file_dir)


def batch_estimate_models(models, cases=''):
    """
            Handy function to estimate several models and cases.

            Parameters
            ----------
            models : string or list
                Names of the models to estimate for each case
            case : string or list
                Name of the case (suffix of output folder)

            Output
            -------
            revised coefficients : csv file (table)
                Updated coefficients for utility functions in ActivitySim per model and case, must be swapped in
                configs directory

            size terms : csv file (table)
                Updated size terms for ActivitySim per model per case, must be swapped in configs directory. Note that
                all the location and destination choice models share the same destination_choice_size_terms.csv input
                file, so if you are updating all these models, you'll need to ensure that updated sections of this file
                for each model are joined together correctly.

            model estimation report : xlsx file (table)
                Estimation report per model per case, including coefficient t-statistic and log likelihood
            """

    if isinstance(models, str):
        models = [models]

    if isinstance(cases, str):
        cases = [cases]

    # Start main timer
    main_tic = time.time()
    logger.info(f'Starting estimation procedure')
    for case in cases:

        # Start estimation
        logger.info(f'Starting estimation for {case} case')
        # Start case timer
        case_tic = time.time()
        for model in models:
            logger.info(f'Starting {model} estimation for {case} case')
            # Start model timer
            model_tic = time.time()

            # TODO check if model output needs brute force correction and give a warning if it does

            # Estimate
            estimate_model(model, case)

            # Stop model timer
            model_toc = time.time()
            logger.info(f'Estimation finished for {model} model for case {case}, runtime: '
                        f'{round(model_toc - model_tic, 2)} seconds '
                        f'({round((model_toc - model_tic) / 60, 2)} minutes)')

        # Stop case timer
        case_toc = time.time()
        logger.info(f'Estimation finished for {case} case, runtime: {round(case_toc - case_tic, 2)} seconds'
                    f' ({round((case_toc - case_tic) / 60, 2)} minutes)')

        # Copy files to right location
        for model in models:
            logger.info(f'Consolidating files for {model} {case} and copying to configs folder')
            copy_files(model, case)

    # Stop main timer
    main_toc = time.time()
    print(f'{now()} - INFO - All estimations finished, total runtime: {round(main_toc - main_tic, 2)} seconds'
          f' ({round((main_toc - main_tic) / 60, 2)} minutes)')


models = ['school_location', 'workplace_location']
# models = [
#     'school_location', 'workplace_location', # 'cdap',
#     'mandatory_tour_frequency', 'mandatory_tour_scheduling_school',
#     'mandatory_tour_scheduling_work', 'nonmand_tour_freq', 'non_mandatory_tour_destination',
#     'non_mandatory_tour_scheduling', 'tour_mode_choice', 'atwork_subtour_frequency', 'atwork_subtour_destination',
#     'atwork_subtour_scheduling', 'atwork_subtour_mode_choice'
# ]
cases = ['base_case_mrdh']
# cases = ['spatial_presampling']
# cases = ['srs_presampling']

batch_estimate_models(models, cases)
# copy_files(models, cases)


# EOF

