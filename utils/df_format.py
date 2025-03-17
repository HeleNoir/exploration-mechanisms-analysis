"""
Specific functions to convert results from exploration mechanism experiments to dataframe with all informaiton.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import utils

# Labels for optimisation problem information
config_labels = [
    'Run',
    'Function',
    'Instance',
    'Dimension',
    'Group'
]

config_dict = {
    'Run': 'int32',
    'Function': 'category',
    'Instance': 'category',
    'Dimension': 'category',
    'Group': 'category',
}

# Labels for logged data
common_labels = [
    'Iterations',
    'Evaluations',
    'BestObjectiveValue',
    'MinimumIndividualDistance',
]

# BBOB function groups
group1 = ['f001', 'f002', 'f003', 'f004', 'f005']
group2 = ['f006', 'f007', 'f008', 'f009']
group3 = ['f010', 'f011', 'f012', 'f013', 'f014']
group4 = ['f015', 'f016', 'f017', 'f018', 'f019']
group5 = ['f020', 'f021', 'f022', 'f023', 'f024']


def dict_to_df(data_dict, algorithm) -> pd.DataFrame:
    """
    Turn the dictionary of dataframes from experiments into one single dataframe, splitting the config
    information into individual columns.
    Labels are specified for exploration-mechanisms
    """
    labels = ['Config'] + ['Algorithm'] + config_labels + common_labels

    convert_dict = {
        'Config': 'string',
        'Algorithm': 'string',
    }
    convert_dict.update(config_dict)

    df = pd.DataFrame(columns=labels)

    for key, value in data_dict.items():
        config = key.split('_')
        entry = [key, algorithm, config[0], config[2], config[3], config[4]]
        if config[2] in group1:
            entry.append('Group1')
        elif config[2] in group2:
            entry.append('Group2')
        elif config[2] in group3:
            entry.append('Group3')
        elif config[2] in group4:
            entry.append('Group4')
        elif config[2] in group5:
            entry.append('Group5')
        iterations = value['mahf::state::common::Iterations'].tolist()
        corrected_iterations = [iterations[0]] + [i + 1 for i in iterations[1:]]
        entry.append(corrected_iterations)
        entry.append(value['mahf::state::common::Evaluations'].tolist())
        entry.append(value['BestObjectiveValue'].tolist())
        entry.append(value['mahf::components::measures::diversity::MinimumIndividualDistance'].tolist())

        df.loc[len(df)] = entry

    df = df.astype(convert_dict)

    return df


def add_dist_to_opt(df: pd.DataFrame, dataset_directory, functions=None):
    """
    Add a column for the distance to the actual optimum for the BBOB function-instance combinations included in a list.
    If no list is provided, all function-instance combinations are used.
    Requires a csv file with bbob function name and optimum values.
    """
    optima_path = dataset_directory / 'bbob_optima.csv'
    optima = pd.read_csv(optima_path, header=None, names=['Function', 'Optimum'])
    if functions is None:
        all_functions = optima['Function'].str.split('_')
        function_list = ['_'.join([all_functions[x][1]] + [all_functions[x][2]]) for x in range(0, len(all_functions))]
        function_list = list(set(function_list))
        df['DistanceToOptimum'] = df.apply(calculate_difference, axis=1, df2=optima, function_list=function_list)
        df['DistanceToOptimum'] = df['DistanceToOptimum'].apply(lambda arr: np.where(arr <= 0.0, sys.float_info.epsilon, arr))
    else:
        df['DistanceToOptimum'] = df.apply(calculate_difference, axis=1, df2=optima, function_list=functions)
        df['DistanceToOptimum'] = df['DistanceToOptimum'].apply(lambda arr: np.where(arr <= 0.0, sys.float_info.epsilon, arr))

    # Drop column after calculating DistanceToOptimum as we (hopefully) don't need it anymore
    df.drop(columns=['BestObjectiveValue'])


def calculate_difference(df1, df2, function_list):
    """ (Note: This function was created with the help of ChatGPT-4o.) """
    for name in function_list:
        if name in df1['Config']:
            matching_rows = df2[df2['Function'].str.contains(name, na=False)]

            if not matching_rows.empty:
                return df1['BestObjectiveValue'] - matching_rows['Optimum'].values[0]
    return None


def add_final_distance(df: pd.DataFrame):
    """
    Add a column for the final distance to the optimum (requires column 'DistanceToOptimum').
    """
    assert 'DistanceToOptimum' in df, 'Cannot add final distance: Column DistanceToOptimum is missing!'
    df['FinalDistance'] = df['DistanceToOptimum'].str[-1]


def add_final_aocc(df: pd.DataFrame):
    """
    Add a column for the overall AOCC value of the run.
    """
    assert 'DistanceToOptimum' in df, 'Cannot add AOCC: Column DistanceToOptimum is missing!'
    aocc_data = df[['DistanceToOptimum']].apply(lambda x: x['DistanceToOptimum'], axis=1, result_type='expand')
    transposed = aocc_data.T
    aocc_values = lambda x: utils.aocc(x)
    transposed = transposed.apply(aocc_values)
    aocc_data = transposed.T
    df['AOCC'] = aocc_data

