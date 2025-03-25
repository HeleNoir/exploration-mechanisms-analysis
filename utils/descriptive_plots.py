import os
from pathlib import Path

import pandas as pd

import utils


# Plot individual info (all data for one or more configs)
def plot_descriptive_individual_run(df: pd.DataFrame, value_columns: [str] = None, hue: str = None, style: str = None,
                                    dataset: str = None, algorithm: str = None, config_name: str = None, save_directory: str | Path = None):
    """
    Creates a .png with the plot for every specified columns over the iterations.
    If a dataframe with more than one row is provided, the plots contain the mean line and the standard deviation.
    If a dataframe with more than one row and a hue is provided, the plots contain individual lines depending on the hue.

    :param df: Dataframe with data to be plotted.
    :param value_columns: List of columns to be plotted, defaults to 'DistanceToOptimum', 'PopulationDistances_mean', 'RadiusDiversity', 'MeanStepSize', 'StepSizeVariance'.
    :param hue: Hue of lines (optional) if several rows are provided.
    :param style: Style of lines (optional) if several rows are provided.
    :param config_name: Name to add to the saved .png (optional); if not provided, will use the first config of the dataframe.
    """
    if value_columns is None:
        value_columns = ['DistanceToOptimum']
    output_dir = save_directory / f'plots/{dataset}/individual_lineplots/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    for col in value_columns:
        if style is None:
            df_full = df[list({'Evaluations', col, hue})]
        else:
            df_full = df[list({'Evaluations', col, hue, style})]
        df_full = df_full.explode(['Evaluations', col])
        df_full = df_full.dropna()
        utils.plot_descriptive_lineplots(df_full, 'Evaluations', col, hue, style,
                                         f'{output_dir}/{algorithm}_{config_name}')
        del df_full


# Plot summarised behaviour data in lineplots
# seaborn lineplot: summary per function
def plot_summarised_lineplots(df: pd.DataFrame, value_columns, hue, style: str = None, dataset: str = '', config_name: str = '', save_directory: str | Path = None):
    if value_columns is None:
        value_columns = ['DistanceToOptimum_mean']
    output_dir = save_directory / f'plots/{dataset}/summary_lineplots/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    for col in value_columns:
        if style is None:
            df_full = df[list({'Evaluations', col, hue})]
        else:
            df_full = df[list({'Evaluations', col, hue, style})]
        df_full = df_full.explode(['Evaluations', col])
        df_full = df_full.dropna()
        utils.plot_descriptive_lineplots(df_full, 'Evaluations', col, hue, style, f'{output_dir}/{config_name}')
        del df_full


def plot_diversity_summarised(df: pd.DataFrame, mean_columns, std_columns, dataset: str, config_name: str, save_directory: str | Path):
    output_dir = save_directory / f'plots/{dataset}/diversity/summary/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    for mean, std in zip(mean_columns, std_columns):
        df_full = df[['Evaluations', mean, std]]
        df_full = df_full.explode(['Evaluations', mean, std])
        utils.plot_descriptive_summarised_lineplots(df_full, 'Evaluations', mean, std, f'{output_dir}/{config_name}')
        del df_full


def plot_comparison(df: pd.DataFrame, algorithms, value_columns, hue, config_name: str = '', save_directory: str | Path = None):
    if value_columns is None:
        value_columns = ['DistanceToOptimum_mean', 'MinimumIndividualDistance_mean']
    df_full = df[list({'Evaluations', value_columns[0], value_columns[1], hue})]
    df_full = df_full.explode(['Evaluations', value_columns[0], value_columns[1]])
    utils.plot_twinaxes(df_full, algorithms, 'Evaluations', hue, f'{save_directory}/{config_name}')
    del df_full

