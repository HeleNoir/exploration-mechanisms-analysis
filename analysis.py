import pathlib

import click
from pathlib import Path
import pandas as pd
import os

import utils


@click.command()
@click.option('-a', '--algorithm', type=click.STRING, default='PSO')
@click.option('-d', '--dimension', type=click.STRING, default='d10')
def main(algorithm: str, dimension: str) -> None:
    dataset = f'{algorithm}_{dimension}'
    base_path = Path(__file__).parent
    save_directory = (Path(base_path / 'data')).resolve()
    print(save_directory)
    df = pd.read_feather(save_directory / f'dataframes/{dataset}_full.feather')
    print(df.head())
    experiment_directory = save_directory / f'analysis/{dataset}/'
    pathlib.Path.mkdir(experiment_directory, parents=True, exist_ok=True)

    value_columns = ['DistanceToOptimum',
                     'MinimumIndividualDistance']
    mean_columns = ['DistanceToOptimum_mean',
                    'MinimumIndividualDistance_mean']
    div_std_columns = ['DistanceToOptimum_std',
                       'MinimumIndividualDistance_std']

    # Summary statistics of process data, summarised over all
    df_process_stats_all = utils.summarise_process_stats(df,
                                                         ['Algorithm'],
                                                         value_columns,
                                                         'Evaluations',
                                                         None,
                                                         dataset,
                                                         'all',
                                                         experiment_directory)

    # Summary statistics of final values, summarised over all
    df_final_stats_all = utils.summarise_final_stats(df,
                                                     ['Algorithm'],
                                                     ['FinalDistance', 'AOCC'],
                                                     None,
                                                     dataset,
                                                     'all',
                                                     experiment_directory)

    utils.plot_summarised_lineplots(df_process_stats_all,
                                    value_columns=mean_columns,
                                    hue='Algorithm',
                                    dataset=dataset,
                                    config_name='all_process', save_directory=experiment_directory)

    utils.plot_diversity_summarised(df_process_stats_all, mean_columns, div_std_columns, dataset,
                                    f'{dataset}_all_diversity',
                                    experiment_directory)

    del df_process_stats_all, df_final_stats_all

    # Problem-dependent analysis

    # Summary statistics of process data, summarised by function
    df_function_process_stats = utils.summarise_process_stats(df,
                                                              ['Function'],
                                                              value_columns,
                                                              'Evaluations',
                                                              ['Algorithm'],
                                                              dataset,
                                                              f'functions',
                                                              experiment_directory)

    # Summary statistics of final values, summarised by function
    df_function_stats = utils.summarise_final_stats(df,
                                                    ['Function'],
                                                    ['FinalDistance', 'AOCC'],
                                                    ['Algorithm'],
                                                    dataset,
                                                    f'functions',
                                                    experiment_directory)

    utils.plot_summarised_lineplots(df_function_process_stats,
                                    value_columns=mean_columns,
                                    hue='Function',
                                    dataset=dataset,
                                    config_name='function_process', save_directory=experiment_directory)

    group_param_dfs = [y for x, y in df_function_process_stats.groupby('Function', observed=False)]
    for group_param_df in group_param_dfs:
        value = group_param_df['Function'].iloc[0]
        utils.plot_diversity_summarised(group_param_df, mean_columns, div_std_columns, dataset,
                                        f'{dataset}_{value}', experiment_directory)

    function_dfs = [y for x, y in df.groupby('Function', observed=False)]
    for function_df in function_dfs:
        f = function_df['Function'].iloc[0]
        # Summary statistics of process data, summarised by instance
        df_instance_process_stats = utils.summarise_process_stats(function_df,
                                                                  ['Instance'],
                                                                  value_columns,
                                                                  'Evaluations',
                                                                  ['Algorithm'],
                                                                  dataset,
                                                                  f'{f}_instances',
                                                                  experiment_directory)

        utils.plot_summarised_lineplots(df_instance_process_stats,
                                        value_columns=mean_columns,
                                        hue='Instance',
                                        dataset=dataset,
                                        config_name=f'{f}_instance_process', save_directory=experiment_directory)

        instance_dfs = [y for x, y in function_df.groupby('Instance', observed=False)]
        for instance_df in instance_dfs:
            i = instance_df['Instance'].iloc[0]
            utils.plot_descriptive_individual_run(instance_df, value_columns=value_columns, hue='Run', style=None,
                                                  algorithm=algorithm, dataset=dataset,
                                                  config_name=f'{dataset}_{f}_{i}', save_directory=experiment_directory)

    del function_dfs, instance_dfs, df_function_process_stats, df_function_stats, df_instance_process_stats


if __name__ == '__main__':
    main()
