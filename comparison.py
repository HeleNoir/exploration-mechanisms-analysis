import pathlib

import click
from pathlib import Path
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import json

import utils


@click.command()
@click.option('-d', '--dimensions', type=click.STRING, default='d10')
def main(dimensions: str) -> None:
    base_path = Path(__file__).parent
    save_directory = (Path(base_path / 'data')).resolve()
    print(save_directory)
    experiment_directory = save_directory / f'comparison/{dimensions}/'
    pathlib.Path.mkdir(experiment_directory, parents=True, exist_ok=True)

    algorithms = ['PSO', 'SHADE', 'PSO_RR',  'PSO_GPGM', 'PSO_NPGM', 'PSO_PDM', 'PSO_SRM']
    process_dict = {}
    results_dict = {}
    functions_dict = {}
    for a in algorithms:
        df_process = pd.read_feather(
            save_directory / f'analysis/{a}_{dimensions}/dataframes/{a}_{dimensions}_functions_process_summaries.feather')
        process_dict[f'{a}_{dimensions}'] = df_process
        df_results = pd.read_feather(
            save_directory / f'analysis/{a}_{dimensions}/dataframes/{a}_{dimensions}_functions_summaries.feather')
        results_dict[f'{a}_{dimensions}'] = df_results

    process_df = pd.concat(process_dict)
    results_df = pd.concat(results_dict)

    functions = ['f001', 'f002', 'f003', 'f004', 'f005', 'f006', 'f007', 'f008', 'f009', 'f010', 'f011', 'f012', 'f013',
                 'f014', 'f015', 'f016', 'f017', 'f018', 'f019', 'f020', 'f021', 'f022', 'f023', 'f024']

    instances = ['i01', 'i02', 'i03', 'i04', 'i05']

    function_groups = [y for x, y in process_df.groupby('Function', observed=False)]
    for function_df in function_groups:
        f_name = function_df['Function'].iloc[0]
        utils.plot_comparison(function_df, ['PSO_RR', 'PSO_GPGM', 'PSO_NPGM', 'PSO_PDM', 'PSO_SRM'], ['DistanceToOptimum_mean', 'MinimumIndividualDistance_mean'], 'Algorithm',
                              f'{f_name}_comparison_PSO_RR', f'{experiment_directory}')
        utils.plot_comparison(function_df, ['SHADE', 'PSO_GPGM', 'PSO_NPGM', 'PSO_PDM', 'PSO_SRM'],
                              ['DistanceToOptimum_mean', 'MinimumIndividualDistance_mean'], 'Algorithm',
                              f'{f_name}_comparison_SHADE', f'{experiment_directory}')
        utils.plot_comparison(function_df, ['PSO', 'PSO_GPGM', 'PSO_NPGM', 'PSO_PDM', 'PSO_SRM'],
                              ['DistanceToOptimum_mean', 'MinimumIndividualDistance_mean'], 'Algorithm',
                              f'{f_name}_comparison_PSO_Variants', f'{experiment_directory}')
    del function_groups, process_df

    ### load all data to get
    for f in functions:
        for a in algorithms:
            df_function_results = pd.read_feather(
                save_directory / f'analysis/{a}_{dimensions}/dataframes/{a}_{dimensions}_{f}_instances_summaries.feather')
            functions_dict[f'{a}_{dimensions}_{f}'] = df_function_results
        function_results_df = pd.concat(functions_dict)

        ### per function statistical tests
        function_alg_data = []
        for a in algorithms:
            i_data = []
            for i in instances:
                i_value = function_results_df.loc[(function_results_df.Algorithm == a) & (function_results_df.Instance == i)]['FinalDistance_mean']
                i_data.append(i_value.iloc[0])
            function_alg_data.append(i_data)
        instance_data = np.array(function_alg_data)

        f_res = st.friedmanchisquare(*instance_data)
        print(f, f_res[1])
        # only perform posthoc test if Friedman test indicates significant differences
        if f_res[1] < 0.05:
            f_avg_ranks = function_results_df.groupby('Instance').FinalDistance_mean.rank(pct=True).groupby(function_results_df.Algorithm).mean()

            # Perform Nemenyi post hoc test (pairwise comparison p-values)
            f_p_values_matrix = sp.posthoc_nemenyi_friedman(results_df, melted=True, block_col='Instance',
                                                          block_id_col='Instance', group_col='Algorithm',
                                                          y_col='FinalDistance_mean')

            results_json = {'Friedman statistic': f_res[0],
                            'Friedman pvalue': f_res[1],
                            'Average ranks': f_avg_ranks.to_json(),
                            'Nemenyi Friedman': f_p_values_matrix.to_json()}
            with open(experiment_directory / f"comparison_{dimensions}_{f}.json", "w") as outfile:
                json.dump(results_json, outfile)

            # Plot Critical Difference Diagram
            plt.figure(figsize=(8, 2), dpi=200)
            sp.critical_difference_diagram(
                f_avg_ranks,  # Average ranks of algorithms
                f_p_values_matrix  # Use the p-values from the Nemenyi test
            )
            plt.savefig(f'{experiment_directory}/crd_{dimensions}_{f}', bbox_inches='tight', pad_inches=0)
            plt.close()


    # Perform Friedman test
    alg_data = []
    for a in algorithms:
        f_data = []
        for f in functions:
            f_value = results_df.loc[(results_df.Algorithm == a) & (results_df.Function == f)]['FinalDistance_mean']
            f_data.append(f_value.iloc[0])
        alg_data.append(f_data)
    data = np.array(alg_data)

    res = st.friedmanchisquare(*data)

    # Compute average ranks of each algorithm
    avg_ranks = results_df.groupby('Function').FinalDistance_mean.rank(pct=True).groupby(results_df.Algorithm).mean()

    # Perform Nemenyi post hoc test (pairwise comparison p-values)
    print(results_df.head())
    p_values_matrix = sp.posthoc_nemenyi_friedman(results_df, melted=True, block_col='Function', block_id_col='Function', group_col='Algorithm', y_col='FinalDistance_mean')
    print(p_values_matrix)

    results_json = {'Friedman statistic': res[0],
                    'Friedman pvalue': res[1],
                    'Average ranks': avg_ranks.to_json(),
                    'Nemenyi Friedman': p_values_matrix.to_json()}
    with open(experiment_directory / f"comparison_{dimensions}.json", "w") as outfile:
        json.dump(results_json, outfile)

    # Plot Critical Difference Diagram
    plt.figure(figsize=(8, 2), dpi=200)
    sp.critical_difference_diagram(
        avg_ranks,  # Average ranks of algorithms
        p_values_matrix  # Use the p-values from the Nemenyi test
    )
    plt.savefig(f'{experiment_directory}/crd_{dimensions}', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    main()
