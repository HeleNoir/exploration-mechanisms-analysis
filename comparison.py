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
    for a in algorithms:
        df_process = pd.read_feather(
            save_directory / f'analysis/{a}_{dimensions}/dataframes/{a}_{dimensions}_functions_process_summaries.feather')
        process_dict[f'{a}_{dimensions}'] = df_process
        df_results = pd.read_feather(
            save_directory / f'analysis/{a}_{dimensions}/dataframes/{a}_{dimensions}_functions_summaries.feather')
        results_dict[f'{a}_{dimensions}'] = df_results

    process_df = pd.concat(process_dict)
    results_df = pd.concat(results_dict)

    function_groups = [y for x, y in process_df.groupby('Function', observed=False)]
    for function_df in function_groups:
        f = function_df['Function'].iloc[0]
        utils.plot_comparison(function_df, ['PSO_RR', 'PSO_GPGM', 'PSO_NPGM', 'PSO_PDM', 'PSO_SRM'], ['DistanceToOptimum_mean', 'MinimumIndividualDistance_mean'], 'Algorithm',
                              f'{f}_comparison_PSO_RR', f'{experiment_directory}')
        utils.plot_comparison(function_df, ['SHADE', 'PSO_GPGM', 'PSO_NPGM', 'PSO_PDM', 'PSO_SRM'],
                              ['DistanceToOptimum_mean', 'MinimumIndividualDistance_mean'], 'Algorithm',
                              f'{f}_comparison_SHADE', f'{experiment_directory}')
        utils.plot_comparison(function_df, ['PSO', 'PSO_GPGM', 'PSO_NPGM', 'PSO_PDM', 'PSO_SRM'],
                              ['DistanceToOptimum_mean', 'MinimumIndividualDistance_mean'], 'Algorithm',
                              f'{f}_comparison_PSO_Variants', f'{experiment_directory}')

    # Perform Friedman test
    functions = ['f001', 'f002', 'f003', 'f004', 'f005', 'f006', 'f007', 'f008', 'f009', 'f010', 'f011', 'f012', 'f013',
                 'f014', 'f015', 'f016', 'f017', 'f018', 'f019', 'f020', 'f021', 'f022', 'f023', 'f024']

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
    sp.critical_difference_diagram(
        avg_ranks,  # Average ranks of algorithms
        p_values_matrix  # Use the p-values from the Nemenyi test
    )
    plt.savefig(f'{experiment_directory}/crd_{dimensions}', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    main()
