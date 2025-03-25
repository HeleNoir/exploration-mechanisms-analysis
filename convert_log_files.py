import os
from typing import Optional

import click
import pathlib

import pandas as pd

import utils


@click.command()
@click.option('-a', '--algorithm', type=click.STRING, default='PSO')
@click.option('-d', '--dimension', type=click.STRING, default='d10')
def main(algorithm: str, dimension: str) -> None:
    base_path = pathlib.Path(__file__).parent
    folder = rf'{algorithm}/{dimension}'
    log_directory = (pathlib.Path(base_path / '..' / 'exploration-mechanisms' / 'data' / folder)).resolve()
    save_directory = (pathlib.Path(base_path / 'data')).resolve()
    dataset_directory = (pathlib.Path(base_path / 'datasets'))
    pathlib.Path.mkdir(save_directory, parents=True, exist_ok=True)

    print(__file__)
    print(log_directory)
    print(save_directory)

    logs = utils.read_log_dir_with_joblib(log_directory)

    df = utils.dict_to_df(logs, algorithm)
    print(df.head())

    pathlib.Path.mkdir(save_directory / f'dataframes', parents=True, exist_ok=True)

    utils.add_dist_to_opt(df, dataset_directory)
    utils.add_final_distance(df)
    utils.add_final_aocc(df)

    df.to_feather(save_directory / f'dataframes/{algorithm}_{dimension}_full.feather')


if __name__ == '__main__':
    main()
