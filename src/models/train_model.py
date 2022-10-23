# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

from catboost import CatBoostRegressor


@click.command()
@click.argument('x_train_path', type=click.Path(exists=True))
@click.argument('target_train_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path())

def main(x_train_path, target_train_filepath, output_model_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    X = pd.read_csv(x_train_path)
    y = pd.read_csv(target_train_filepath)

    catboost = CatBoostRegressor()

    catboost.fit(X, y['SalePrice'])

    catboost.save_model(output_model_filepath)

    # fit, save model or hyperparameters tuning using somethink like RandomizedSearchCV

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
