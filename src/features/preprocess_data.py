# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from preprocess import preprocess_data
import pickle


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_x_data_filepath', type=click.Path())
@click.argument('output_target_filepath', type=click.Path())
@click.argument('output_ecnoder_filepath', type=click.Path())
def main(input_filepath, output_x_data_filepath, output_target_filepath, output_ecnoder_filepath):
    """ Runs data processing scripts to turn raw data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    dataset = pd.read_csv(input_filepath)

    X_data, target, encoder = preprocess_data(dataset)

    X_data.to_csv(output_x_data_filepath)
    target.to_csv(output_target_filepath)

    pickle.dump(encoder, open(output_ecnoder_filepath, 'wb'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
