# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_x', type=click.Path(exists=True))
@click.argument('input_target', type=click.Path(exists=True))

@click.argument('output_x_train', type=click.Path())
@click.argument('output_target_train', type=click.Path())

@click.argument('output_x_val', type=click.Path())
@click.argument('output_target_val', type=click.Path())

def main(input_x, input_target, output_x_train, output_target_train, output_x_val, output_target_val):
  
    logger = logging.getLogger(__name__)
    logger.info('making final data set from preprocessed data')

    data_features = pd.read_csv(input_x)
    data_target = pd.read_csv(input_target)

    X_train, X_val, y_train, y_val = train_test_split(
        data_features,  data_target, test_size=0.2, random_state=42)


    X_train.to_csv(output_x_train)
    y_train.to_csv(output_target_train)

    X_val.to_csv(output_x_val)
    y_val.to_csv(output_target_val)
    
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
