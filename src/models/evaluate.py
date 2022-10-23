import logging
import os
import json

import click
import pandas as pd
import pickle


from sklearn.metrics import mean_squared_error

@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path(exists=True))

def main(input_data_filepath, input_target_filepath, output_model_filepath):
  
    logger = logging.getLogger(__name__)
    logger.info('making validation metrics')

    
    val_data = pd.read_csv(input_data_filepath)
    val_target = pd.read_csv(input_target_filepath)



    trained_model = pickle.load(open(output_model_filepath, 'rb'))

    y_pred = trained_model.predict(val_data)

    metrics = {

        'RMSE': mean_squared_error(val_target['SalePrice'], y_pred)
       
    }

    metrics_path = os.path.join("reports", "figures", "metrics.json")

    with open(metrics_path, "w") as outfile:
        json.dump(metrics, outfile)

main()