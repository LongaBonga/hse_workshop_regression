import pandas as pd
import numpy as np
from typing import Tuple

from category_encoders import OneHotEncoder

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:

    data.drop(columns = ['PoolQC','MoSold','YearBuilt','YrSold'], inplace = True)

    category_cols = [x for x in data.columns if data[x].dtype == 'object']
    binary_cols = [x for x in data.columns if len(data[x].unique()) == 2]
    numerical_cols = [x for x in data.columns if (x not in category_cols + binary_cols)]


    # One Hot Encoding (+ drop target)
    encoder = OneHotEncoder(cols = category_cols, use_cat_names = True)
    target = data['SalePrice'].apply(np.log1p)
    data = encoder.fit_transform(data[data.columns])

    return data, target, encoder



