import pandas as pd

def generate_features(data: pd.DataFrame) -> pd.DataFrame:

    data['MiscFeature'] = (data['MiscFeature'].notna()).astype(int)

    data['HasGarage'] = (data['GarageType'].notna()).astype(int)

    data['HasPool'] = (data['PoolArea'] > 0).astype(int)

    porch_cols = ['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']
    data['HasPorch'] = (data[porch_cols].sum(axis = 1) > 0).astype(int)

    data['HasFireplace'] = (data['Fireplaces'] > 0).astype(int)

    data['HasFence'] = (data['Fence'].notna() > 0).astype(int)

    data['HasVeneer'] = (data['MasVnrType'].notna()).astype(int)

    data['Has2ndFloor'] = (data['2ndFlrSF'] > 0).astype(int)

    data['HasBasement'] = (data['BsmtCond'].notna() > 0).astype(int)

    data['Remodel'] = (data['YearRemodAdd'] != data['YearBuilt']).astype(int)

    data['House_Age'] = data['YrSold'] - data['YearBuilt']

    sf_cols = ['TotalBsmtSF','1stFlrSF','2ndFlrSF']
    data['TotalSF'] = data[sf_cols].sum(axis = 1)

    data['TotalBath'] = data[['FullBath','BsmtFullBath']].sum(axis = 1) + 0.5 * data[['HalfBath','BsmtHalfBath']].sum(axis = 1)
    
    data['TotalPorch'] = data[porch_cols].sum(axis = 1)

    data["LivLotRatio"] = data["GrLivArea"] / data["LotArea"]

    data["Spaciousness"] = (data["1stFlrSF"] + data["2ndFlrSF"]) / data["TotRmsAbvGrd"]

    data['TotalLot'] = data['LotFrontage'] + data['LotArea']

    data['TotalBsmtFin'] = data['BsmtFinSF1'] + data['BsmtFinSF2']

    data["PCA_Feature1"] = data['GrLivArea'] + data['TotalBsmtSF']

    data["PCA_Feature2"] = data['YearRemodAdd'] * data['TotalBsmtSF']

    return data
