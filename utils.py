import numpy as np
from itertools import groupby
from sklearn.linear_model import LinearRegression
from operator import itemgetter
import pandas as pd
import pandas_ta as ta


def impute_missing_dates(df, new_row_indices):

    
    # Define the window size for linear regression
    window_size = 5

    # Group the consecutive missing indices into windows
    windows = []
    for k, g in groupby(enumerate(new_row_indices), lambda ix: ix[0] - ix[1]):
        windows.append(list(map(itemgetter(1), g)))

    # Loop over the columns
    for col in df.columns:
        # Loop over the windows
        for window in windows:
            # Get the indices of non-missing values for this column
            start = max(window[0] - window_size, 0)
            end = min(window[-1] + window_size + 1, len(df))
            non_missing = df.loc[start:end, col].dropna()
            
            # Fit a linear regression using the previous 5 and next 5 non-missing values
            if len(non_missing) >= window_size * 2:
                X = np.array(non_missing.index).reshape(-1, 1)
                y = non_missing.values.reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict the missing values
                for i in window:
                    if pd.isna(df.loc[i, col]):
                        y_pred = model.predict([[i]])[0][0]
                        # Fill the missing value with the predicted value
                        df.loc[i, col] = y_pred
    return df


'''
Feature engineering
'''

def get_pressure_diff_depth(df):

    # Pressure depth differences
    df['Pressure diff 7061-6982 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z02D6982Ps_psi']
    df['Pressure diff 7061-6837 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z04D6837Ps_psi']
    df['Pressure diff 7061-6720 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z05D6720Ps_psi']
    df['Pressure diff 7061-6632 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z06D6632Ps_psi']
    df['Pressure diff 7061-6416 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z07D6416Ps_psi']
    df['Pressure diff 7061-5840 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z08D5840Ps_psi']
    df['Pressure diff 7061-5482 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z0910D5482Ps_psi']
    df['Pressure diff 7061-5653 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z09D5653Ps_psi']
    df['Pressure diff 7061-5001 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 7061-4917 ft'] = df['Avg_VW1_Z01D7061Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 6982-6837 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z04D6837Ps_psi']
    df['Pressure diff 6982-6720 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z05D6720Ps_psi']
    df['Pressure diff 6982-6632 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z06D6632Ps_psi']
    df['Pressure diff 6982-6416 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z07D6416Ps_psi']
    df['Pressure diff 6982-5840 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z08D5840Ps_psi']
    df['Pressure diff 6982-5482 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z0910D5482Ps_psi']
    df['Pressure diff 6982-5653 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z09D5653Ps_psi']
    df['Pressure diff 6982-5001 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 6982-4917 ft'] = df['Avg_VW1_Z02D6982Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 6837-6720 ft'] = df['Avg_VW1_Z04D6837Ps_psi'] - df['Avg_VW1_Z05D6720Ps_psi']
    df['Pressure diff 6837-6632 ft'] = df['Avg_VW1_Z04D6837Ps_psi'] - df['Avg_VW1_Z06D6632Ps_psi']
    df['Pressure diff 6837-6416 ft'] = df['Avg_VW1_Z04D6837Ps_psi'] - df['Avg_VW1_Z07D6416Ps_psi']
    df['Pressure diff 6837-5840 ft'] = df['Avg_VW1_Z04D6837Ps_psi'] - df['Avg_VW1_Z08D5840Ps_psi']
    df['Pressure diff 6837-5482 ft'] = df['Avg_VW1_Z04D6837Ps_psi'] - df['Avg_VW1_Z0910D5482Ps_psi']
    df['Pressure diff 6837-5653 ft'] = df['Avg_VW1_Z04D6837Ps_psi'] - df['Avg_VW1_Z09D5653Ps_psi']
    df['Pressure diff 6837-5001 ft'] = df['Avg_VW1_Z04D6837Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 6837-4917 ft'] = df['Avg_VW1_Z04D6837Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 6720-6632 ft'] = df['Avg_VW1_Z05D6720Ps_psi'] - df['Avg_VW1_Z06D6632Ps_psi']
    df['Pressure diff 6720-6416 ft'] = df['Avg_VW1_Z05D6720Ps_psi'] - df['Avg_VW1_Z07D6416Ps_psi']
    df['Pressure diff 6720-5840 ft'] = df['Avg_VW1_Z05D6720Ps_psi'] - df['Avg_VW1_Z08D5840Ps_psi']
    df['Pressure diff 6720-5482 ft'] = df['Avg_VW1_Z05D6720Ps_psi'] - df['Avg_VW1_Z0910D5482Ps_psi']
    df['Pressure diff 6720-5653 ft'] = df['Avg_VW1_Z05D6720Ps_psi'] - df['Avg_VW1_Z09D5653Ps_psi']
    df['Pressure diff 6720-5001 ft'] = df['Avg_VW1_Z05D6720Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 6720-4917 ft'] = df['Avg_VW1_Z05D6720Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 6632-6416 ft'] = df['Avg_VW1_Z06D6632Ps_psi'] - df['Avg_VW1_Z07D6416Ps_psi']
    df['Pressure diff 6632-5840 ft'] = df['Avg_VW1_Z06D6632Ps_psi'] - df['Avg_VW1_Z08D5840Ps_psi']
    df['Pressure diff 6632-5482 ft'] = df['Avg_VW1_Z06D6632Ps_psi'] - df['Avg_VW1_Z0910D5482Ps_psi']
    df['Pressure diff 6632-5653 ft'] = df['Avg_VW1_Z06D6632Ps_psi'] - df['Avg_VW1_Z09D5653Ps_psi']
    df['Pressure diff 6632-5001 ft'] = df['Avg_VW1_Z06D6632Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 6632-4917 ft'] = df['Avg_VW1_Z06D6632Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 6416-5840 ft'] = df['Avg_VW1_Z07D6416Ps_psi'] - df['Avg_VW1_Z08D5840Ps_psi']
    df['Pressure diff 6416-5482 ft'] = df['Avg_VW1_Z07D6416Ps_psi'] - df['Avg_VW1_Z0910D5482Ps_psi']
    df['Pressure diff 6416-5653 ft'] = df['Avg_VW1_Z07D6416Ps_psi'] - df['Avg_VW1_Z09D5653Ps_psi']
    df['Pressure diff 6416-5001 ft'] = df['Avg_VW1_Z07D6416Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 6416-4917 ft'] = df['Avg_VW1_Z07D6416Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 5840-5653 ft'] = df['Avg_VW1_Z08D5840Ps_psi'] - df['Avg_VW1_Z09D5653Ps_psi']
    df['Pressure diff 5840-5482 ft'] = df['Avg_VW1_Z08D5840Ps_psi'] - df['Avg_VW1_Z0910D5482Ps_psi']
    df['Pressure diff 5840-5001 ft'] = df['Avg_VW1_Z08D5840Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 5840-4917 ft'] = df['Avg_VW1_Z08D5840Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 5653-5482 ft'] = df['Avg_VW1_Z09D5653Ps_psi'] - df['Avg_VW1_Z0910D5482Ps_psi']
    df['Pressure diff 5653-5001 ft'] = df['Avg_VW1_Z09D5653Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 5653-4917 ft'] = df['Avg_VW1_Z09D5653Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 5482-5001 ft'] = df['Avg_VW1_Z0910D5482Ps_psi'] - df['Avg_VW1_Z10D5001Ps_psi']
    df['Pressure diff 5482-4917 ft'] = df['Avg_VW1_Z0910D5482Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    df['Pressure diff 5001-4917 ft'] = df['Avg_VW1_Z10D5001Ps_psi'] - df['Avg_VW1_Z11D4917Ps_psi']

    return df

def get_temperature_diff_depth(df):

    # Temperature depth differences
    df['Temperature diff 7061-6982 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z02D6982Tp_F']
    df['Temperature diff 7061-6837 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z04D6837Tp_F']
    df['Temperature diff 7061-6720 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z05D6720Tp_F']
    df['Temperature diff 7061-6632 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z06D6632Tp_F']
    df['Temperature diff 7061-6416 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z07D6416Tp_F']
    df['Temperature diff 7061-5840 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z08D5840Tp_F']
    df['Temperature diff 7061-5482 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z0910D5482Tp_F']
    df['Temperature diff 7061-5653 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z09D5653Tp_F']
    df['Temperature diff 7061-5001 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 7061-4917 ft'] = df['Avg_VW1_Z01D7061Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 6982-6837 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z04D6837Tp_F']
    df['Temperature diff 6982-6720 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z05D6720Tp_F']
    df['Temperature diff 6982-6632 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z06D6632Tp_F']
    df['Temperature diff 6982-6416 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z07D6416Tp_F']
    df['Temperature diff 6982-5840 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z08D5840Tp_F']
    df['Temperature diff 6982-5482 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z0910D5482Tp_F']
    df['Temperature diff 6982-5653 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z09D5653Tp_F']
    df['Temperature diff 6982-5001 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 6982-4917 ft'] = df['Avg_VW1_Z02D6982Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 6837-6720 ft'] = df['Avg_VW1_Z04D6837Tp_F'] - df['Avg_VW1_Z05D6720Tp_F']
    df['Temperature diff 6837-6632 ft'] = df['Avg_VW1_Z04D6837Tp_F'] - df['Avg_VW1_Z06D6632Tp_F']
    df['Temperature diff 6837-6416 ft'] = df['Avg_VW1_Z04D6837Tp_F'] - df['Avg_VW1_Z07D6416Tp_F']
    df['Temperature diff 6837-5840 ft'] = df['Avg_VW1_Z04D6837Tp_F'] - df['Avg_VW1_Z08D5840Tp_F']
    df['Temperature diff 6837-5482 ft'] = df['Avg_VW1_Z04D6837Tp_F'] - df['Avg_VW1_Z0910D5482Tp_F']
    df['Temperature diff 6837-5653 ft'] = df['Avg_VW1_Z04D6837Tp_F'] - df['Avg_VW1_Z09D5653Tp_F']
    df['Temperature diff 6837-5001 ft'] = df['Avg_VW1_Z04D6837Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 6837-4917 ft'] = df['Avg_VW1_Z04D6837Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 6720-6632 ft'] = df['Avg_VW1_Z05D6720Tp_F'] - df['Avg_VW1_Z06D6632Tp_F']
    df['Temperature diff 6720-6416 ft'] = df['Avg_VW1_Z05D6720Tp_F'] - df['Avg_VW1_Z07D6416Tp_F']
    df['Temperature diff 6720-5840 ft'] = df['Avg_VW1_Z05D6720Tp_F'] - df['Avg_VW1_Z08D5840Tp_F']
    df['Temperature diff 6720-5482 ft'] = df['Avg_VW1_Z05D6720Tp_F'] - df['Avg_VW1_Z0910D5482Tp_F']
    df['Temperature diff 6720-5653 ft'] = df['Avg_VW1_Z05D6720Tp_F'] - df['Avg_VW1_Z09D5653Tp_F']
    df['Temperature diff 6720-5001 ft'] = df['Avg_VW1_Z05D6720Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 6720-4917 ft'] = df['Avg_VW1_Z05D6720Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 6632-6416 ft'] = df['Avg_VW1_Z06D6632Tp_F'] - df['Avg_VW1_Z07D6416Tp_F']
    df['Temperature diff 6632-5840 ft'] = df['Avg_VW1_Z06D6632Tp_F'] - df['Avg_VW1_Z08D5840Tp_F']
    df['Temperature diff 6632-5482 ft'] = df['Avg_VW1_Z06D6632Tp_F'] - df['Avg_VW1_Z0910D5482Tp_F']
    df['Temperature diff 6632-5653 ft'] = df['Avg_VW1_Z06D6632Tp_F'] - df['Avg_VW1_Z09D5653Tp_F']
    df['Temperature diff 6632-5001 ft'] = df['Avg_VW1_Z06D6632Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 6632-4917 ft'] = df['Avg_VW1_Z06D6632Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 6416-5840 ft'] = df['Avg_VW1_Z07D6416Tp_F'] - df['Avg_VW1_Z08D5840Tp_F']
    df['Temperature diff 6416-5482 ft'] = df['Avg_VW1_Z07D6416Tp_F'] - df['Avg_VW1_Z0910D5482Tp_F']
    df['Temperature diff 6416-5653 ft'] = df['Avg_VW1_Z07D6416Tp_F'] - df['Avg_VW1_Z09D5653Tp_F']
    df['Temperature diff 6416-5001 ft'] = df['Avg_VW1_Z07D6416Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 6416-4917 ft'] = df['Avg_VW1_Z07D6416Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 5840-5653 ft'] = df['Avg_VW1_Z08D5840Tp_F'] - df['Avg_VW1_Z09D5653Tp_F']
    df['Temperature diff 5840-5482 ft'] = df['Avg_VW1_Z08D5840Tp_F'] - df['Avg_VW1_Z0910D5482Tp_F']
    df['Temperature diff 5840-5001 ft'] = df['Avg_VW1_Z08D5840Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 5840-4917 ft'] = df['Avg_VW1_Z08D5840Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 5653-5482 ft'] = df['Avg_VW1_Z09D5653Tp_F'] - df['Avg_VW1_Z0910D5482Tp_F']
    df['Temperature diff 5653-5001 ft'] = df['Avg_VW1_Z09D5653Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 5653-4917 ft'] = df['Avg_VW1_Z09D5653Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 5482-5001 ft'] = df['Avg_VW1_Z0910D5482Tp_F'] - df['Avg_VW1_Z10D5001Tp_F']
    df['Temperature diff 5482-4917 ft'] = df['Avg_VW1_Z0910D5482Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    df['Temperature diff 5001-4917 ft'] = df['Avg_VW1_Z10D5001Tp_F'] - df['Avg_VW1_Z11D4917Tp_F']

    return df


def create_lag_features(df, features, n_lags=3):
    """
    Create lag features, percentage change, and difference features for the given dataframe and feature columns.
    """

    lagged_df = df.copy()

    for feature in features:
        for lag in range(1, n_lags + 1):
            # Create lagged features
            lagged_feature = f"{feature}_lag{lag}"
            lagged_df[lagged_feature] = df[feature].shift(lag)

            # Create percentage change features
            pct_change_feature = f"{feature}_pct_change{lag}"
            lagged_df[pct_change_feature] = df[feature].pct_change(lag)

            # Create difference features
            diff_feature = f"{feature}_diff{lag}"
            lagged_df[diff_feature] = df[feature].diff(lag)

    return lagged_df

def create_abs_features(df, features):
    """
    Create absolute value and log absolute value features for the given dataframe and feature columns.
    """

    for col in features:
        df['Abs '+col] = np.abs(df[col])
        df['Log abs '+col] = np.log(np.abs(df[col])+1e-7)

    return df


def create_trend_features(df, features):
    """
    Create exponential moving average (EMA) and simple moving average (SMA) trend features for the given dataframe and feature columns.
    """

    for col in features:
        # Add trend indicators
        df[col+' ema 5'] = ta.ema(df[col], length=5)
        df[col+' sma 5'] = ta.sma(df[col], length=5)

    return df