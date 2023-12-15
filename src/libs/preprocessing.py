from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

import math

import numpy as np
import pandas as pd


def select_feature(df: pd.DataFrame, feature: str) -> (pd.DataFrame, np.ndarray):
    data = df.filter([feature])
    values = data.values
    return data, values


def data_scaling(data: np.ndarray) -> (np.ndarray, MinMaxScaler):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def train_test_split(data: np.ndarray, test_size_pct: float, n_predictors: int
                     , continuity: bool = True) -> (np.ndarray, np.ndarray):
    test_size = math.ceil(len(data) * test_size_pct)

    tscv = TimeSeriesSplit(test_size=test_size)
    for train_index, test_index in tscv.split(data):
        train_data, test_data = data[train_index], data[test_index]

    # because I want to predict values right after training data
    # TimeSeriesSplit will not include data in the rolling window
    if continuity:
        test_data = np.concatenate((train_data[-n_predictors:], test_data))

    return train_data, test_data


def extract_predictors_target(data: np.ndarray, n_predictors: int) -> (np.ndarray, np.ndarray):

    X, y = [], []
    for i in range(len(data) - n_predictors):
        X.append(data[i:(i+n_predictors)])
        y.append(data[i + n_predictors])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y

