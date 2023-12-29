from src.config import PATH_TO_FEATURES, PATH_TO_MODELS

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math

from enum import Enum
from typing import Union


class ScalerType(Enum):
    minmax = 'MinMax'
    standard = 'Standard'


def save_features(feature_data: pd.DataFrame, file_name: str):
    file_name = f'{file_name}.pickle'
    with open(os.path.join(PATH_TO_FEATURES, file_name), 'wb') as pkl:
        pickle.dump(feature_data, pkl)


def load_features(file_name: str) -> pd.DataFrame:
    file_name = f'{file_name}.pickle'
    with open(os.path.join(PATH_TO_FEATURES, file_name), 'rb') as pkl:
        features = pickle.load(pkl)
    return features


def feature_train_test_split(feature: np.ndarray, test_size_pct: float) -> (np.ndarray, np.ndarray):
    test_size = len(feature) - math.ceil(len(feature) * (1 - test_size_pct))

    tscv = TimeSeriesSplit(n_splits=3, test_size=test_size)
    for train_index, test_index in tscv.split(feature):
        train_data, test_data = feature[train_index], feature[test_index]

    return train_data, test_data


def feature_scaling(train_data: np.ndarray, type: ScalerType) -> Union[StandardScaler, MinMaxScaler]:
    if type == ScalerType.standard:
        scaler = StandardScaler()
        scaler = scaler.fit(train_data)
        file_name = 'standard_scaler.pickle'
    elif type == ScalerType.minmax:
        scaler = MinMaxScaler()
        scaler = scaler.fit(train_data)
        file_name = 'minmax_scaler.pickle'

    with open(os.path.join(PATH_TO_MODELS, file_name), 'wb') as pkl:
        pickle.dump(scaler, pkl)

    return scaler


def extract_targets(feature, lag: int, n_forecast: int) -> (np.ndarray, np.ndarray):
    X, y = [], []
    for i in range(len(feature) - lag - n_forecast + 1):
        X.append(feature[i:(i + lag)])
        y.append(feature[(i + lag):(i + lag + n_forecast)])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y
