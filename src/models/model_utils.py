from keras.models import Sequential
from keras.layers import Dense, LSTM

from src.config import PATH_TO_MODELS
import os
import pickle

import numpy as np
from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=1, epochs=1)
    return model


def save_model(model: Sequential(), file_name: str):
    file_name = f'{file_name}.pickle'
    with open(os.path.join(PATH_TO_MODELS, file_name), 'wb') as pkl:
        pickle.dump(model, pkl)


def load_model(file_name) -> Sequential():
    file_name = f'{file_name}.pickle'
    with open(os.path.join(PATH_TO_MODELS, file_name), 'rb') as pkl:
        model = pickle.load(pkl)
    return model


def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, model: Sequential()
                   , scaler: Union[StandardScaler, MinMaxScaler]) -> float:
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    y_test = scaler.inverse_transform(y_test)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    return rmse
