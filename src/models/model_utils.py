from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

from src.config import PATH_TO_MODELS
import os
import pickle

import numpy as np
from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_model(X_train: np.ndarray, y_train: np.ndarray, n_forecast: int) -> Sequential():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(n_forecast))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=1, epochs=1)
    return model


def save_lstm_model(model: Sequential(), file_name: str):
    file_name = f'{file_name}.h5'
    model.save(os.path.join(PATH_TO_MODELS, file_name))


def load_lstm_model(file_name) -> Sequential():
    file_name = f'{file_name}.h5'
    model = load_model(os.path.join(PATH_TO_MODELS, file_name))
    return model


def load_scaler_model(file_name) -> Union[StandardScaler, MinMaxScaler]:
    file_name = f'{file_name}.pickle'
    with open(os.path.join(PATH_TO_MODELS, file_name), 'rb') as pkl:
        scaler = pickle.load(pkl)
    return scaler


def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, model: Sequential()
                   , scaler: Union[StandardScaler, MinMaxScaler]) -> float:
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
    y_test = scaler.inverse_transform(y_test)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    return rmse


def make_prediction(features: np.ndarray, model: Sequential()
                    , scaler: Union[StandardScaler, MinMaxScaler]) -> float:
    features = features.reshape(-1, 1)
    scaled_feature = scaler.transform(features)
    scaled_feature = scaled_feature.reshape(scaled_feature.shape[1], scaled_feature.shape[0], 1)

    predictions_scaled = model.predict(scaled_feature)
    predictions = scaler.inverse_transform(predictions_scaled)

    return predictions
