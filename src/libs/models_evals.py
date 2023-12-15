from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler
import numpy as np

def LSTM_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    return model


def get_predictions(model: Sequential, scaler: MinMaxScaler, X_test: np.ndarray) -> np.ndarray:
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions


def evaluation(predictions: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler) -> float:
    y_test = scaler.inverse_transform(y_test)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    return rmse

