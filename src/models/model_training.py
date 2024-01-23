from src.feature_engineering.feature_preprocess import get_features
from src.models.model_utils import train_model, save_lstm_model
from src.config import N_FORECAST


def model_training(n_forecast):
    X_train, y_train, X_test, y_test = get_features(n_forecast)

    model = train_model(X_train, y_train, n_forecast)
    print('Model successfully trained.')

    model_name = 'LSTM_model'
    save_lstm_model(model, model_name)
    print('Model successfully saved.')



if __name__ == '__main__':
    model_training(N_FORECAST)
