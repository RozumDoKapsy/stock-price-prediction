from src.models.model_utils import load_model, evaluate_model
from src.feature_engineering.feature_preprocess import get_features
from src.config import N_FORECAST


def model_evaluation(n_forecast):
    model_name = 'LSTM_model'
    model = load_model(model_name)

    scaler_name = 'Standard_scaler'
    scaler = load_model(scaler_name)

    X_train, y_train, X_test, y_test = get_features(n_forecast)

    rmse = evaluate_model(X_test, y_test, model, scaler)
    print(rmse)
    return rmse


if __name__ == '__main__':
    model_evaluation(N_FORECAST)
