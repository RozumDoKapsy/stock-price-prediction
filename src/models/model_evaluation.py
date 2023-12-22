from src.models.model_utils import load_model, evaluate_model
from src.feature_engineering.feature_preprocess import get_features


def model_evaluation():
    model_name = 'LSTM_model'
    model = load_model(model_name)

    scaler_name = 'Standard_scaler'
    scaler = load_model(scaler_name)

    X_train, y_train, X_test, y_test = get_features()

    rmse = evaluate_model(X_test, y_test, model, scaler)
    return rmse


if __file__ == '__main__':
    model_evaluation()
