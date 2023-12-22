from src.feature_engineering.feature_preprocess import get_features
from src.models.model_utils import train_model, save_model


def model_training():
    X_train, y_train, X_test, y_test = get_features()

    model = train_model(X_train, y_train)

    model_name = 'LSTM_model'
    save_model(model, model_name)


if __file__ == '__main__':
    model_training()