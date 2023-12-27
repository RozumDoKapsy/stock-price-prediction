from src.feature_engineering import feature_utils
from src.config import TEST_SIZE_PCT, LAG, N_FORECAST


def get_features(n_forecast):
    feature_file_name = 'index_stock_features'
    features = feature_utils.load_features(feature_file_name)

    index_list = features['Index'].unique()

    features_index = features[features['Index'] == '^GSPC'].drop(columns=['Index'])
    features_index = features_index.set_index('Date').values
    train_data, test_data = feature_utils.feature_train_test_split(features_index, TEST_SIZE_PCT)

    scaler = feature_utils.feature_scaling(train_data, feature_utils.ScalerType.standard)
    train_data_scaled = scaler.transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    X_train, y_train = feature_utils.extract_targets(train_data_scaled, LAG, n_forecast)
    X_test, y_test = feature_utils.extract_targets(test_data_scaled, LAG, n_forecast)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    get_features(N_FORECAST)
