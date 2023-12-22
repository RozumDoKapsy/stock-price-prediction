""" This module serves for creating and saving features to feature store."""

from src.data_preprocessing.data_utils import load_index_data
from src.feature_engineering.feature_utils import save_features


def feature_prep():
    data = load_index_data()
    features = data[['Date', 'Index', 'Close']]

    feature_file_name = 'index_stock_features'
    save_features(features, feature_file_name)


if __name__ == '__main__':
    feature_prep()
    