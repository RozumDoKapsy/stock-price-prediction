import sys
import os
from pathlib import Path

project_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.append(str(project_path))

from src.data_preprocessing.data_utils import load_index_data
from src.models.model_utils import load_lstm_model, load_scaler_model, make_prediction
from src.config import LAG, N_FORECAST, FEATURE_NAME, PATH_TO_DATA

import datetime
import pandas as pd


def get_predictions(index_code: str) -> list[dict]:
    model = load_lstm_model('LSTM_model')
    scaler = load_scaler_model('standard_scaler')

    data = load_index_data()
    data = data[data['Index'] == index_code]
    features = data[FEATURE_NAME][-LAG:].values

    predictions = make_prediction(features, model, scaler).flatten()

    # TODO: dates without weekends
    today = datetime.datetime.today()
    prediction_dates = pd.date_range(today.strftime('%Y-%m-%d'), periods=N_FORECAST, name='Date', freq='B')

    prediction_list = []

    for i in range(len(predictions)):
        prediction_dict = {
            'date': prediction_dates[i],
            'prediction': predictions[i]
        }
        prediction_list.append(prediction_dict)
    return prediction_list


def load_predictions() -> pd.DataFrame:
    path = PATH_TO_DATA
    file_name = 'index_stock_predictions.csv'

    predictions_df = pd.read_csv(os.path.join(path, file_name))
    predictions_df['prediction'] = predictions_df['prediction'].astype('float32')
    return predictions_df


def save_predictions(predictions_new: list[dict]):
    today = datetime.datetime.today()

    predictions_df_new = pd.DataFrame(predictions_new)
    predictions_df_new['date_of_prediction'] = today.strftime('%Y-%m-%d')
    predictions_df_new['day_order'] = list(range(1, 11))
    predictions_df_new = predictions_df_new[['date_of_prediction', 'date', 'day_order', 'prediction']]
    predictions_df_new['date'] = predictions_df_new['date'].dt.strftime('%Y-%m-%d')

    predictions_df_old = load_predictions()

    predictions_df = pd.concat([predictions_df_old, predictions_df_new]).reset_index(drop=True)

    path = PATH_TO_DATA
    file_name = 'index_stock_predictions.csv'
    predictions_df.to_csv(os.path.join(path, file_name), index=False)


if __name__ == '__main__':
    predictions = get_predictions('^GSPC')
    save_predictions(predictions)
