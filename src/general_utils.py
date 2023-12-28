from src.data_preprocessing.data_utils import download_yahoo_data
from src.models.model_utils import load_model, make_prediction
from src.config import LAG, N_FORECAST, FEATURE_NAME

import datetime
import pandas as pd


def get_predictions(index_code):
    model = load_model('LSTM_model')
    scaler = load_model('standard_scaler')

    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=100)
    data = download_yahoo_data(index_code, start_date, end_date)
    features = data[FEATURE_NAME][-LAG:].values

    predictions = make_prediction(features, model, scaler).flatten()

    # TODO: dates without weekends
    prediction_dates = pd.date_range(end_date.strftime('%Y-%m-%d'), periods=N_FORECAST, name='Date')

    prediction_list = []

    for i in range(len(predictions)):
        prediction_dict = {
            'date': prediction_dates[i],
            'prediction': predictions[i]
        }
        prediction_list.append(prediction_dict)
    return prediction_list

