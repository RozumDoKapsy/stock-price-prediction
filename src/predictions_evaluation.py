from src.data_preprocessing.data_utils import load_index_data
from src.predictions import load_predictions

import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def evaluation(index: str, day: int):
    real_data = load_index_data()
    real_data = real_data[real_data['Index'] == index]

    predictions = load_predictions()
    predictions = predictions[predictions['day_order'] == day]
    predictions['date'] = pd.to_datetime(predictions['date'])

    merged_data = pd.merge(predictions, real_data[['Date', 'Close']], how='left', left_on='date', right_on='Date')
    merged_data = merged_data[['prediction', 'Close']]
    merged_data = merged_data.dropna()

    rmse = mean_squared_error(merged_data['Close'], merged_data['prediction'], squared=False)
    mae = mean_absolute_error(merged_data['Close'], merged_data['prediction'])
    mape = mean_absolute_percentage_error(merged_data['Close'], merged_data['prediction'])

    return mse, rmse, mae, mape


for i in range(10):
    mse, rmse, mae, mape = evaluation('^GSPC', i+1)
    print(f'Prediction metrics for day {i+1}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'MAPE: {round(mape*100,1)}')

