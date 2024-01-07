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

    if not merged_data.empty:
        rmse = round(mean_squared_error(merged_data['Close'], merged_data['prediction'], squared=False), 1)
        mae = round(mean_absolute_error(merged_data['Close'], merged_data['prediction']), 1)
        mape = round(mean_absolute_percentage_error(merged_data['Close'], merged_data['prediction']) * 100, 1)

        metrics_dict = {
            'day_order': day,
            'metrics_df': {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
        }
        return metrics_dict


if __name__ == '__main__':
    for i in range(10):
        metrics = evaluation('^GSPC', i+1)
        print(metrics)
