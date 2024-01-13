stock-price-prediction
------

The goal of this project is to create application, that is predicting stock prices for specific market indices using 
LSTM for multiday prediction. The secondary goal is to adopt principles of ML project (MLOps).

The LSTM model is currently trained to make 10-day predictions and it's results can be viewed in [Streamlit application](https://stock-price-prediction-9acqkprqyxq3rrrumizyuy.streamlit.app/).

The application is also monitoring performance of predictions for each day in 10-day sequence using metrics like RMSE, MAE and MAPE.

The flow of new data and predictions is self-sustaining and both are regularly updated with the use of GitHub Actions.

### Planned updates (TO-DO list):

- create API on getting predictions for whole 10-day sequence or specific days
- create predictions for other market indices (currently provides predictions for only S&P500 Index)
- deploy the whole application using tools like Heroku