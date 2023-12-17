import streamlit as st

import yfinance as yf
from pandas_datareader import data as pdr

from datetime import datetime

from libs import preprocessing
from libs import models_evals

import pandas as pd
import plotly.express as px

st.title('Stock Price Prediction')
yf.pdr_override()


@st.cache_data(show_spinner=False)
def get_time_window():
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    return start, end


@st.cache_data(show_spinner=False)
def get_finance_data(index_code, start, end):
    data = pdr.get_data_yahoo(index_code, start=start, end=end)
    return data


@st.cache_data(show_spinner=False)
def data_preprocessing(df, feature, test_size_pct, n_predictors):
    data, values = preprocessing.select_feature(df, feature)
    scaled_data, scaler = preprocessing.data_scaling(data)
    train_data, test_data = preprocessing.train_test_split(scaled_data, test_size_pct, n_predictors)
    X_train, y_train = preprocessing.extract_predictors_target(train_data, n_predictors)
    X_test, y_test = preprocessing.extract_predictors_target(test_data, n_predictors)
    return X_train, y_train, X_test, y_test, scaler, train_data


@st.cache_data(show_spinner=False)
def create_model(X_train, y_train):
    model = models_evals.LSTM_model(X_train, y_train)
    return model


@st.cache_data(show_spinner=False)
def get_predictions(_model, _scaler, X_test):
    predictions = models_evals.get_predictions(_model, _scaler, X_test)
    return predictions


@st.cache_data(show_spinner=False)
def get_evaluation(predictions, y_test, _scaler):
    evaluation = models_evals.evaluation(predictions, y_test, _scaler)
    return evaluation

# TODO: získat přehled všech ETF fondů (např. S&P50, NASADAQ atd.)
index_code_list = ['^GSPC', '^IXIC']

index_selectbox = st.sidebar.selectbox('Choose Index', index_code_list, index=None, placeholder='Choose index')

start, end = get_time_window()
if index_selectbox is not None:
    data = get_finance_data(index_selectbox, start, end)

    col_selectbox = st.sidebar.selectbox('Choose feature', data.columns, index=None, placeholder='Choose feature')
    test_size_slider = st.sidebar.slider('Choose test size (%)', min_value=0, max_value=100, value=5, step=5)
    n_predictors_input = st.sidebar.number_input('Choose number of predictors', step=10)

    (X_train, y_train, X_test
     , y_test, scaler, train_data) = data_preprocessing(data, col_selectbox
                                                        , test_size_slider/100, int(n_predictors_input))

    model = create_model(X_train, y_train)
    predictions = get_predictions(model, scaler, X_test)
    evaluation = get_evaluation(predictions, y_test, scaler)
    st.write(f'RMSE: {round(evaluation,2)}')

    num_of_rows = len(train_data)

    viz_df = data.filter([col_selectbox])

    viz_df_train = viz_df[:num_of_rows].reset_index()
    viz_df_eval = viz_df[num_of_rows:].reset_index()
    viz_df_eval['predictions'] = predictions

    viz_df_merge = pd.merge(viz_df_train, viz_df_eval, how='outer', on='Date')
    viz_df_merge.columns = ['Date', 'Train', 'Validation', 'Prediction']

    fig = px.line(viz_df_merge, x='Date', y=viz_df_merge.columns[1:], title=f'{col_selectbox} price (USD)')
    st.plotly_chart(fig, use_container_width=True)
