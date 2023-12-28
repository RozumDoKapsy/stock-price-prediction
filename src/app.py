import sys
import os
from pathlib import Path

import pandas as pd

project_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.append(str(project_path))

import streamlit as st
from src.data_preprocessing.data_utils import download_yahoo_data, load_index_data, load_index_list
from src.general_utils import get_predictions
from src.config import LAG

import datetime

import plotly.express as px


st.title('Stock Price Prediction')


@st.cache_data(show_spinner=False)
def load_data():
    data = load_index_data()
    return data


@st.cache_data(show_spinner=False)
def create_predictions(index_code):
    predictions = get_predictions(index_code)
    return predictions


index_data = load_data()
indices_dict = load_index_list()

# TODO: část, kde bude predikce na X dní dopředu - 1, 7, 30, 60 (radio button)
# TODO: zabalit do funkce a volat i v rámci API
index = '^GSPC'
index_name = indices_dict[index]

end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=100)
data = download_yahoo_data(index, start_date, end_date)
data = data[-LAG:]

predictions = get_predictions(index)

predictions_df = pd.DataFrame.from_dict(predictions)
predictions_df.index = predictions_df['date']

data_plt = pd.concat([data, predictions_df])
data_plt = data_plt[['Close', 'prediction']].reset_index()
data_plt.columns = ['Date', 'Actual', 'Prediction']

fig = px.line(data_plt, x='Date', y=['Actual', 'Prediction']
              , labels={'value': 'Closing Price (USD)'
                        , 'Actual': 'Last 60 days'
                        , 'Predictions': '10-day Prediction'})


var_names = ['Last 60 days', '10-day Prediction']

for i, name in enumerate(var_names):
    fig.data[i].name = name

st.plotly_chart(fig)


# # TODO: část, kde bude přesnost modelu - metriky, graf
