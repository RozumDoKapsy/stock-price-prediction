import sys
import os
from pathlib import Path

project_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
sys.path.append(str(project_path))

import yfinance as yf
from pandas_datareader import data as pdr
import datetime
import pandas as pd

import pickle
import json

from src.config import PATH_TO_DATA


def download_yahoo_data(index_id: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Downloads Yahoo data for specific index a transforms them to DataFrame.

    :param index_id: Yahoo Index Symbol
    :param start: starting date for getting data
    :param end: ending date for getting data
    :return: Yahoo Index Data
    """
    yf.pdr_override()
    data = pdr.get_data_yahoo(index_id, start=start, end=end)
    return data


def save_yahoo_data(index_data: pd.DataFrame):
    """
    Saves Index Data to pickle file.

    :param index_data: DataFrame with index data
    """
    path = PATH_TO_DATA
    file_name = 'index_stock_data.pickle'
    with open(os.path.join(path, file_name), 'wb') as pkl:
        pickle.dump(index_data, pkl)


def load_index_list() -> dict[str, str]:
    """
    Loads ID and Name for specific indices from JSON file.

    :return: dict where Index ID is key and Index Name is value
    """
    path = PATH_TO_DATA
    file_name = 'stock_market_index_list.json'
    with open(os.path.join(path, file_name), 'rb') as f:
        indices_list = json.load(f)
    return indices_list


def load_index_data():
    path = PATH_TO_DATA
    file_name = 'index_stock_data.pickle'
    with open(os.path.join(path, file_name), 'rb') as pkl:
        index_data = pickle.load(pkl)
    return index_data
