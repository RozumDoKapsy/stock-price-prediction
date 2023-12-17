import yfinance as yf
from pandas_datareader import data as pdr
import datetime
import pandas as pd

import os
import pickle
import json


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


def save_yahoo_data(index_data: pd.DataFrame, path: os.PathLike):
    """
    Saves Index Data to pickle file.

    :param index_data: DataFrame with index data
    :param path: path to folder where to save the file
    """
    file_name = 'index_data.pickle'
    with open(os.path.join(path, file_name), 'wb') as pkl:
        pickle.dump(index_data, pkl)


def load_indices_list(path: os.PathLike) -> dict[str, str]:
    """
    Loads ID and Name for specific indices from JSON file.

    :param path: path to folder with JSON file
    :return: dict where Index ID is key and Index Name is value
    """
    file_name = 'indices_list.json'
    with open(os.path.join(path, file_name), 'rb') as f:
        indices_list = json.load(f)
    return indices_list
