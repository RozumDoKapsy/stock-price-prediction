from src.constants import PATH_TO_RAW_DATA

from data_utils import download_yahoo_data, save_yahoo_data, load_indices_list
import datetime

import pandas as pd


def get_data():
    indices_list = load_indices_list(PATH_TO_RAW_DATA)

    start = datetime.date(2010, 1, 1)
    end = datetime.datetime.today()

    df_list = []
    for index in indices_list.keys():
        data = download_yahoo_data(index, start, end)
        data['index'] = index
        df_list.append(data)

    indices_data = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
    cols = indices_data.columns.tolist()
    cols = [cols[-1]] + cols[:-1]

    indices_data = indices_data[cols]

    save_yahoo_data(indices_data, PATH_TO_RAW_DATA)
    print('Data successfully donwloaded.')


if __name__ == "__main__":
    get_data()
