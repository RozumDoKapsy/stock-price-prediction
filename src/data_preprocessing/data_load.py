from data_utils import download_yahoo_data, save_yahoo_data, load_index_list
import datetime

import pandas as pd


def get_data():
    indices_list = load_index_list()

    start = datetime.date(2010, 1, 1)
    end = datetime.datetime.today()

    df_list = []
    for index in indices_list.keys():
        data = download_yahoo_data(index, start, end)
        data['Index'] = index
        df_list.append(data)

    indices_data = pd.concat(df_list).reset_index()
    cols = indices_data.columns.tolist()
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]

    indices_data = indices_data[cols]

    save_yahoo_data(indices_data)
    print('Data successfully donwloaded.')


if __name__ == '__main__':
    get_data()
