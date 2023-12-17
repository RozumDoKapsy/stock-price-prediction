import streamlit as st
from src.data_preprocessing.data_utils import load_indices_data, load_indices_list

st.title('Stock Price Prediction')


@st.cache_data(show_spinner=False)
def load_data():
    data = load_indices_data()
    return data


index_data = load_data()
indices_dict = load_indices_list()

index_list = index_data['index'].unique().tolist()

index_selectbox = st.sidebar.selectbox('Choose index', index_list, index=None, placeholder='Choose index')

if index_selectbox:
    index_name = indices_dict[index_selectbox]
    st.subheader(f'Data of {index_name} Index')
    filtered_data = index_data[index_data['index'] == index_selectbox].reset_index(drop=True)
    st.dataframe(filtered_data)

