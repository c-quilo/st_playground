import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title = 'Cesario',
    page_icon = '🌎',
    initial_sidebar_state = 'expanded',
)

# st.title('Uber pickups in NYC')

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache_data
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# # Create a text element and let the reader know the data is loading
# data_load_state = st.text('Loading data ...')
# #Load 10,000 rows of data into the dataframe
# data = load_data(10000)
# # Notify the reader that the data was succesfully loaded.
# data_load_state.text('Loading data ... done!')

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')

# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

# st.bar_chart(hist_values)

# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)

import requests

API_KEY = st.secrets['API_KEY']
API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"

headers = {"Authorization": f"Bearer {API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

with st.form(key='my_form'):
    labels = st.multiselect('Choose your labels', ['bro', 'sis', 'faq'])
    inputs = st.text_area('Enter your text for classification',
            'I want a refund')
    payload = {
        "inputs": inputs,
        "parameters": {"candidate_labels": labels},
    }
    submitted = st.form_submit_button('Classify!')
    if submitted:
        output = query(payload)
        st.write(output)

