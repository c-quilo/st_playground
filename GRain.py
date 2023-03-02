import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
import plotly.express as px

st.set_page_config(
    page_title = 'Cesario',
    page_icon = '🌎',
    initial_sidebar_state = 'expanded',
)

st.title('Generating rainfall patterns')
st.markdown('Generative rainfall patterns over South America based on PERSIANN')

PATH = st.secrets['PATH']

if st.button('Generate rainfall samples'):
    result = predict(1, PATH)
    fig = px.imshow(result ,aspect='equal')  
    st.plotly_chart(fig)
    #st.map(result)