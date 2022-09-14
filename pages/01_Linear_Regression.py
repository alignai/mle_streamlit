import streamlit as st
import os 
import pandas as pd
import numpy as np
from components.components import build_header

st.set_page_config(
    page_title="Linear Regression",
    page_icon="🚀",
    )

st.sidebar.title("Linear Regression")

build_header("Linear Regression")


if st.session_state.get('data') is not None:
    data = st.session_state.get('data')
else:
    data_file = os.path.join('data', 'space_trip.csv')
    data = pd.read_csv(data_file, index_col = 0)
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace = True)
    data['Time'] = np.arange(len(data.index))
    st.session_state['data'] = data
    

output_markdown = """
    ## Step 2 - Training - Linear Regression Model
    Lets begin our first model, a simple linear regression model to fit the equation below:

    `target = weight_1 * lag_1 + weight_2 * lag_2 + bias`

    Our goals for this section are:
    1. Create Linear Regression Model
    2. Hyperparameter Tuning <- Not Done, Linear Regression doesn't have
    """
st.markdown(output_markdown)
