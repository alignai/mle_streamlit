import streamlit as st
import os 
import pandas as pd
import numpy as np
import plotly.express as px

from components.components import build_header
from components.ml_functions import plot_time_series, plotly_ts_plot

def run():
    """Runs main page"""
    
    st.set_page_config(
    page_title="MLE EX0",
    page_icon="🚀",
    )

    st.sidebar.title("Machine Learning Engineering")

    build_header("AlignAI")

    # Load the data:
    data_file = os.path.join('data', 'space_trip.csv')
    data = pd.read_csv(data_file, index_col = 0)
    data['Month'] = pd.to_datetime(data['Month'])
    

    output_markdown = """
    ## Step 1: Load Data

    Before we begin, we will need to load in the data we intend to work with, the US Retail Sales Data in this case.

    Our goals here are to:
    1. Read in the data from a csv file
    2. Format the data as a time series
    3. Explore/Visualize the time series

    We use the following code to load `space_trip.csv`:

    ```
    data = pd.read_csv('space_trip.csv', index_col = 0)
    data['Month'] = pd.to_datetime(data['Month'])
    ``` 

    The output of `data.dtypes` is:
    """
    st.markdown(output_markdown)

    st.code(data.dtypes)

    st.markdown('A preview of our data using `data.head()`')
    st.dataframe(data.head())

    output_markdown = """
    ```
    data.set_index('Month', inplace = True)
    data['Time'] = np.arange(len(data.index))
    ```
    """

    st.markdown("Set Datetime index and create a placeholder column for plotting")
    st.markdown(output_markdown)

    data.set_index('Month', inplace = True)
    data['Time'] = np.arange(len(data.index))
    st.session_state['data'] = data

    st.header("Plotting our data")
    st.markdown("We use `maplotlib` and `seaborn` to create `regplots` of our data with the following function:")
    st.markdown("""
    ```
    def plot_time_series(column_to_plot, data):
        fig, ax = plt.subplots(figsize = (10,6))
        ax.plot('Time', column_to_plot, data=data, color='gray', alpha = 0.75)
        ax = sns.regplot(x='Time', y=column_to_plot, data=data, ci=None, scatter_kws=dict(color='0.25'))
        ax.set_title(f'Time Series Plot of Space Trips {column_to_plot} Sales');
        plt.show()
    ```
    """)
    columns_to_plot = ['# of Trips', 'Total Passengers', 'Fuel']
    for i in columns_to_plot:
        fig = plot_time_series(columns_to_plot[0], data)
        st.pyplot(fig)    

    st.header("Plotly versions of plots")
    st.markdown("Code:")
    st.markdown("""
    ```
    df = data.loc[ '1992-01-01':'2019-12-01']
    fig = px.line(df, x=df.index, y = df.columns, template = 'plotly_dark', markers=True)
    fig.show()
    ```
    """)
    df = data.loc[ '1992-01-01':'2019-12-01']
    fig = px.line(df, x=df.index, y = df.columns, template = 'plotly_dark', markers=True)
    st.plotly_chart(fig)

    # Feel Free to adjust and plot the columns you are interested in
    columns_to_plot = ['# of Trips', 'Total Passengers', 'Fuel']

    # For plots
    for i in columns_to_plot:
        fig = plotly_ts_plot(data, i)
        st.plotly_chart(fig)

if __name__ == "__main__":
    run()