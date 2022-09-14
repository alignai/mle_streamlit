import streamlit as st
import os 
import pandas as pd
import numpy as np

from components.components import build_header

def run():
    """Runs main page"""
    
    st.set_page_config(
    page_title="MLE EX0",
    page_icon="🚀",
    )

    st.sidebar.title("Machine Learning Engineering")

    build_header("AlignAI")

    st.header("Exercise 0")

if __name__ == "__main__":
    run()