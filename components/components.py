import streamlit as st
import os
from PIL import Image

def build_header(title):
    img_file = os.path.join('images', 'align_ai.png')
    logo = Image.open(img_file)
    st.session_state['header_img'] = logo
    st.session_state['caption'] = "AlignAI"
    st.session_state['email'] = "<info@ikonosanalytics.com>"
    
    st.image(logo, caption=st.session_state['caption'])

    st.title(title)
    st.markdown('<info@ikonosanalytics.com>')
    
    st.markdown('## [Colab Notebook](https://colab.research.google.com/drive/1EctVHf2Y1w7_HA6aFaEUpFVHqpZrphu3?usp=sharing)')
    return True