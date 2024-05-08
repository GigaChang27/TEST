import streamlit as st
import pandas as pd



pages = st.container(border=False ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("main_page.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/1_data_preprocessing.py")



